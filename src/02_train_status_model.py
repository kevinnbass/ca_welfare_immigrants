"""
SIPP-based legal status classification model.

This script:
1. Loads SIPP data with immigration status variables
2. Derives legal status labels (LPR vs non-LPR/likely unauthorized)
3. Trains a classification model
4. Evaluates model performance
5. Saves model for use in ACS imputation

CRITICAL: If SIPP legal status variables are unavailable or inadequate,
this script STOPS and reports findings. It does NOT proceed with fallback.

Usage:
    python -m src.02_train_status_model [--year YEAR]
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class SIPPStatusVariableCheck:
    """Result of checking SIPP for required legal status variables."""

    has_citizenship: bool = False
    has_lpr_status: bool = False
    has_status_at_entry: bool = False
    has_year_lpr: bool = False
    citizenship_var: Optional[str] = None
    lpr_var: Optional[str] = None
    entry_status_var: Optional[str] = None
    year_lpr_var: Optional[str] = None
    can_derive_status: bool = False
    message: str = ""


class SIPPStatusDeriver:
    """
    Derive legal status labels from SIPP data.

    SIPP variable names vary by panel year. This class handles the mapping
    and derives a consistent status label.
    """

    # Known SIPP variable mappings by panel year
    # These need to be verified against actual SIPP codebooks
    VARIABLE_MAPPINGS = {
        2022: {
            "citizenship": "ECITIZEN",  # Citizenship status
            "lpr_status": "AIMSTAT",  # Immigration status
            "entry_status": "TIMSTAT",  # Status at time of immigration
            "year_lpr": "TYRLPR",  # Year became LPR
        },
        2023: {
            "citizenship": "ECITIZEN",
            "lpr_status": "AIMSTAT",
            "entry_status": "TIMSTAT",
            "year_lpr": "TYRLPR",
        },
        2024: {
            "citizenship": "ECITIZEN",
            "lpr_status": "AIMSTAT",
            "entry_status": "TIMSTAT",
            "year_lpr": "TYRLPR",
        },
    }

    def __init__(self, year: int):
        self.year = year
        self.var_map = self.VARIABLE_MAPPINGS.get(year, {})

    def check_variables(self, df: pd.DataFrame) -> SIPPStatusVariableCheck:
        """
        Check which required variables are present in SIPP data.

        Args:
            df: SIPP DataFrame

        Returns:
            SIPPStatusVariableCheck with availability info
        """
        result = SIPPStatusVariableCheck()

        # Check citizenship
        cit_var = self.var_map.get("citizenship")
        if cit_var and cit_var in df.columns:
            result.has_citizenship = True
            result.citizenship_var = cit_var

        # Check LPR status
        lpr_var = self.var_map.get("lpr_status")
        if lpr_var and lpr_var in df.columns:
            result.has_lpr_status = True
            result.lpr_var = lpr_var

        # Check entry status
        entry_var = self.var_map.get("entry_status")
        if entry_var and entry_var in df.columns:
            result.has_status_at_entry = True
            result.entry_status_var = entry_var

        # Check year became LPR
        ylpr_var = self.var_map.get("year_lpr")
        if ylpr_var and ylpr_var in df.columns:
            result.has_year_lpr = True
            result.year_lpr_var = ylpr_var

        # Determine if we can derive status
        # Minimum requirement: citizenship + (lpr_status OR entry_status)
        result.can_derive_status = result.has_citizenship and (
            result.has_lpr_status or result.has_status_at_entry
        )

        if result.can_derive_status:
            result.message = (
                f"SIPP {self.year} has required variables: "
                f"citizenship={result.citizenship_var}, "
                f"lpr={result.lpr_var or 'N/A'}, "
                f"entry={result.entry_status_var or 'N/A'}"
            )
        else:
            missing = []
            if not result.has_citizenship:
                missing.append("citizenship")
            if not (result.has_lpr_status or result.has_status_at_entry):
                missing.append("lpr_status or entry_status")
            result.message = f"SIPP {self.year} MISSING required variables: {', '.join(missing)}"

        return result

    def derive_status_label(
        self,
        df: pd.DataFrame,
        check: SIPPStatusVariableCheck,
    ) -> pd.Series:
        """
        Derive legal status label for noncitizens.

        Categories:
        - LPR: Has lawful permanent resident status
        - NON_LPR_LEGAL: Legal temporary status (visa holders)
        - LIKELY_ILLEGAL: Noncitizen without legal status indicators
        - CITIZEN: U.S. citizen (naturalized or born)
        - UNKNOWN: Cannot determine

        Args:
            df: SIPP DataFrame
            check: Variable availability check

        Returns:
            Series with status labels
        """
        if not check.can_derive_status:
            raise ValueError(f"Cannot derive status: {check.message}")

        status = pd.Series(index=df.index, dtype="object")
        status[:] = "UNKNOWN"

        # Citizens first
        if check.citizenship_var:
            # Typical coding: 1 = citizen, 2 = noncitizen
            # Verify with actual codebook
            citizen_mask = df[check.citizenship_var] == 1
            status[citizen_mask] = "CITIZEN"

        # For noncitizens, check LPR status
        noncitizen_mask = status == "UNKNOWN"

        if check.lpr_var and check.lpr_var in df.columns:
            # Typical coding for immigration status:
            # 1 = LPR/permanent resident
            # 2 = Temporary legal (visa)
            # 3 = Other/unknown
            lpr_mask = noncitizen_mask & (df[check.lpr_var] == 1)
            status[lpr_mask] = "LPR"

            temp_legal_mask = noncitizen_mask & (df[check.lpr_var] == 2)
            status[temp_legal_mask] = "NON_LPR_LEGAL"

        elif check.entry_status_var and check.entry_status_var in df.columns:
            # Use entry status as proxy for legal status
            # Typical SIPP coding for entry status:
            # 1 = Entered as LPR/permanent resident
            # 2 = Entered as refugee or asylee
            # 3 = Other legal entry (student visa, work visa, etc.)
            # Values >= 4 or missing typically indicate uncertain/other entry
            entry = df[check.entry_status_var]
            legal_entry_mask = noncitizen_mask & (entry.isin([1, 2, 3]))
            status[legal_entry_mask] = "NON_LPR_LEGAL"
            logger.debug(
                f"Classified {legal_entry_mask.sum()} noncitizens as legal via entry status"
            )

        # Remaining noncitizens without legal status indicators
        still_unknown = noncitizen_mask & (status == "UNKNOWN")
        status[still_unknown] = "LIKELY_ILLEGAL"

        # Log distribution
        logger.info(f"Derived status distribution:\n{status.value_counts()}")

        return status


def find_sipp_file(year: int) -> Optional[Path]:
    """
    Find SIPP data file for a given year.

    Args:
        year: SIPP panel year

    Returns:
        Path to SIPP data file, or None if not found
    """
    sipp_dir = config.RAW_DATA_DIR / f"sipp_{year}"

    if not sipp_dir.exists():
        return None

    # Look for common file patterns
    patterns = ["*.csv", "*.dat", "*.sas7bdat"]

    for pattern in patterns:
        files = list(sipp_dir.glob(pattern))
        if files:
            return files[0]

    return None


def load_sipp_data(file_path: Path, year: int = 2024) -> pd.DataFrame:
    """
    Load SIPP data file.

    Args:
        file_path: Path to SIPP file
        year: SIPP panel year (for column selection)

    Returns:
        DataFrame with SIPP records
    """
    logger.info(f"Loading SIPP data: {file_path}")

    # Define columns needed for status derivation and covariates
    # Only load required columns to avoid memory issues with large SIPP files
    required_cols = [
        # Status variables
        "ECITIZEN",
        "AIMSTAT",
        "TIMSTAT",
        "TYRLPR",
        # Demographics
        "TAGE",
        "PRTAGE",
        "ESEX",
        "EEDUC",
        "EEDUCATE",
        "EMS",
        # Employment
        "RMESR",
        # Race/ethnicity
        "ERACE",
        "EHISPAN",
        "EORIGIN",
        # Nativity
        "EBORNUS",
        # Health/welfare
        "RCUTYP27",
        # Weights
        config.SIPP_PERSON_WEIGHT,
    ]

    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        # Try to detect delimiter by reading first line
        with open(file_path, "r") as f:
            first_line = f.readline()

        # SIPP files often use pipe delimiter
        if "|" in first_line and first_line.count("|") > first_line.count(","):
            delimiter = "|"
            logger.info("Detected pipe-delimited CSV")
        else:
            delimiter = ","

        # Get available columns
        if delimiter == "|":
            available_cols = first_line.strip().split("|")
        else:
            available_cols = first_line.strip().split(",")

        # Filter to only columns that exist
        cols_to_load = [c for c in required_cols if c in available_cols]
        logger.info(f"Loading {len(cols_to_load)} of {len(available_cols)} columns")

        # Use chunked reading to avoid memory issues with large SIPP files
        chunks = []
        chunk_size = 50000
        for chunk in pd.read_csv(
            file_path,
            delimiter=delimiter,
            usecols=cols_to_load,
            chunksize=chunk_size,
            low_memory=True,
        ):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        del chunks  # Free memory
    elif suffix == ".sas7bdat":
        try:
            import pyreadstat

            df, meta = pyreadstat.read_sas7bdat(file_path)
        except ImportError:
            logger.error("pyreadstat not installed. Install with: pip install pyreadstat")
            raise
    elif suffix == ".dat":
        # Fixed-width format - needs layout file
        logger.error("Fixed-width format not yet supported. Convert to CSV first.")
        raise NotImplementedError("Fixed-width SIPP format not supported")
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    logger.info(f"Loaded {len(df):,} SIPP records")
    return df


def create_harmonized_covariates(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Create harmonized covariates that match ACS coding.

    Args:
        df: SIPP DataFrame
        year: SIPP panel year

    Returns:
        DataFrame with harmonized covariates
    """
    df = df.copy()

    # This is a placeholder - actual variable names depend on SIPP year
    # and need to be verified against the codebook

    # Age
    if "TAGE" in df.columns:
        df["age"] = df["TAGE"]
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 17, 24, 34, 44, 54, 64, 100],
            labels=["0-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
        )
    elif "PRTAGE" in df.columns:
        df["age"] = df["PRTAGE"]

    # Sex
    if "ESEX" in df.columns:
        df["female"] = (df["ESEX"] == 2).astype(int)

    # Education
    if "EEDUCATE" in df.columns:
        edu = df["EEDUCATE"].fillna(0)
        df["edu_category"] = "Less than HS"
        df.loc[edu >= 39, "edu_category"] = "HS grad"
        df.loc[edu >= 40, "edu_category"] = "Some college"
        df.loc[edu >= 43, "edu_category"] = "Bachelor's+"

    # Marital status
    if "EMS" in df.columns:
        df["married"] = (df["EMS"] == 1).astype(int)

    # Employment
    if "RMESR" in df.columns:
        esr = df["RMESR"].fillna(0)
        df["employed"] = (esr.isin([1, 2, 3, 4])).astype(int)
        df["in_labor_force"] = (esr.isin([1, 2, 3, 4, 5, 6, 7])).astype(int)

    # Health insurance - Medicaid/public coverage indicator
    # RCUTYP27: Medicaid coverage type (2014+ panels)
    # EHEESSION: Health insurance session variable (older panels)
    if "RCUTYP27" in df.columns:
        # RCUTYP27 = 1 indicates Medicaid coverage
        df["has_medicaid"] = (df["RCUTYP27"] == 1).astype(int)
        logger.debug("Created has_medicaid from RCUTYP27")
    elif "EHEESSION" in df.columns:
        # EHEESSION = 1 indicates public health coverage
        df["has_medicaid"] = (df["EHEESSION"] == 1).astype(int)
        logger.debug("Created has_medicaid from EHEESSION")
    else:
        # No health insurance variable available - create placeholder
        df["has_medicaid"] = 0
        logger.debug("No health insurance variable found, defaulting has_medicaid to 0")

    logger.info("Created harmonized covariates")
    return df


def prepare_model_features(
    df: pd.DataFrame,
    status_col: str = "status_label",
    use_weights: bool = True,
) -> tuple[pd.DataFrame, pd.Series, Optional[np.ndarray], list, list]:
    """
    Prepare features, target, and weights for model training.

    Args:
        df: DataFrame with covariates and status label
        status_col: Column with status label
        use_weights: Whether to extract SIPP sample weights

    Returns:
        Tuple of (X, y, weights, numeric_cols, categorical_cols)
        weights is None if use_weights=False or weight column not found
    """
    # Define feature columns
    numeric_cols = ["age", "years_in_us"]
    categorical_cols = [
        "female",
        "edu_category",
        "married",
        "employed",
        "in_labor_force",
        "english_well",
        "birth_region",
    ]

    # Filter to available columns
    numeric_cols = [c for c in numeric_cols if c in df.columns and df[c].notna().any()]
    categorical_cols = [c for c in categorical_cols if c in df.columns and df[c].notna().any()]

    if not numeric_cols and not categorical_cols:
        raise ValueError("No valid feature columns found")

    logger.info(f"Numeric features: {numeric_cols}")
    logger.info(f"Categorical features: {categorical_cols}")

    # Filter to noncitizens for model training
    noncitizen_mask = df[status_col].isin(["LPR", "NON_LPR_LEGAL", "LIKELY_ILLEGAL"])

    df_model = df[noncitizen_mask].copy()

    # Create binary target: 1 = likely unauthorized, 0 = legal (LPR or temp)
    df_model["is_illegal"] = (df_model[status_col] == "LIKELY_ILLEGAL").astype(int)

    # Prepare X
    X = df_model[numeric_cols + categorical_cols].copy()

    # Handle missing values
    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())

    for col in categorical_cols:
        X[col] = X[col].fillna("Unknown").astype(str)

    y = df_model["is_illegal"]

    # Extract sample weights if requested and available
    weights = None
    if use_weights and config.USE_SIPP_SAMPLE_WEIGHTS:
        weight_col = config.SIPP_PERSON_WEIGHT
        if weight_col in df_model.columns:
            weights = df_model[weight_col].values
            # Normalize weights to mean 1 for numerical stability
            weights = weights / weights.mean()
            logger.info(f"Using SIPP sample weights from {weight_col}")
        else:
            logger.warning(f"Weight column {weight_col} not found; training without weights")

    return X, y, weights, numeric_cols, categorical_cols


def train_logistic_regression(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: list,
    categorical_cols: list,
    sample_weights: Optional[np.ndarray] = None,
) -> tuple[Pipeline, dict]:
    """
    Train logistic regression model.

    Args:
        X: Feature DataFrame
        y: Target Series
        numeric_cols: Numeric feature column names
        categorical_cols: Categorical feature column names
        sample_weights: Optional SIPP sample weights for weighted training

    Returns:
        Tuple of (fitted pipeline, metrics dict)
    """
    weighted_str = " (weighted)" if sample_weights is not None else ""
    logger.info(f"Training logistic regression model{weighted_str}...")

    # Build preprocessing pipeline
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        drop="first", sparse_output=False, handle_unknown="ignore"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Build full pipeline
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(**config.LOGISTIC_REG_PARAMS)),
        ]
    )

    # Split data (with weights if provided)
    if sample_weights is not None:
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, sample_weights, test_size=0.2, random_state=config.RANDOM_SEED, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=config.RANDOM_SEED, stratify=y
        )
        w_train, _ = None, None  # w_test unused

    # Fit model (with sample weights if provided)
    if w_train is not None:
        pipeline.fit(X_train, y_train, classifier__sample_weight=w_train)
    else:
        pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model_type": "logistic_regression",
        "weighted": sample_weights is not None,
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "class_balance_train": y_train.mean(),
        "class_balance_test": y_test.mean(),
    }

    # Cross-validation AUC
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc")
    metrics["cv_auc_mean"] = cv_scores.mean()
    metrics["cv_auc_std"] = cv_scores.std()

    logger.info(f"Logistic Regression - Test AUC: {metrics['auc']:.3f}")
    logger.info(
        f"Cross-validation AUC: {metrics['cv_auc_mean']:.3f} (+/- {metrics['cv_auc_std']:.3f})"
    )

    # Classification report
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=["Legal", "Unauthorized"]))

    return pipeline, metrics


def train_gradient_boosting(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: list,
    categorical_cols: list,
    sample_weights: Optional[np.ndarray] = None,
) -> tuple[Pipeline, dict]:
    """
    Train gradient boosting model.

    Args:
        X: Feature DataFrame
        y: Target Series
        numeric_cols: Numeric feature column names
        categorical_cols: Categorical feature column names
        sample_weights: Optional SIPP sample weights for weighted training

    Returns:
        Tuple of (fitted pipeline, metrics dict)
    """
    weighted_str = " (weighted)" if sample_weights is not None else ""
    logger.info(f"Training gradient boosting model{weighted_str}...")

    # Build preprocessing pipeline
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        drop="first", sparse_output=False, handle_unknown="ignore"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Build full pipeline
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                GradientBoostingClassifier(
                    **config.GRADIENT_BOOST_PARAMS,
                    random_state=config.RANDOM_SEED,
                ),
            ),
        ]
    )

    # Split data (with weights if provided)
    if sample_weights is not None:
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, sample_weights, test_size=0.2, random_state=config.RANDOM_SEED, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=config.RANDOM_SEED, stratify=y
        )
        w_train, _ = None, None  # w_test unused

    # Fit model (with sample weights if provided)
    if w_train is not None:
        pipeline.fit(X_train, y_train, classifier__sample_weight=w_train)
    else:
        pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model_type": "gradient_boosting",
        "weighted": sample_weights is not None,
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    # Cross-validation AUC
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc")
    metrics["cv_auc_mean"] = cv_scores.mean()
    metrics["cv_auc_std"] = cv_scores.std()

    logger.info(f"Gradient Boosting - Test AUC: {metrics['auc']:.3f}")
    logger.info(
        f"Cross-validation AUC: {metrics['cv_auc_mean']:.3f} (+/- {metrics['cv_auc_std']:.3f})"
    )

    return pipeline, metrics


def save_model(
    pipeline: Pipeline,
    metrics: dict,
    numeric_cols: list,
    categorical_cols: list,
    model_name: str,
) -> Path:
    """
    Save trained model and metadata.

    Args:
        pipeline: Trained pipeline
        metrics: Performance metrics
        numeric_cols: Numeric feature columns
        categorical_cols: Categorical feature columns
        model_name: Name for saved model

    Returns:
        Path to saved model
    """
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save pipeline
    model_path = config.MODELS_DIR / f"{model_name}.joblib"
    joblib.dump(pipeline, model_path)

    # Save metadata
    metadata = {
        "model_name": model_name,
        "metrics": metrics,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
    }

    meta_path = config.MODELS_DIR / f"{model_name}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved model: {model_path}")
    logger.info(f"Saved metadata: {meta_path}")

    return model_path


def main():
    """Main entry point for SIPP model training."""
    parser = argparse.ArgumentParser(description="Train legal status imputation model from SIPP")
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="SIPP panel year (default: try most recent available)",
    )
    parser.add_argument(
        "--model",
        choices=["logistic", "boosting", "both"],
        default="both",
        help="Model type to train",
    )
    parser.add_argument(
        "--weighted",
        action="store_true",
        default=True,
        help="Use SIPP sample weights during training (default: True)",
    )
    parser.add_argument(
        "--no-weighted",
        action="store_false",
        dest="weighted",
        help="Do not use SIPP sample weights during training",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Generate calibration diagnostics (plots and metrics)",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("SIPP Legal Status Model Training")
    logger.info("=" * 60)

    # Find SIPP data
    years_to_try = [args.year] if args.year else config.SIPP_YEARS

    sipp_file = None
    sipp_year = None

    for year in years_to_try:
        sipp_file = find_sipp_file(year)
        if sipp_file:
            sipp_year = year
            break

    if not sipp_file:
        logger.error("=" * 60)
        logger.error("CRITICAL: SIPP DATA NOT FOUND")
        logger.error("=" * 60)
        logger.error(f"Searched for SIPP data in years: {years_to_try}")
        logger.error("The pipeline cannot proceed without SIPP data for status imputation.")
        logger.error("")
        logger.error("Options:")
        logger.error("1. Run 'python -m src.00_fetch_data' to download SIPP data")
        logger.error(
            "2. Manually download SIPP from Census Bureau and place in data/raw/sipp_YEAR/"
        )
        logger.error("3. Review docs/manual_download.md for instructions")
        logger.error("")
        logger.error("PIPELINE STOPPED.")
        return 1

    # Load SIPP data
    logger.info(f"\nUsing SIPP {sipp_year} from: {sipp_file}")
    sipp_df = load_sipp_data(sipp_file)

    # Check for required variables
    deriver = SIPPStatusDeriver(sipp_year)
    var_check = deriver.check_variables(sipp_df)

    logger.info(f"\nVariable check: {var_check.message}")

    if not var_check.can_derive_status:
        logger.error("=" * 60)
        logger.error("CRITICAL: SIPP LEGAL STATUS VARIABLES UNAVAILABLE")
        logger.error("=" * 60)
        logger.error(var_check.message)
        logger.error("")
        logger.error("The SIPP data file was found, but it does not contain the required")
        logger.error("variables to derive legal status (citizenship + LPR status).")
        logger.error("")
        logger.error("This may be because:")
        logger.error("1. The wrong SIPP file was downloaded (need immigration module)")
        logger.error("2. The variable names have changed in this SIPP panel")
        logger.error("3. The immigration topical module is in a separate file")
        logger.error("")
        logger.error("Available columns in SIPP file:")
        logger.error(f"{list(sipp_df.columns)[:20]}...")  # First 20 columns
        logger.error("")
        logger.error("PIPELINE STOPPED. Manual review required.")
        return 1

    # Derive status labels
    logger.info("\nDeriving legal status labels...")
    sipp_df["status_label"] = deriver.derive_status_label(sipp_df, var_check)

    # Create harmonized covariates
    sipp_df = create_harmonized_covariates(sipp_df, sipp_year)

    # Prepare features
    try:
        X, y, weights, numeric_cols, categorical_cols = prepare_model_features(
            sipp_df, use_weights=args.weighted
        )
    except ValueError as e:
        logger.error(f"Failed to prepare features: {e}")
        logger.error("PIPELINE STOPPED.")
        return 1

    logger.info(f"\nTraining data: {len(X):,} noncitizens")
    logger.info(f"Class balance: {y.mean():.1%} unauthorized")

    # Check minimum sample size
    if len(X) < 100:
        logger.error(f"Insufficient sample size: {len(X)} noncitizens")
        logger.error("Need at least 100 noncitizens to train a reliable model.")
        logger.error("PIPELINE STOPPED.")
        return 1

    # Train models
    models = {}
    metrics = {}

    if args.model in ["logistic", "both"]:
        lr_pipeline, lr_metrics = train_logistic_regression(
            X, y, numeric_cols, categorical_cols, sample_weights=weights
        )
        models["logistic"] = lr_pipeline
        metrics["logistic"] = lr_metrics

        # Check AUC threshold
        if lr_metrics["auc"] < config.MIN_MODEL_AUC:
            logger.warning(
                f"Logistic regression AUC ({lr_metrics['auc']:.3f}) below threshold ({config.MIN_MODEL_AUC})"
            )

    if args.model in ["boosting", "both"]:
        gb_pipeline, gb_metrics = train_gradient_boosting(
            X, y, numeric_cols, categorical_cols, sample_weights=weights
        )
        models["boosting"] = gb_pipeline
        metrics["boosting"] = gb_metrics

    # Save models
    logger.info("\nSaving models...")
    for model_name, pipeline in models.items():
        save_model(
            pipeline,
            metrics[model_name],
            numeric_cols,
            categorical_cols,
            f"status_model_{model_name}_{sipp_year}",
        )

    # Generate diagnostics if requested
    if args.diagnostics:
        from .utils.diagnostics import generate_calibration_report

        logger.info("\nGenerating calibration diagnostics...")

        # Get subgroup labels if available
        subgroup_labels = None
        subgroup_name = None
        if "birth_region" in X.columns:
            subgroup_labels = X["birth_region"].values
            subgroup_name = "birth_region"

        for model_name, pipeline in models.items():
            y_prob = pipeline.predict_proba(X)[:, 1]

            generate_calibration_report(
                y_true=y.values,
                y_prob=y_prob,
                sample_weight=weights,
                subgroup_labels=subgroup_labels,
                subgroup_name=subgroup_name or "subgroup",
                output_dir=config.OUTPUTS_FIGURES_DIR,
                model_name=f"status_{model_name}_{sipp_year}",
            )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("MODEL TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"SIPP year: {sipp_year}")
    logger.info(f"Training samples: {len(X):,}")
    logger.info(f"Weighted training: {weights is not None}")

    for model_name, m in metrics.items():
        logger.info(f"\n{model_name.upper()}:")
        logger.info(f"  Test AUC: {m['auc']:.3f}")
        logger.info(f"  CV AUC: {m['cv_auc_mean']:.3f} (+/- {m['cv_auc_std']:.3f})")

    # Recommend primary model
    best_model = max(metrics, key=lambda k: metrics[k]["cv_auc_mean"])
    logger.info(f"\nRecommended primary model: {best_model}")

    logger.info("\nModel training complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
