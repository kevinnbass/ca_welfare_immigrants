"""
Apply legal status imputation to ACS data.

This script:
1. Loads trained status model from SIPP
2. Applies model to ACS noncitizens to get P(undocumented)
3. Creates multiple imputed datasets
4. Calibrates to Pew Research CA undocumented totals
5. Saves imputed datasets for rate estimation

Usage:
    python -m src.03_impute_status_acs [--year YEAR] [--n-imputations N]
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from . import config
from .utils.imputation import (
    create_bernoulli_imputations,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_processed_acs(year: int) -> pd.DataFrame:
    """
    Load processed ACS data.

    Args:
        year: ACS year

    Returns:
        DataFrame with processed ACS data
    """
    file_path = config.PROCESSED_DATA_DIR / f"acs_{year}_ca_processed.parquet"

    if not file_path.exists():
        raise FileNotFoundError(f"Processed ACS file not found: {file_path}")

    logger.info(f"Loading processed ACS: {file_path}")
    df = pd.read_parquet(file_path)
    logger.info(f"Loaded {len(df):,} records")

    return df


def find_model(model_type: str = "logistic") -> tuple[Path, Path]:
    """
    Find trained status model.

    Args:
        model_type: 'logistic' or 'boosting'

    Returns:
        Tuple of (model_path, metadata_path)
    """
    # Search for model files
    pattern = f"status_model_{model_type}_*.joblib"
    models = list(config.MODELS_DIR.glob(pattern))

    if not models:
        raise FileNotFoundError(f"No trained {model_type} model found in {config.MODELS_DIR}")

    # Use most recent if multiple
    model_path = sorted(models)[-1]
    meta_path = model_path.with_name(model_path.stem + "_metadata.json")

    return model_path, meta_path


def load_model(model_path: Path, meta_path: Path) -> tuple:
    """
    Load trained model and metadata.

    Args:
        model_path: Path to model file
        meta_path: Path to metadata file

    Returns:
        Tuple of (pipeline, metadata dict)
    """
    logger.info(f"Loading model: {model_path}")
    pipeline = joblib.load(model_path)

    try:
        with open(meta_path) as f:
            metadata = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model metadata file is corrupted or invalid JSON: {meta_path}. Error: {e}"
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"Model metadata file not found: {meta_path}")

    logger.info(f"Model type: {metadata['model_name']}")
    logger.info(f"Training AUC: {metadata['metrics']['auc']:.3f}")

    return pipeline, metadata


def load_pew_calibration_target(year: int) -> Optional[float]:
    """
    Load Pew Research calibration target for CA unauthorized population.

    Args:
        year: Target year

    Returns:
        Calibration target, or None if not available
    """
    pew_file = config.EXTERNAL_DATA_DIR / "pew_state_trends.xlsx"

    if not pew_file.exists():
        logger.warning(f"Pew file not found: {pew_file}")
        logger.info(f"Using fallback calibration target: {config.PEW_CA_UNAUTHORIZED_2023:,}")
        return config.PEW_CA_UNAUTHORIZED_2023

    try:
        # Read Pew Excel file
        # Structure varies; this is a placeholder
        df = pd.read_excel(pew_file, sheet_name=0)

        # Find California row and appropriate year column
        # This needs to be adjusted based on actual Pew file structure
        ca_row = df[df.iloc[:, 0].str.contains("California", case=False, na=False)]

        if len(ca_row) > 0:
            # Look for year column
            for col in df.columns:
                if str(year) in str(col):
                    value = ca_row[col].values[0]
                    if pd.notna(value):
                        # Convert to number (may be in thousands)
                        if isinstance(value, str):
                            value = float(value.replace(",", ""))
                        logger.info(f"Loaded Pew CA {year} target: {value:,.0f}")
                        return value

        logger.warning("Could not parse Pew file, using fallback")
        return config.PEW_CA_UNAUTHORIZED_2023

    except (ValueError, KeyError) as e:
        logger.warning(f"Error parsing Pew data structure: {e}")
        return config.PEW_CA_UNAUTHORIZED_2023
    except pd.errors.EmptyDataError as e:
        logger.warning(f"Pew file is empty: {e}")
        return config.PEW_CA_UNAUTHORIZED_2023
    except FileNotFoundError as e:
        logger.warning(f"Pew file not found during read: {e}")
        return config.PEW_CA_UNAUTHORIZED_2023
    except PermissionError as e:
        logger.warning(f"Permission denied reading Pew file: {e}")
        return config.PEW_CA_UNAUTHORIZED_2023


def prepare_acs_features(
    df: pd.DataFrame,
    numeric_cols: list,
    categorical_cols: list,
) -> pd.DataFrame:
    """
    Prepare ACS features to match model training features.

    Args:
        df: ACS DataFrame
        numeric_cols: Numeric feature columns from model
        categorical_cols: Categorical feature columns from model

    Returns:
        DataFrame with prepared features

    Raises:
        KeyError: If critical columns are entirely missing
    """
    # Validate that at least some expected columns exist
    all_expected = numeric_cols + categorical_cols
    available = [col for col in all_expected if col in df.columns]
    if len(available) < len(all_expected) * 0.5:
        missing = [col for col in all_expected if col not in df.columns]
        raise KeyError(
            f"Too many feature columns missing ({len(missing)}/{len(all_expected)}). "
            f"Missing: {missing[:10]}..."
        )

    X = df[[col for col in numeric_cols + categorical_cols if col in df.columns]].copy()

    # Handle missing values (same as training)
    for col in numeric_cols:
        if col in X.columns:
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = 0  # Default if column missing

    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].fillna("Unknown").astype(str)
        else:
            X[col] = "Unknown"

    return X


def apply_model_to_acs(
    df: pd.DataFrame,
    pipeline,
    metadata: dict,
) -> pd.DataFrame:
    """
    Apply status model to ACS noncitizens.

    Args:
        df: ACS DataFrame
        pipeline: Trained sklearn pipeline
        metadata: Model metadata

    Returns:
        DataFrame with probability predictions added
    """
    df = df.copy()

    # Identify noncitizens
    noncitizen_mask = df["is_noncitizen"] == 1
    n_noncitizens = noncitizen_mask.sum()
    logger.info(f"Applying model to {n_noncitizens:,} noncitizens")

    # Prepare features
    numeric_cols = metadata["numeric_features"]
    categorical_cols = metadata["categorical_features"]

    X = prepare_acs_features(df[noncitizen_mask], numeric_cols, categorical_cols)

    # Predict probabilities
    probs = pipeline.predict_proba(X)[:, 1]

    # Validate probability bounds - raise error if values are outside [0, 1]
    invalid_mask = (probs < 0) | (probs > 1)
    n_invalid = invalid_mask.sum()
    if n_invalid > 0:
        invalid_values = probs[invalid_mask]
        raise ValueError(
            f"Model produced {n_invalid} probability values outside [0, 1]. "
            f"Range: [{invalid_values.min():.6f}, {invalid_values.max():.6f}]. "
            f"This indicates a model error that must be fixed before proceeding."
        )

    # Add to dataframe
    df["p_unauthorized"] = 0.0
    df.loc[noncitizen_mask, "p_unauthorized"] = probs

    # Summary statistics
    logger.info("P(unauthorized) among noncitizens:")
    logger.info(f"  Mean: {probs.mean():.3f}")
    logger.info(f"  Median: {np.median(probs):.3f}")
    logger.info(f"  Std: {probs.std():.3f}")
    logger.info(f"  Min: {probs.min():.3f}, Max: {probs.max():.3f}")

    return df


def create_imputed_datasets(
    df: pd.DataFrame,
    calibration_target: Optional[float] = None,
    n_imputations: int = 10,
) -> pd.DataFrame:
    """
    Create multiple imputed status datasets.

    Args:
        df: ACS DataFrame with probabilities
        calibration_target: Target total for undocumented (Pew)
        n_imputations: Number of imputations

    Returns:
        DataFrame with imputation columns added
    """
    df = df.copy()

    noncitizen_mask = df["is_noncitizen"] == 1
    probs = df.loc[noncitizen_mask, "p_unauthorized"].values

    # Create imputations
    logger.info(f"Creating {n_imputations} imputations...")
    imputations = create_bernoulli_imputations(
        probs,
        n_imputations=n_imputations,
        random_state=config.RANDOM_SEED,
    )

    # Add to dataframe
    for i in range(n_imputations):
        col_name = f"undoc_imp_{i}"
        df[col_name] = 0
        df.loc[noncitizen_mask, col_name] = imputations[:, i]

    # Calibrate each imputation if target provided
    if calibration_target is not None:
        logger.info(f"Calibrating to target: {calibration_target:,.0f}")

        for i in range(n_imputations):
            col_name = f"undoc_imp_{i}"

            # Calculate current weighted total
            current_total = (df[col_name] * df[config.ACS_PERSON_WEIGHT]).sum()

            if current_total > 0:
                # Calibration ratio
                ratio = calibration_target / current_total

                # Create calibrated weight column
                cal_weight_col = f"weight_cal_{i}"
                df[cal_weight_col] = df[config.ACS_PERSON_WEIGHT].copy()
                df.loc[df[col_name] == 1, cal_weight_col] *= ratio

                new_total = (df[col_name] * df[cal_weight_col]).sum()
                logger.info(f"Imputation {i}: {current_total:,.0f} -> {new_total:,.0f}")

    return df


def create_final_status_column(
    df: pd.DataFrame,
    imputation_index: int,
) -> pd.Series:
    """
    Create final detailed status column for an imputation.

    Categories:
    - US_BORN
    - NATURALIZED
    - LEGAL_NONCITIZEN
    - ILLEGAL

    Args:
        df: DataFrame with imputation columns
        imputation_index: Which imputation to use

    Returns:
        Series with status categories
    """
    imp_col = f"undoc_imp_{imputation_index}"

    status = pd.Series(index=df.index, dtype="object")

    # US-born
    status[df["is_us_born"] == 1] = "US_BORN"

    # Naturalized
    status[df["is_naturalized"] == 1] = "NATURALIZED"

    # Noncitizens: split by imputation
    noncitizen_mask = df["is_noncitizen"] == 1

    if imp_col in df.columns:
        # Legal noncitizens (imputed as legal)
        legal_nc_mask = noncitizen_mask & (df[imp_col] == 0)
        status[legal_nc_mask] = "LEGAL_NONCITIZEN"

        # Illegal (imputed)
        undoc_mask = noncitizen_mask & (df[imp_col] == 1)
        status[undoc_mask] = "ILLEGAL"
    else:
        status[noncitizen_mask] = "NONCITIZEN"

    return status


def create_aggregate_status_column(detailed_status: pd.Series) -> pd.Series:
    """
    Create aggregate 3-way status from detailed status.

    Categories:
    - US_BORN
    - LEGAL_IMMIGRANT (naturalized + legal noncitizen)
    - ILLEGAL

    Args:
        detailed_status: Detailed status series

    Returns:
        Aggregate status series
    """
    agg_status = detailed_status.copy()

    # Combine naturalized and legal noncitizen
    agg_status[detailed_status.isin(["NATURALIZED", "LEGAL_NONCITIZEN"])] = "LEGAL_IMMIGRANT"

    return agg_status


def summarize_imputed_populations(
    df: pd.DataFrame,
    n_imputations: int,
) -> None:
    """
    Print summary of imputed population totals.

    Args:
        df: DataFrame with imputation columns
        n_imputations: Number of imputations
    """
    logger.info("\n" + "=" * 60)
    logger.info("IMPUTED POPULATION SUMMARY")
    logger.info("=" * 60)

    totals = {"US_BORN": [], "NATURALIZED": [], "LEGAL_NONCITIZEN": [], "ILLEGAL": []}

    for i in range(n_imputations):
        status = create_final_status_column(df, i)

        for cat in totals:
            mask = status == cat
            if f"weight_cal_{i}" in df.columns:
                weight_col = f"weight_cal_{i}"
            else:
                weight_col = config.ACS_PERSON_WEIGHT

            total = df.loc[mask, weight_col].sum()
            totals[cat].append(total)

    # Print mean and range for each category
    for cat, values in totals.items():
        mean_val = np.mean(values)
        min_val = np.min(values)
        max_val = np.max(values)
        logger.info(f"{cat}:")
        logger.info(f"  Mean: {mean_val:,.0f}")
        logger.info(f"  Range: [{min_val:,.0f}, {max_val:,.0f}]")


def save_imputed_data(
    df: pd.DataFrame,
    year: int,
    n_imputations: int,
) -> Path:
    """
    Save imputed data.

    Args:
        df: DataFrame with imputations
        year: ACS year
        n_imputations: Number of imputations

    Returns:
        Path to saved file
    """
    # Add status columns for each imputation
    for i in range(n_imputations):
        df[f"status_detail_{i}"] = create_final_status_column(df, i)
        df[f"status_agg_{i}"] = create_aggregate_status_column(df[f"status_detail_{i}"])

    output_path = config.PROCESSED_DATA_DIR / f"acs_{year}_ca_imputed.parquet"

    try:
        df.to_parquet(output_path, index=False)
    except OSError as e:
        raise OSError(f"Failed to save imputed data to {output_path}: {e}")
    except PermissionError as e:
        raise PermissionError(f"Permission denied saving to {output_path}: {e}")

    # Verify file was written successfully
    if not output_path.exists():
        raise IOError(f"File write appeared to succeed but file not found: {output_path}")

    logger.info(f"Saved imputed data: {output_path}")
    return output_path


def main():
    """Main entry point for status imputation."""
    parser = argparse.ArgumentParser(description="Apply status imputation to ACS data")
    parser.add_argument(
        "--year",
        type=int,
        default=2023,
        help="ACS year (default: 2023)",
    )
    parser.add_argument(
        "--model",
        choices=["logistic", "boosting"],
        default="logistic",
        help="Model type to use (default: logistic)",
    )
    parser.add_argument(
        "--n-imputations",
        type=int,
        default=config.N_IMPUTATIONS,
        help=f"Number of imputations (default: {config.N_IMPUTATIONS})",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Skip calibration to Pew totals",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("ACS Status Imputation")
    logger.info("=" * 60)

    # Load processed ACS
    try:
        acs_df = load_processed_acs(args.year)
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("Run 'python -m src.01_clean_acs' first")
        return 1

    # Load trained model
    try:
        model_path, meta_path = find_model(args.model)
        pipeline, metadata = load_model(model_path, meta_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("Run 'python -m src.02_train_status_model' first")
        return 1

    # Load calibration target
    if args.no_calibration:
        calibration_target = None
        logger.info("Calibration disabled")
    else:
        calibration_target = load_pew_calibration_target(args.year)

    # Apply model to ACS
    acs_df = apply_model_to_acs(acs_df, pipeline, metadata)

    # Create imputed datasets
    acs_df = create_imputed_datasets(
        acs_df,
        calibration_target=calibration_target,
        n_imputations=args.n_imputations,
    )

    # Summarize
    summarize_imputed_populations(acs_df, args.n_imputations)

    # Save
    save_imputed_data(acs_df, args.year, args.n_imputations)

    logger.info("\nStatus imputation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
