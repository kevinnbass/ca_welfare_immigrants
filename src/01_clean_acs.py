"""
ACS PUMS data cleaning and indicator creation.

This script:
1. Loads raw ACS PUMS person and housing files
2. Creates welfare program indicators
3. Creates observable immigration status variables
4. Links household and person records
5. Creates covariates for status imputation model
6. Validates data quality

Usage:
    python -m src.01_clean_acs [--year YEAR] [--validate]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import config
from .utils.validation import DataValidator
from .utils.weights import weighted_count

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def find_acs_files(year: int, survey: str = "1-Year") -> dict[str, Path]:
    """
    Find downloaded ACS PUMS files for a given year.

    Args:
        year: Survey year
        survey: Survey type

    Returns:
        Dict with 'person' and 'housing' file paths
    """
    survey_str = survey.lower().replace("-", "")
    base_dir = config.RAW_DATA_DIR / f"acs_{year}_{survey_str}_{config.CA_STATE_ABBR}"

    files = {}

    # Look for person file
    person_patterns = [
        f"psam_p{config.CA_STATE_FIPS}.csv",
        f"ss*p{config.CA_STATE_ABBR}.csv",
        "psam_p*.csv",
    ]
    for pattern in person_patterns:
        matches = list(base_dir.glob(pattern))
        if matches:
            files["person"] = matches[0]
            break

    # Look for housing file
    housing_patterns = [
        f"psam_h{config.CA_STATE_FIPS}.csv",
        f"ss*h{config.CA_STATE_ABBR}.csv",
        "psam_h*.csv",
    ]
    for pattern in housing_patterns:
        matches = list(base_dir.glob(pattern))
        if matches:
            files["housing"] = matches[0]
            break

    return files


def load_acs_person(file_path: Path, usecols: Optional[list] = None) -> pd.DataFrame:
    """
    Load ACS PUMS person file.

    Args:
        file_path: Path to person CSV
        usecols: Optional list of columns to load

    Returns:
        DataFrame with person records
    """
    logger.info(f"Loading ACS person file: {file_path}")

    # Define dtype hints for common columns
    dtype_hints = {
        "SERIALNO": str,
        "SPORDER": int,
        "ST": str,
    }

    df = pd.read_csv(
        file_path,
        usecols=usecols,
        dtype=dtype_hints,
        low_memory=False,
    )

    logger.info(f"Loaded {len(df):,} person records")
    return df


def load_acs_housing(file_path: Path, usecols: Optional[list] = None) -> pd.DataFrame:
    """
    Load ACS PUMS housing file.

    Args:
        file_path: Path to housing CSV
        usecols: Optional list of columns to load

    Returns:
        DataFrame with housing records
    """
    logger.info(f"Loading ACS housing file: {file_path}")

    dtype_hints = {
        "SERIALNO": str,
        "ST": str,
    }

    df = pd.read_csv(
        file_path,
        usecols=usecols,
        dtype=dtype_hints,
        low_memory=False,
    )

    logger.info(f"Loaded {len(df):,} housing records")
    return df


def create_welfare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary welfare program indicators.

    Args:
        df: Person DataFrame with welfare variables

    Returns:
        DataFrame with indicator columns added
    """
    df = df.copy()

    # Medicaid/Medi-Cal (HINS4 = 1 means has Medicaid)
    if config.ACS_MEDICAID in df.columns:
        df["medicaid"] = (df[config.ACS_MEDICAID] == 1).astype(int)
    else:
        logger.warning(f"Column {config.ACS_MEDICAID} not found, setting medicaid to NaN")
        df["medicaid"] = np.nan

    # SSI (SSIP > 0 means received SSI)
    if config.ACS_SSI in df.columns:
        df["ssi"] = (df[config.ACS_SSI].fillna(0) > 0).astype(int)
    else:
        logger.warning(f"Column {config.ACS_SSI} not found, setting ssi to NaN")
        df["ssi"] = np.nan

    # Public Assistance Income (PAP > 0 means received public assistance)
    if config.ACS_PUBLIC_ASSISTANCE in df.columns:
        df["public_assistance"] = (df[config.ACS_PUBLIC_ASSISTANCE].fillna(0) > 0).astype(int)
    else:
        logger.warning(f"Column {config.ACS_PUBLIC_ASSISTANCE} not found")
        df["public_assistance"] = np.nan

    # SNAP is at household level, will be merged from housing file later
    # Placeholder
    df["snap"] = np.nan

    # Any cash benefit (SSI or public assistance)
    df["any_cash"] = ((df["ssi"] == 1) | (df["public_assistance"] == 1)).astype(int)

    logger.info("Created welfare indicators: medicaid, ssi, public_assistance, any_cash")
    return df


def create_observable_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create observable immigration status from NATIVITY and CIT.

    Categories:
    - US_BORN: Born in the US (NATIVITY = 1)
    - NATURALIZED: Foreign-born citizen (NATIVITY = 2, CIT != 5)
    - NONCITIZEN: Not a citizen (CIT = 5)

    Args:
        df: Person DataFrame

    Returns:
        DataFrame with status columns added
    """
    df = df.copy()

    # Initialize status column
    df["observable_status"] = "UNKNOWN"

    # US-born (NATIVITY = 1 means native born)
    if config.ACS_NATIVITY in df.columns:
        us_born_mask = df[config.ACS_NATIVITY] == 1
        df.loc[us_born_mask, "observable_status"] = "US_BORN"

        # Foreign-born
        foreign_born_mask = df[config.ACS_NATIVITY] == 2
    else:
        logger.warning(f"Column {config.ACS_NATIVITY} not found")
        foreign_born_mask = pd.Series(False, index=df.index)

    # Citizenship status
    if config.ACS_CITIZENSHIP in df.columns:
        # Naturalized citizen (foreign-born and citizen)
        # CIT values: 1-4 = citizen, 5 = not a citizen
        citizen_mask = df[config.ACS_CITIZENSHIP].isin([1, 2, 3, 4])
        noncitizen_mask = df[config.ACS_CITIZENSHIP] == 5

        # Naturalized = foreign-born AND citizen
        naturalized_mask = foreign_born_mask & citizen_mask
        df.loc[naturalized_mask, "observable_status"] = "NATURALIZED"

        # Noncitizen
        df.loc[noncitizen_mask, "observable_status"] = "NONCITIZEN"
    else:
        logger.warning(f"Column {config.ACS_CITIZENSHIP} not found")

    # Create binary indicators
    df["is_us_born"] = (df["observable_status"] == "US_BORN").astype(int)
    df["is_naturalized"] = (df["observable_status"] == "NATURALIZED").astype(int)
    df["is_noncitizen"] = (df["observable_status"] == "NONCITIZEN").astype(int)
    df["is_foreign_born"] = (df["observable_status"].isin(["NATURALIZED", "NONCITIZEN"])).astype(int)

    # Log distribution
    status_counts = df["observable_status"].value_counts()
    logger.info(f"Observable status distribution:\n{status_counts}")

    return df


def create_imputation_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create harmonized covariates for status imputation model.

    These variables must be available in both SIPP and ACS.

    Args:
        df: Person DataFrame

    Returns:
        DataFrame with covariate columns added
    """
    df = df.copy()

    # Age groups (using centralized config constants)
    if config.ACS_AGE in df.columns:
        df["age"] = df[config.ACS_AGE]
        df["age_group"] = pd.cut(
            df["age"],
            bins=config.AGE_BINS,
            labels=config.AGE_LABELS,
        )
    else:
        df["age"] = np.nan
        df["age_group"] = np.nan

    # Sex (1 = Male, 2 = Female in ACS)
    if config.ACS_SEX in df.columns:
        df["female"] = (df[config.ACS_SEX] == 2).astype(int)
    else:
        df["female"] = np.nan

    # Education (simplified categories)
    if config.ACS_EDUCATION in df.columns:
        edu = df[config.ACS_EDUCATION].fillna(0)
        df["edu_category"] = "Less than HS"
        df.loc[edu >= 16, "edu_category"] = "HS grad"
        df.loc[edu >= 18, "edu_category"] = "Some college"
        df.loc[edu >= 21, "edu_category"] = "Bachelor's+"
    else:
        df["edu_category"] = np.nan

    # Marital status (simplified)
    if config.ACS_MARITAL_STATUS in df.columns:
        df["married"] = (df[config.ACS_MARITAL_STATUS] == 1).astype(int)
    else:
        df["married"] = np.nan

    # Year of entry (for foreign-born)
    if config.ACS_YEAR_OF_ENTRY in df.columns:
        df["year_of_entry"] = df[config.ACS_YEAR_OF_ENTRY]
        # Calculate years in US (approximate)
        # Note: Need to know survey year
        df["years_in_us"] = np.nan  # Will be calculated based on survey year
    else:
        df["year_of_entry"] = np.nan
        df["years_in_us"] = np.nan

    # English ability (1 = very well, 2 = well, 3 = not well, 4 = not at all)
    if config.ACS_ENGLISH_ABILITY in df.columns:
        eng = df[config.ACS_ENGLISH_ABILITY].fillna(0)
        df["english_well"] = (eng.isin([1, 2])).astype(int)
    else:
        df["english_well"] = np.nan

    # Employment status
    if config.ACS_EMPLOYMENT_STATUS in df.columns:
        esr = df[config.ACS_EMPLOYMENT_STATUS].fillna(0)
        # 1, 2 = employed (civilian); 4, 5 = armed forces
        df["employed"] = (esr.isin([1, 2, 4, 5])).astype(int)
        df["in_labor_force"] = (esr.isin([1, 2, 3, 4, 5])).astype(int)
    else:
        df["employed"] = np.nan
        df["in_labor_force"] = np.nan

    # Poverty ratio
    if config.ACS_POVERTY_RATIO in df.columns:
        povpip = df[config.ACS_POVERTY_RATIO].fillna(-1)
        df["below_poverty"] = ((povpip >= 0) & (povpip < 100)).astype(int)
        df["low_income"] = ((povpip >= 0) & (povpip < 200)).astype(int)
    else:
        df["below_poverty"] = np.nan
        df["low_income"] = np.nan

    # Health insurance
    if config.ACS_HEALTH_INSURANCE in df.columns:
        df["has_insurance"] = (df[config.ACS_HEALTH_INSURANCE] == 1).astype(int)
    else:
        df["has_insurance"] = np.nan

    # Region of birth (simplified)
    if config.ACS_PLACE_OF_BIRTH in df.columns:
        pobp = df[config.ACS_PLACE_OF_BIRTH].fillna(0)
        df["birth_region"] = "US"

        # Mexico
        df.loc[pobp == 303, "birth_region"] = "Mexico"

        # Central America (includes 310-399 range approximately)
        df.loc[(pobp >= 310) & (pobp <= 399), "birth_region"] = "Central America"

        # South America
        df.loc[(pobp >= 360) & (pobp <= 374), "birth_region"] = "South America"

        # Asia
        df.loc[(pobp >= 400) & (pobp <= 499), "birth_region"] = "Asia"

        # Europe
        df.loc[(pobp >= 100) & (pobp <= 199), "birth_region"] = "Europe"

        # Other
        df.loc[(pobp >= 500) & (pobp <= 599), "birth_region"] = "Other"
    else:
        df["birth_region"] = np.nan

    logger.info("Created imputation covariates")
    return df


def merge_household_data(
    person_df: pd.DataFrame,
    housing_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge household-level data to person records.

    Args:
        person_df: Person DataFrame
        housing_df: Housing DataFrame

    Returns:
        Merged DataFrame
    """
    # Select household-level variables to merge
    hh_cols = ["SERIALNO"]

    # SNAP (FS) is at household level
    if config.ACS_SNAP in housing_df.columns:
        hh_cols.append(config.ACS_SNAP)

    # Household size
    if config.ACS_HH_SIZE in housing_df.columns:
        hh_cols.append(config.ACS_HH_SIZE)

    # Household weights
    if config.ACS_HH_WEIGHT in housing_df.columns:
        hh_cols.append(config.ACS_HH_WEIGHT)

    # Add replicate weights if present
    rep_cols = [f"{config.ACS_REP_WEIGHT_PREFIX_HH}{i}" for i in range(1, 81)]
    for col in rep_cols:
        if col in housing_df.columns:
            hh_cols.append(col)

    hh_subset = housing_df[hh_cols].drop_duplicates(subset=["SERIALNO"])

    # Validate merge key exists in both DataFrames
    if "SERIALNO" not in person_df.columns:
        raise KeyError("SERIALNO column missing from person DataFrame")
    if "SERIALNO" not in hh_subset.columns:
        raise KeyError("SERIALNO column missing from housing DataFrame")

    # Check for duplicate keys that could cause unexpected row multiplication
    person_dupes = person_df["SERIALNO"].duplicated().sum()
    housing_dupes = hh_subset["SERIALNO"].duplicated().sum()
    if housing_dupes > 0:
        logger.warning(f"Housing data has {housing_dupes} duplicate SERIALNOs after dedup")

    n_before = len(person_df)

    # Merge
    merged = person_df.merge(hh_subset, on="SERIALNO", how="left")

    # Validate merge result
    n_after = len(merged)
    if n_after != n_before:
        logger.warning(
            f"Merge changed row count: {n_before:,} -> {n_after:,} "
            f"(diff: {n_after - n_before:+,})"
        )

    # Create SNAP indicator at person level
    if config.ACS_SNAP in merged.columns:
        merged["snap"] = (merged[config.ACS_SNAP] == 1).astype(int)
        merged["hh_snap"] = merged["snap"]  # Alias

    logger.info(f"Merged household data: {len(merged):,} records")
    return merged


def create_any_benefit_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create combined 'any benefit' indicator.

    Args:
        df: DataFrame with individual benefit indicators

    Returns:
        DataFrame with any_benefit column added
    """
    df = df.copy()

    benefit_cols = ["medicaid", "snap", "ssi", "public_assistance"]
    available_cols = [c for c in benefit_cols if c in df.columns and df[c].notna().any()]

    if available_cols:
        df["any_benefit"] = (df[available_cols].fillna(0).max(axis=1) > 0).astype(int)
        logger.info(f"Created any_benefit indicator from: {available_cols}")
    else:
        df["any_benefit"] = np.nan
        logger.warning("No benefit columns available for any_benefit indicator")

    return df


def save_processed_data(
    df: pd.DataFrame,
    year: int,
    suffix: str = "",
) -> Path:
    """
    Save processed data to parquet.

    Args:
        df: Processed DataFrame
        year: Survey year
        suffix: Optional filename suffix

    Returns:
        Path to saved file
    """
    filename = f"acs_{year}_ca_processed{suffix}.parquet"
    output_path = config.PROCESSED_DATA_DIR / filename

    df.to_parquet(output_path, index=False)
    logger.info(f"Saved processed data: {output_path} ({len(df):,} records)")

    return output_path


def summarize_data(df: pd.DataFrame, weight_col: str = "PWGTP") -> None:
    """
    Print summary statistics for the processed data.

    Args:
        df: Processed DataFrame
        weight_col: Weight column for weighted statistics
    """
    logger.info("\n" + "=" * 60)
    logger.info("DATA SUMMARY")
    logger.info("=" * 60)

    # Total records and population
    logger.info(f"Total records: {len(df):,}")
    if weight_col in df.columns:
        pop = df[weight_col].sum()
        logger.info(f"Weighted population: {pop:,.0f}")

    # Status distribution
    if "observable_status" in df.columns:
        logger.info("\nObservable Status Distribution (weighted):")
        for status in ["US_BORN", "NATURALIZED", "NONCITIZEN", "UNKNOWN"]:
            mask = df["observable_status"] == status
            if mask.any() and weight_col in df.columns:
                count = df.loc[mask, weight_col].sum()
                pct = count / df[weight_col].sum() * 100
                logger.info(f"  {status}: {count:,.0f} ({pct:.1f}%)")

    # Welfare participation (weighted)
    if weight_col in df.columns:
        logger.info("\nWelfare Participation Rates (overall):")
        for program in ["medicaid", "snap", "ssi", "public_assistance", "any_benefit"]:
            if program in df.columns and df[program].notna().any():
                rate = (df[program] * df[weight_col]).sum() / df[weight_col].sum()
                logger.info(f"  {program}: {rate:.1%}")


def main():
    """Main entry point for ACS data cleaning."""
    parser = argparse.ArgumentParser(description="Clean and process ACS PUMS data")
    parser.add_argument(
        "--year",
        type=int,
        default=2023,
        help="ACS survey year (default: 2023)",
    )
    parser.add_argument(
        "--survey",
        choices=["1-Year", "5-Year"],
        default="1-Year",
        help="ACS survey type",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation checks",
    )
    parser.add_argument(
        "--output-csv",
        action="store_true",
        help="Also save as CSV (in addition to parquet)",
    )

    args = parser.parse_args()

    # Validate year
    try:
        config.validate_year(args.year)
    except ValueError as e:
        logger.error(str(e))
        return 1

    logger.info("=" * 60)
    logger.info(f"ACS PUMS Data Cleaning - {args.year} {args.survey}")
    logger.info("=" * 60)

    # Find data files
    files = find_acs_files(args.year, args.survey)

    if "person" not in files:
        logger.error(f"Person file not found for {args.year}")
        return 1

    # Load person data
    person_df = load_acs_person(files["person"])

    # Create welfare indicators
    person_df = create_welfare_indicators(person_df)

    # Create observable status
    person_df = create_observable_status(person_df)

    # Create imputation covariates
    person_df = create_imputation_covariates(person_df)

    # Calculate years in US
    if "year_of_entry" in person_df.columns:
        person_df["years_in_us"] = args.year - person_df["year_of_entry"]
        person_df.loc[person_df["years_in_us"] < 0, "years_in_us"] = np.nan

    # Merge housing data if available
    if "housing" in files:
        housing_df = load_acs_housing(files["housing"])
        person_df = merge_household_data(person_df, housing_df)
    else:
        logger.warning("Housing file not found, SNAP indicator will be missing")

    # Create combined indicator
    person_df = create_any_benefit_indicator(person_df)

    # Validate if requested
    if args.validate:
        logger.info("\nRunning validation checks...")
        validator = DataValidator(args.year)
        validator.validate_acs_data(person_df)
        logger.info(validator.summary())

    # Save processed data
    output_path = save_processed_data(person_df, args.year)

    if args.output_csv:
        csv_path = output_path.with_suffix(".csv")
        person_df.to_csv(csv_path, index=False)
        logger.info(f"Also saved as CSV: {csv_path}")

    # Print summary
    summarize_data(person_df)

    logger.info("\nACS data cleaning complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
