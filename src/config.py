"""
Configuration constants for the California welfare participation analysis.
"""

from enum import Enum
from pathlib import Path
from typing import Final


class StatusGroup(str, Enum):
    """Immigration status group categories (type-safe enum)."""

    # Observable status (from ACS)
    US_BORN = "US_BORN"
    NATURALIZED = "NATURALIZED"
    NONCITIZEN = "NONCITIZEN"

    # Imputed status (model-derived)
    LEGAL_IMMIGRANT = "LEGAL_IMMIGRANT"
    LEGAL_NONCITIZEN = "LEGAL_NONCITIZEN"
    ILLEGAL = "ILLEGAL"

    # Special values
    UNKNOWN = "UNKNOWN"


# =============================================================================
# DIRECTORY PATHS
# =============================================================================

PROJECT_ROOT: Final[Path] = Path(__file__).parent.parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
EXTERNAL_DATA_DIR: Final[Path] = DATA_DIR / "external"
ADMIN_DATA_DIR: Final[Path] = DATA_DIR / "admin"
BENCHMARKS_DIR: Final[Path] = EXTERNAL_DATA_DIR / "benchmarks"
OUTPUTS_DIR: Final[Path] = PROJECT_ROOT / "outputs"
TABLES_DIR: Final[Path] = OUTPUTS_DIR / "tables"
FIGURES_DIR: Final[Path] = OUTPUTS_DIR / "figures"
POOLED_OUTPUTS_DIR: Final[Path] = OUTPUTS_DIR / "pooled"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"
DOCS_DIR: Final[Path] = PROJECT_ROOT / "docs"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
BOOTSTRAP_MODELS_DIR: Final[Path] = MODELS_DIR / "bootstrap"

# All required directories
_REQUIRED_DIRS: Final[list[Path]] = [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    EXTERNAL_DATA_DIR,
    ADMIN_DATA_DIR,
    BENCHMARKS_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    POOLED_OUTPUTS_DIR,
    MODELS_DIR,
    BOOTSTRAP_MODELS_DIR,
]


def ensure_directories() -> None:
    """
    Create required project directories if they don't exist.

    Call this function explicitly at the start of pipeline scripts
    rather than relying on import-time side effects.
    """
    for dir_path in _REQUIRED_DIRS:
        dir_path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA SOURCE URLS
# =============================================================================

# ACS PUMS
ACS_PUMS_BASE_URL: Final[str] = "https://www2.census.gov/programs-surveys/acs/data/pums"

# SIPP
SIPP_BASE_URL: Final[str] = "https://www2.census.gov/programs-surveys/sipp/data/datasets"

# Pew Research unauthorized immigrant estimates
PEW_STATE_TRENDS_URL: Final[str] = (
    "https://www.pewresearch.org/wp-content/uploads/sites/20/2025/08/"
    "RE_2025.08.21_Unauthorized-immigrants_detailed-tables_state-trends.xlsx"
)
PEW_LABOR_FORCE_URL: Final[str] = (
    "https://www.pewresearch.org/wp-content/uploads/sites/20/2025/08/"
    "RE_2025.08.21_Unauthorized-immigrants_detailed-tables_labor-force-by-state.xlsx"
)

# MPI (may require scraping)
MPI_CA_PROFILE_URL: Final[str] = (
    "https://www.migrationpolicy.org/data/unauthorized-immigrant-population/state/CA"
)

# CHIS (requires registration)
CHIS_PUF_URL: Final[str] = "https://healthpolicy.ucla.edu/our-work/public-use-files"

# =============================================================================
# DATA YEARS
# =============================================================================

# Primary analysis year (try 2024 first, fall back to 2023)
PRIMARY_YEARS: Final[list[int]] = [2024, 2023]  # In order of preference
POOLED_YEARS: Final[list[int]] = [2019, 2020, 2021, 2022, 2023]

# SIPP years to try
SIPP_YEARS: Final[list[int]] = [2024, 2023, 2022]

# =============================================================================
# CALIFORNIA STATE CODE
# =============================================================================

CA_STATE_FIPS: Final[str] = "06"
CA_STATE_ABBR: Final[str] = "ca"
CA_STATE_NAME: Final[str] = "California"

# =============================================================================
# ACS PUMS VARIABLE NAMES (verify with data dictionary for each year)
# =============================================================================

# Person-level identifiers
ACS_PERSON_ID: Final[str] = "SERIALNO"
ACS_PERSON_NUM: Final[str] = "SPORDER"

# Weights
ACS_PERSON_WEIGHT: Final[str] = "PWGTP"
ACS_HH_WEIGHT: Final[str] = "WGTP"
ACS_REP_WEIGHT_PREFIX_PERSON: Final[str] = "PWGTP"  # PWGTP1-PWGTP80
ACS_REP_WEIGHT_PREFIX_HH: Final[str] = "WGTP"  # WGTP1-WGTP80
N_REPLICATE_WEIGHTS: Final[int] = 80

# Demographics
ACS_AGE: Final[str] = "AGEP"
ACS_SEX: Final[str] = "SEX"
ACS_RACE: Final[str] = "RAC1P"
ACS_HISPANIC: Final[str] = "HISP"

# Nativity and citizenship
ACS_NATIVITY: Final[str] = "NATIVITY"  # 1=Native, 2=Foreign born
ACS_CITIZENSHIP: Final[str] = "CIT"  # 1-4=Citizen, 5=Not a citizen
ACS_PLACE_OF_BIRTH: Final[str] = "POBP"
ACS_YEAR_OF_ENTRY: Final[str] = "YOEP"

# Education
ACS_EDUCATION: Final[str] = "SCHL"

# Employment
ACS_EMPLOYMENT_STATUS: Final[str] = "ESR"
ACS_INDUSTRY: Final[str] = "INDP"
ACS_OCCUPATION: Final[str] = "OCCP"

# Language
ACS_ENGLISH_ABILITY: Final[str] = "ENG"

# Marital status
ACS_MARITAL_STATUS: Final[str] = "MAR"

# Income and poverty
ACS_POVERTY_RATIO: Final[str] = "POVPIP"
ACS_TOTAL_INCOME: Final[str] = "PINCP"

# Welfare/benefit variables
ACS_MEDICAID: Final[str] = "HINS4"  # Medicaid/means-tested coverage
ACS_SNAP: Final[str] = "FS"  # SNAP/Food stamps (household level)
ACS_SSI: Final[str] = "SSIP"  # SSI income (amount)
ACS_PUBLIC_ASSISTANCE: Final[str] = "PAP"  # Public assistance income (amount)

# Health insurance
ACS_HEALTH_INSURANCE: Final[str] = "HICOV"

# Household variables (from housing file)
ACS_HH_SIZE: Final[str] = "NP"
ACS_RELATIONSHIP: Final[str] = "RELSHIPP"  # or RELP depending on year

# =============================================================================
# IMMIGRATION STATUS DEFINITIONS
# =============================================================================

# Use StatusGroup enum (defined at top of file) for all status values.
# Available values:
#   Observable: StatusGroup.US_BORN, StatusGroup.NATURALIZED, StatusGroup.NONCITIZEN
#   Imputed: StatusGroup.LEGAL_IMMIGRANT, StatusGroup.LEGAL_NONCITIZEN, StatusGroup.ILLEGAL
#   Special: StatusGroup.UNKNOWN

# =============================================================================
# WELFARE PROGRAM DEFINITIONS
# =============================================================================

WELFARE_PROGRAMS: Final[dict[str, dict]] = {
    # ==========================================================================
    # PERSON-LEVEL PROGRAMS (use PWGTP weights)
    # ==========================================================================
    "medicaid": {
        "variable": ACS_MEDICAID,
        "condition": "== 1",
        "level": "person",
        "unit": "person",
        "estimand_id": "medicaid_person_rate",
        "weight_col": ACS_PERSON_WEIGHT,
        "weight_prefix": ACS_REP_WEIGHT_PREFIX_PERSON,
        "indicator_col": "medicaid",
        "label": "Medicaid/Medi-Cal",
        "period": "current",
    },
    "ssi": {
        "variable": ACS_SSI,
        "condition": "> 0",
        "level": "person",
        "unit": "person",
        "estimand_id": "ssi_person_rate",
        "weight_col": ACS_PERSON_WEIGHT,
        "weight_prefix": ACS_REP_WEIGHT_PREFIX_PERSON,
        "indicator_col": "ssi",
        "label": "SSI",
        "period": "past_12_months",
    },
    "public_assistance": {
        "variable": ACS_PUBLIC_ASSISTANCE,
        "condition": "> 0",
        "level": "person",
        "unit": "person",
        "estimand_id": "public_assistance_person_rate",
        "weight_col": ACS_PERSON_WEIGHT,
        "weight_prefix": ACS_REP_WEIGHT_PREFIX_PERSON,
        "indicator_col": "public_assistance",
        "label": "Public Assistance Income",
        "period": "past_12_months",
    },
    # ==========================================================================
    # SNAP/CALFRESH - HOUSEHOLD-LEVEL ESTIMANDS (use WGTP weights)
    # These compute: % of households receiving SNAP
    # ==========================================================================
    "snap_household_householder": {
        "variable": ACS_SNAP,
        "condition": "== 1",
        "level": "household",
        "unit": "household",
        "estimand_id": "snap_household_rate_householder",
        "weight_col": ACS_HH_WEIGHT,
        "weight_prefix": ACS_REP_WEIGHT_PREFIX_HH,
        "indicator_col": "snap",
        "hh_status_rule": "householder",
        "label": "SNAP/CalFresh (HH Rate, Householder Status)",
        "period": "past_12_months",
    },
    "snap_household_highest_risk": {
        "variable": ACS_SNAP,
        "condition": "== 1",
        "level": "household",
        "unit": "household",
        "estimand_id": "snap_household_rate_highest_risk",
        "weight_col": ACS_HH_WEIGHT,
        "weight_prefix": ACS_REP_WEIGHT_PREFIX_HH,
        "indicator_col": "snap",
        "hh_status_rule": "highest_risk",
        "label": "SNAP/CalFresh (HH Rate, Highest-Risk Status)",
        "period": "past_12_months",
    },
    # ==========================================================================
    # SNAP/CALFRESH - PERSON-LEVEL ESTIMAND (use PWGTP weights)
    # This computes: % of persons living in SNAP-receiving households
    # ==========================================================================
    "snap_person": {
        "variable": ACS_SNAP,
        "condition": "== 1",
        "level": "household",  # Variable is at HH level
        "unit": "person",  # But we estimate at person level
        "estimand_id": "snap_person_in_hh_rate",
        "weight_col": ACS_PERSON_WEIGHT,
        "weight_prefix": ACS_REP_WEIGHT_PREFIX_PERSON,
        "indicator_col": "snap",
        "label": "SNAP/CalFresh (% Persons in HH)",
        "period": "past_12_months",
    },
    # ==========================================================================
    # LEGACY SNAP KEY (deprecated - kept for backwards compatibility)
    # ==========================================================================
    "snap": {
        "variable": ACS_SNAP,
        "condition": "== 1",
        "level": "household",
        "unit": "person",  # Legacy behavior: person weights on HH variable
        "estimand_id": "snap_legacy",
        "weight_col": ACS_PERSON_WEIGHT,
        "weight_prefix": ACS_REP_WEIGHT_PREFIX_PERSON,
        "indicator_col": "snap",
        "label": "SNAP/CalFresh (Legacy)",
        "period": "past_12_months",
        "deprecated": True,  # Use snap_person or snap_household_* instead
    },
}

# =============================================================================
# IMPUTATION SETTINGS
# =============================================================================

N_IMPUTATIONS: Final[int] = 10
RANDOM_SEED: Final[int] = 42

# Model hyperparameters
LOGISTIC_REG_PARAMS: Final[dict] = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
}

GRADIENT_BOOST_PARAMS: Final[dict] = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "min_samples_leaf": 50,
}

# =============================================================================
# CALIBRATION TARGETS (Pew Research, 2023)
# =============================================================================

# Fallback values if Pew data download fails
# Source: Pew Research Center, August 2025 report
# These are approximate and should be updated from downloaded data
PEW_CA_UNAUTHORIZED_2023: Final[int] = 2_200_000  # Approximate
PEW_CA_UNAUTHORIZED_LF_2023: Final[int] = 1_500_000  # Approximate

# =============================================================================
# STATISTICAL THRESHOLDS
# =============================================================================

# Cell suppression rules (Census Bureau standards for ACS data)
# Reference: ACS Accuracy of the Data documentation
MIN_UNWEIGHTED_N: Final[int] = 30  # Standard minimum sample size for reliable estimates
MAX_COEFFICIENT_OF_VARIATION: Final[float] = 0.30  # 30% max CV for publishable estimates

# Model performance thresholds
MIN_MODEL_AUC: Final[float] = 0.65  # Minimum AUC for acceptable discrimination

# =============================================================================
# DEMOGRAPHIC BINS
# =============================================================================

# Age bins for analysis (matches SIPP and ACS categorical breakdowns)
AGE_BINS: Final[list[int]] = [0, 17, 24, 34, 44, 54, 64, 100]
AGE_LABELS: Final[list[str]] = ["0-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]

# =============================================================================
# BIRTH REGION FIPS CODES
# Reference: Census Bureau POBP (Place of Birth) codes
# =============================================================================

FIPS_MEXICO: Final[int] = 303
FIPS_CENTRAL_AMERICA_RANGE: Final[tuple[int, int]] = (310, 399)
FIPS_SOUTH_AMERICA_RANGE: Final[tuple[int, int]] = (360, 374)
FIPS_ASIA_RANGE: Final[tuple[int, int]] = (400, 499)
FIPS_EUROPE_RANGE: Final[tuple[int, int]] = (100, 199)
FIPS_OTHER_RANGE: Final[tuple[int, int]] = (500, 599)

# =============================================================================
# OUTPUT FILE NAMES
# =============================================================================

OUTPUT_RATES_FILE: Final[str] = "ca_rates_by_group_program_year.csv"
OUTPUT_POPULATION_FILE: Final[str] = "ca_population_by_group_year.csv"
OUTPUT_REPORT_FILE: Final[str] = "ca_welfare_by_immigration_status.md"

# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

# Valid year range for ACS data
VALID_YEAR_RANGE: Final[tuple[int, int]] = (2015, 2030)


def validate_year(year: int) -> None:
    """
    Validate that year is within acceptable range.

    Args:
        year: Year to validate

    Raises:
        ValueError: If year is outside valid range
    """
    if not VALID_YEAR_RANGE[0] <= year <= VALID_YEAR_RANGE[1]:
        raise ValueError(
            f"Year {year} is outside valid range [{VALID_YEAR_RANGE[0]}, {VALID_YEAR_RANGE[1]}]"
        )


def validate_required_columns(
    df,  # pd.DataFrame, but avoiding import
    required_columns: list[str],
    context: str = "",
) -> None:
    """
    Validate that required columns exist in DataFrame.

    Args:
        df: DataFrame to check
        required_columns: List of required column names
        context: Context string for error message

    Raises:
        KeyError: If any required column is missing
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        ctx = f" in {context}" if context else ""
        raise KeyError(
            f"Missing required columns{ctx}: {missing}. "
            f"Available columns: {list(df.columns)[:20]}..."
        )


# =============================================================================
# BOOTSTRAP SETTINGS
# =============================================================================

N_BOOTSTRAP_REPLICATES: Final[int] = 100
N_IMPUTATIONS_PER_BOOTSTRAP: Final[int] = 5
BOOTSTRAP_BATCH_SIZE: Final[int] = 20
USE_BOOTSTRAP_UNCERTAINTY: Final[bool] = False  # Default off, enable with --bootstrap

# SIPP sample weight for weighted training
SIPP_PERSON_WEIGHT: Final[str] = "WPFINWGT"
USE_SIPP_SAMPLE_WEIGHTS: Final[bool] = True


# =============================================================================
# UNDERREPORTING ADJUSTMENT
# =============================================================================

ADJUST_FOR_UNDERREPORTING: Final[bool] = False  # Default off

# Administrative-to-survey ratios (from published research)
# Source: Meyer, Mok, Sullivan (2015); USDA QC data; CMS MAX data
UNDERREPORTING_RATIOS: Final[dict[str, dict]] = {
    "snap": {
        "admin_to_survey_ratio": 1.45,  # ~69% capture rate in ACS
        "source": "USDA QC / Census comparison, 2023",
        "uncertainty_bounds": (1.35, 1.55),
    },
    "medicaid": {
        "admin_to_survey_ratio": 1.15,  # ~87% capture rate
        "source": "CMS MAX data / ACS comparison",
        "uncertainty_bounds": (1.10, 1.20),
    },
}


# =============================================================================
# ADMINISTRATIVE DATA SOURCES
# =============================================================================

ADMIN_DATA_SOURCES: Final[dict[str, dict]] = {
    "calfresh": {
        "name": "CalFresh DFA256",
        "description": "Monthly participation and benefit issuance",
        "url": "https://data.ca.gov/dataset/calfresh-data",
        "update_frequency": "monthly",
    },
    "calworks": {
        "name": "CalWORKs",
        "description": "Monthly applications and cases",
        "url": "https://catalog.data.gov/dataset/calworks",
        "update_frequency": "monthly",
    },
    "medi_cal": {
        "name": "Medi-Cal Admin",
        "description": "Adult Full Scope Expansion Programs",
        "url": "https://data.chhs.ca.gov/",
        "update_frequency": "monthly",
    },
    "ssi": {
        "name": "SSA SSI",
        "description": "SSI recipient counts by state/county",
        "url": "https://www.ssa.gov/policy/docs/statcomps/ssi_sc/",
        "update_frequency": "annual",
    },
}


# =============================================================================
# POOLED-YEAR ANALYSIS
# =============================================================================

# Variable harmonization across years (year -> {old_name: new_name})
VARIABLE_HARMONIZATION: Final[dict[int, dict[str, str]]] = {
    2019: {"RELP": "RELSHIPP"},
    2020: {"RELP": "RELSHIPP"},
    # Add year-specific mappings as needed when variable names change
}


# =============================================================================
# CALIBRATION DIAGNOSTICS
# =============================================================================

CALIBRATION_SUBGROUPS: Final[dict[str, list]] = {
    "birth_region": ["Mexico", "Central_America", "South_America", "Asia", "Europe", "Other"],
    "years_in_us": [0, 5, 10, 20, 100],  # Bin edges
    "education": ["Less_than_HS", "HS_grad", "Some_college", "Bachelors_plus"],
}


# =============================================================================
# UNAUTHORIZED BENCHMARK SOURCES
# =============================================================================

UNAUTHORIZED_BENCHMARK_SOURCES: Final[dict[str, str]] = {
    "pew": "Pew Research Center",
    "mpi": "Migration Policy Institute",
    "cms": "Center for Migration Studies",
    "dhs": "Department of Homeland Security",
}
