"""
Analysis unit utilities for household vs person level estimation.

This module provides utilities for correctly handling the distinction
between person-level and household-level estimands, ensuring the correct
weights (PWGTP vs WGTP) are used for each type of analysis.

Key concepts:
- Person-level estimands: % of persons with a characteristic (use PWGTP)
- Household-level estimands: % of households with a characteristic (use WGTP)
- SNAP is a household-level variable but can be analyzed as either:
  - % of households receiving SNAP (household-level with WGTP)
  - % of persons living in SNAP households (person-level with PWGTP)
"""

import logging
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Status hierarchy for "highest_risk" household classification
# Lower number = higher risk
STATUS_RISK_ORDER: dict[str, int] = {
    "ILLEGAL": 0,  # Highest risk
    "LEGAL_NONCITIZEN": 1,
    "LEGAL_IMMIGRANT": 2,
    "NATURALIZED": 3,
    "US_BORN": 4,  # Lowest risk
    "NONCITIZEN": 1,  # Observable status - treat as high risk
    "UNKNOWN": 5,
}


@dataclass
class EstimandSpec:
    """
    Specification for a welfare program estimand.

    Attributes:
        estimand_id: Unique identifier for this estimand
        program_key: Key in WELFARE_PROGRAMS config
        program_variable: ACS variable name for the program indicator
        unit: Analysis unit ('person' or 'household')
        weight_col: Main weight column to use
        weight_prefix: Prefix for replicate weight columns
        indicator_col: Column name for the program indicator
        hh_status_rule: How to classify household status (for household-level)
        label: Human-readable label
        deprecated: Whether this estimand is deprecated
    """

    estimand_id: str
    program_key: str
    program_variable: str
    unit: Literal["person", "household"]
    weight_col: str
    weight_prefix: str
    indicator_col: str
    hh_status_rule: Optional[str] = None  # "householder" or "highest_risk"
    label: str = ""
    deprecated: bool = False


def get_estimand_spec(program_key: str) -> EstimandSpec:
    """
    Get estimand specification from WELFARE_PROGRAMS config.

    Args:
        program_key: Key in WELFARE_PROGRAMS (e.g., "snap_household_householder")

    Returns:
        EstimandSpec with all required information

    Raises:
        KeyError: If program_key not found in config
    """
    from .. import config

    if program_key not in config.WELFARE_PROGRAMS:
        raise KeyError(
            f"Unknown program key: {program_key}. "
            f"Available: {list(config.WELFARE_PROGRAMS.keys())}"
        )

    prog = config.WELFARE_PROGRAMS[program_key]

    # Check for deprecated programs
    if prog.get("deprecated", False):
        import warnings

        warnings.warn(
            f"Program key '{program_key}' is deprecated. "
            f"Use the explicit estimand keys instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    return EstimandSpec(
        estimand_id=prog.get("estimand_id", program_key),
        program_key=program_key,
        program_variable=prog["variable"],
        unit=prog.get("unit", "person"),
        weight_col=prog.get("weight_col", config.ACS_PERSON_WEIGHT),
        weight_prefix=prog.get("weight_prefix", config.ACS_REP_WEIGHT_PREFIX_PERSON),
        indicator_col=prog.get("indicator_col", program_key),
        hh_status_rule=prog.get("hh_status_rule"),
        label=prog.get("label", program_key),
        deprecated=prog.get("deprecated", False),
    )


def prepare_household_level_data(
    df: pd.DataFrame,
    status_col: str,
    hh_status_rule: str = "householder",
    hh_id_col: str = "SERIALNO",
    person_num_col: str = "SPORDER",
) -> pd.DataFrame:
    """
    Prepare data for household-level estimation.

    Reduces person-level data to one row per household with a household-level
    status classification.

    Args:
        df: Person-level DataFrame
        status_col: Status classification column (e.g., "status_agg_0")
        hh_status_rule: How to classify household status
            - "householder": Use householder's (reference person) status
            - "highest_risk": Use highest-risk member's status
        hh_id_col: Household identifier column
        person_num_col: Person number within household column

    Returns:
        Household-level DataFrame with one row per household

    Raises:
        ValueError: If hh_status_rule is not recognized
    """
    if hh_status_rule == "householder":
        # Filter to householders (SPORDER == 1 is reference person)
        hh_df = df[df[person_num_col] == 1].copy()
        logger.debug(
            f"Filtered to {len(hh_df):,} householders from {len(df):,} persons"
        )

    elif hh_status_rule == "highest_risk":
        # Assign risk score and take min (highest risk) per household
        df = df.copy()
        df["_status_risk"] = (
            df[status_col].map(STATUS_RISK_ORDER).fillna(len(STATUS_RISK_ORDER))
        )

        # Get index of highest-risk member per household
        idx = df.groupby(hh_id_col)["_status_risk"].idxmin()
        hh_df = df.loc[idx].copy()
        hh_df = hh_df.drop(columns=["_status_risk"])
        logger.debug(
            f"Selected highest-risk member for {len(hh_df):,} households "
            f"from {len(df):,} persons"
        )

    else:
        raise ValueError(
            f"Unknown hh_status_rule: '{hh_status_rule}'. "
            f"Expected 'householder' or 'highest_risk'"
        )

    return hh_df


def validate_weight_columns(
    df: pd.DataFrame,
    spec: EstimandSpec,
    n_replicate_weights: int = 80,
) -> None:
    """
    Validate that required weight columns exist.

    Args:
        df: DataFrame to validate
        spec: Estimand specification
        n_replicate_weights: Expected number of replicate weights

    Raises:
        KeyError: If required weight columns are missing
    """
    required = [spec.weight_col]
    required += [f"{spec.weight_prefix}{i}" for i in range(1, n_replicate_weights + 1)]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing weight columns for {spec.estimand_id}: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )


def get_all_estimand_specs() -> dict[str, EstimandSpec]:
    """
    Get all non-deprecated estimand specifications.

    Returns:
        Dictionary of estimand_id -> EstimandSpec
    """
    from .. import config

    specs = {}
    for program_key in config.WELFARE_PROGRAMS:
        prog = config.WELFARE_PROGRAMS[program_key]
        if not prog.get("deprecated", False):
            spec = get_estimand_spec(program_key)
            specs[spec.estimand_id] = spec

    return specs


def classify_household_status(
    df: pd.DataFrame,
    status_col: str,
    hh_id_col: str = "SERIALNO",
    method: str = "both",
) -> pd.DataFrame:
    """
    Create household-level status classifications.

    Adds columns for each classification method to the person-level DataFrame.

    Args:
        df: Person-level DataFrame
        status_col: Status classification column
        hh_id_col: Household identifier column
        method: Classification method(s) to compute
            - "householder": Add hh_status_householder column
            - "highest_risk": Add hh_status_highest_risk column
            - "both": Add both columns (default)

    Returns:
        DataFrame with household status columns added
    """
    df = df.copy()

    if method in ("householder", "both"):
        # Get householder's status and merge back
        householder_status = df[df["SPORDER"] == 1][[hh_id_col, status_col]].copy()
        householder_status = householder_status.rename(
            columns={status_col: "hh_status_householder"}
        )
        df = df.merge(householder_status, on=hh_id_col, how="left")

    if method in ("highest_risk", "both"):
        # Get highest-risk member's status
        df["_status_risk"] = (
            df[status_col].map(STATUS_RISK_ORDER).fillna(len(STATUS_RISK_ORDER))
        )
        highest_risk = df.groupby(hh_id_col).apply(
            lambda g: g.loc[g["_status_risk"].idxmin(), status_col],
            include_groups=False,
        )
        highest_risk = highest_risk.reset_index()
        highest_risk.columns = [hh_id_col, "hh_status_highest_risk"]
        df = df.merge(highest_risk, on=hh_id_col, how="left")
        df = df.drop(columns=["_status_risk"])

    return df


def compute_rate_for_estimand(
    df: pd.DataFrame,
    estimand: EstimandSpec,
    status_col: str,
    status_value: str,
    n_replicate_weights: int = 80,
) -> tuple[float, float, float, int, float]:
    """
    Compute rate and variance for a specific estimand.

    Handles both person-level and household-level estimands with correct weights.

    Args:
        df: DataFrame with data
        estimand: Estimand specification
        status_col: Status classification column
        status_value: Status value to filter on
        n_replicate_weights: Number of replicate weights

    Returns:
        Tuple of (rate, variance, se, n_unweighted, n_weighted)
    """
    from .weights import weighted_proportion

    # Prepare data based on unit
    if estimand.unit == "household":
        if estimand.hh_status_rule is None:
            raise ValueError(
                f"Household-level estimand {estimand.estimand_id} requires hh_status_rule"
            )
        work_df = prepare_household_level_data(
            df,
            status_col=status_col,
            hh_status_rule=estimand.hh_status_rule,
        )
    else:
        work_df = df

    # Validate weights exist
    validate_weight_columns(work_df, estimand, n_replicate_weights)

    # Filter to status group
    mask = work_df[status_col] == status_value
    df_group = work_df[mask].copy()

    n_unweighted = len(df_group)
    n_weighted = df_group[estimand.weight_col].sum()

    if n_unweighted == 0:
        logger.debug(f"No records found for status '{status_value}'")
        return np.nan, np.nan, np.nan, 0, 0.0

    # Check for valid indicator values
    indicator_col = estimand.indicator_col
    if indicator_col not in df_group.columns:
        logger.warning(f"Indicator column '{indicator_col}' not in DataFrame")
        return np.nan, np.nan, np.nan, n_unweighted, n_weighted

    # Main estimate
    rate = weighted_proportion(
        df_group[indicator_col],
        df_group[estimand.weight_col],
        validate_binary=False,
    )

    # Replicate estimates for variance
    rep_cols = [f"{estimand.weight_prefix}{i}" for i in range(1, n_replicate_weights + 1)]
    rate_reps = []
    for rep_col in rep_cols:
        if rep_col in df_group.columns:
            rate_r = weighted_proportion(
                df_group[indicator_col],
                df_group[rep_col],
                validate_binary=False,
            )
            rate_reps.append(rate_r)

    if not rate_reps:
        return rate, np.nan, np.nan, n_unweighted, n_weighted

    rate_reps = np.array(rate_reps)

    # SDR variance (4/80 factor for ACS successive difference replication)
    variance = (4.0 / 80.0) * np.sum((rate_reps - rate) ** 2)
    se = np.sqrt(variance)

    return rate, variance, se, n_unweighted, n_weighted
