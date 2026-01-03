"""
Compute welfare participation rates with uncertainty quantification.

This script:
1. Loads imputed ACS data
2. Computes welfare rates by status group for each program
3. Uses replicate weights for survey variance
4. Combines multiple imputations using Rubin's rules
5. Outputs final tables with confidence intervals

Usage:
    python -m src.04_estimate_rates [--year YEAR]
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .utils.analysis_units import (
    EstimandSpec,
    get_estimand_spec,
    prepare_household_level_data,
)
from .utils.imputation import MIResult, combine_mi_results_rubins_rules
from .utils.weights import (
    coefficient_of_variation,
    confidence_interval,
    weighted_proportion,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class RateEstimate:
    """Container for a single rate estimate with uncertainty."""

    year: int
    unit: str  # 'person' or 'household'
    group: str
    program: str
    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    cv: float
    n_unweighted: int
    n_weighted: float
    source: str
    reliable: bool = True
    suppressed: bool = False
    estimand_id: str = ""


def compute_rate_for_estimand(
    df: pd.DataFrame,
    estimand: EstimandSpec,
    status_col: str,
    status_value: str,
    n_replicate_weights: int = 80,
) -> tuple[float, float, float, int, float]:
    """
    Compute rate and variance for a specific estimand with correct weights.

    This function handles both person-level and household-level estimands,
    using the appropriate weights (PWGTP vs WGTP) based on the estimand spec.

    Args:
        df: DataFrame with data
        estimand: EstimandSpec defining the program and weights to use
        status_col: Status classification column
        status_value: Status value to filter on
        n_replicate_weights: Number of replicate weights (default 80)

    Returns:
        Tuple of (rate, variance, se, n_unweighted, n_weighted)
    """
    # Prepare data based on unit (household vs person)
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

    # Filter to status group
    mask = work_df[status_col] == status_value
    df_group = work_df[mask].copy()

    n_unweighted = len(df_group)

    if n_unweighted == 0:
        logger.debug(f"No records found for status '{status_value}'")
        return np.nan, np.nan, np.nan, 0, 0.0

    # Use correct weight columns from estimand spec
    weight_col = estimand.weight_col
    weight_prefix = estimand.weight_prefix
    indicator_col = estimand.indicator_col

    n_weighted = df_group[weight_col].sum()

    # Check for valid indicator values
    if indicator_col not in df_group.columns:
        logger.warning(f"Indicator column '{indicator_col}' not in DataFrame")
        return np.nan, np.nan, np.nan, n_unweighted, n_weighted

    # Check for replicate weights
    rep_cols = [f"{weight_prefix}{i}" for i in range(1, n_replicate_weights + 1)]
    present_cols = [c for c in rep_cols if c in work_df.columns]
    n_rep_weights = len(present_cols)

    if n_rep_weights == 0:
        # No replicate weights - simple estimate without variance
        rate = weighted_proportion(
            df_group[indicator_col], df_group[weight_col], validate_binary=False
        )
        return rate, np.nan, np.nan, n_unweighted, n_weighted

    # Compute main estimate
    rate = weighted_proportion(df_group[indicator_col], df_group[weight_col], validate_binary=False)

    # Replicate estimates for variance
    rate_reps = []
    for rep_col in present_cols:
        if rep_col in df_group.columns:
            rate_r = weighted_proportion(
                df_group[indicator_col], df_group[rep_col], validate_binary=False
            )
            rate_reps.append(rate_r)

    if not rate_reps:
        return rate, np.nan, np.nan, n_unweighted, n_weighted

    rate_reps = np.array(rate_reps)

    # SDR variance (4/80 factor for ACS successive difference replication)
    variance = (4.0 / n_replicate_weights) * np.sum((rate_reps - rate) ** 2)
    se = np.sqrt(variance)

    return rate, variance, se, n_unweighted, n_weighted


def load_imputed_acs(year: int) -> pd.DataFrame:
    """
    Load imputed ACS data.

    Args:
        year: ACS year

    Returns:
        DataFrame with imputed data
    """
    file_path = config.PROCESSED_DATA_DIR / f"acs_{year}_ca_imputed.parquet"

    if not file_path.exists():
        raise FileNotFoundError(f"Imputed ACS file not found: {file_path}")

    logger.info(f"Loading imputed ACS: {file_path}")
    df = pd.read_parquet(file_path)
    logger.info(f"Loaded {len(df):,} records")

    return df


def compute_rate_with_replicate_variance(
    df: pd.DataFrame,
    indicator_col: str,
    status_col: str,
    status_value: str,
    weight_col: str = "PWGTP",
    weight_prefix: str = "PWGTP",
) -> tuple[float, float, float]:
    """
    Compute rate and variance using replicate weights for a specific status group.

    Args:
        df: DataFrame
        indicator_col: Welfare indicator column
        status_col: Status classification column
        status_value: Value to filter on
        weight_col: Main weight column
        weight_prefix: Replicate weight prefix

    Returns:
        Tuple of (estimate, variance, standard_error)
    """
    # Filter to status group
    mask = df[status_col] == status_value
    df_group = df[mask].copy()

    if len(df_group) == 0:
        logger.debug(f"No records found for status '{status_value}'")
        return np.nan, np.nan, np.nan

    # Check for valid indicator values
    if indicator_col not in df_group.columns:
        logger.warning(f"Indicator column '{indicator_col}' not in DataFrame")
        return np.nan, np.nan, np.nan

    indicator_values = df_group[indicator_col].dropna()
    if len(indicator_values) == 0:
        logger.debug(f"All indicator values are NaN for status '{status_value}'")
        return np.nan, np.nan, np.nan

    # Check for replicate weights - strict validation requires exactly 80
    rep_cols = [f"{weight_prefix}{i}" for i in range(1, 81)]
    present_cols = [c for c in rep_cols if c in df.columns]
    n_rep_weights = len(present_cols)

    if n_rep_weights == 0:
        # No replicate weights - simple estimate without variance
        rate = weighted_proportion(
            df_group[indicator_col], df_group[weight_col], validate_binary=False
        )
        return rate, np.nan, np.nan

    if n_rep_weights != 80:
        # Strict validation: require exactly 80 replicate weights
        raise ValueError(
            f"Expected 80 replicate weights, found {n_rep_weights}. "
            f"Missing: {[c for c in rep_cols if c not in df.columns][:5]}..."
        )

    # Main estimate
    rate = weighted_proportion(df_group[indicator_col], df_group[weight_col], validate_binary=False)

    # Replicate estimates
    rate_reps = []
    for rep_col in rep_cols:
        rate_r = weighted_proportion(
            df_group[indicator_col], df_group[rep_col], validate_binary=False
        )
        rate_reps.append(rate_r)

    rate_reps = np.array(rate_reps)

    # SDR variance (4/80 factor for ACS successive difference replication)
    variance = (4.0 / 80.0) * np.sum((rate_reps - rate) ** 2)
    se = np.sqrt(variance)

    return rate, variance, se


def compute_rates_single_imputation(
    df: pd.DataFrame,
    imputation_index: int,
    programs: list[str],
    weight_col: str = "PWGTP",
) -> pd.DataFrame:
    """
    Compute rates for all programs and status groups for a single imputation.

    Uses EstimandSpec to determine correct weights for each program.
    Household-level SNAP estimands use WGTP weights, person-level use PWGTP.

    Args:
        df: DataFrame with imputation columns
        imputation_index: Which imputation to use
        programs: List of program keys (from WELFARE_PROGRAMS or indicator columns)
        weight_col: Default weight column (used for non-estimand programs)

    Returns:
        DataFrame with rate estimates

    Raises:
        KeyError: If required columns are missing
    """
    results = []

    status_col = f"status_agg_{imputation_index}"

    # Validate required columns exist
    config.validate_required_columns(
        df, [status_col, weight_col], context=f"imputation {imputation_index}"
    )

    # Check if calibrated weights exist
    cal_weight_col = f"weight_cal_{imputation_index}"
    has_cal_weights = cal_weight_col in df.columns

    for status_value in ["US_BORN", "LEGAL_IMMIGRANT", "ILLEGAL"]:
        for program in programs:
            # Check if this is a defined estimand in WELFARE_PROGRAMS
            if program in config.WELFARE_PROGRAMS:
                try:
                    estimand = get_estimand_spec(program)

                    # For household-level estimands, use different approach
                    rate, variance, se, n_unweighted, n_weighted = compute_rate_for_estimand(
                        df,
                        estimand=estimand,
                        status_col=status_col,
                        status_value=status_value,
                    )

                    results.append(
                        {
                            "imputation": imputation_index,
                            "group": status_value,
                            "program": program,
                            "rate": rate,
                            "variance": variance,
                            "se": se,
                            "n_unweighted": n_unweighted,
                            "n_weighted": n_weighted,
                            "unit": estimand.unit,
                            "estimand_id": estimand.estimand_id,
                        }
                    )
                    continue

                except Exception as e:
                    logger.warning(f"Error computing estimand {program}: {e}")

            # Fall back to legacy behavior for non-estimand programs
            # (like "any_benefit", "any_cash")
            if program not in df.columns:
                continue

            weight_to_use = cal_weight_col if has_cal_weights else weight_col
            mask = df[status_col] == status_value
            n_unweighted = mask.sum()
            n_weighted = df.loc[mask, weight_to_use].sum()

            rate, variance, se = compute_rate_with_replicate_variance(
                df,
                indicator_col=program,
                status_col=status_col,
                status_value=status_value,
                weight_col=weight_to_use,
            )

            results.append(
                {
                    "imputation": imputation_index,
                    "group": status_value,
                    "program": program,
                    "rate": rate,
                    "variance": variance,
                    "se": se,
                    "n_unweighted": n_unweighted,
                    "n_weighted": n_weighted,
                    "unit": "person",  # Default for legacy programs
                    "estimand_id": program,
                }
            )

    return pd.DataFrame(results)


def combine_imputations(
    results_df: pd.DataFrame,
    n_imputations: int,
) -> pd.DataFrame:
    """
    Combine results across imputations using Rubin's rules.

    Args:
        results_df: DataFrame with per-imputation results
        n_imputations: Number of imputations

    Returns:
        DataFrame with combined estimates
    """
    combined = []

    # Group by group and program
    for (group, program), group_df in results_df.groupby(["group", "program"]):
        estimates = group_df["rate"].values
        variances = group_df["variance"].values

        # Handle NaN variances
        if np.any(np.isnan(variances)):
            # Use just between-imputation variance
            mi_result = MIResult(
                estimate=np.nanmean(estimates),
                within_variance=0,
                between_variance=np.nanvar(estimates, ddof=1),
                total_variance=np.nanvar(estimates, ddof=1) * (1 + 1 / n_imputations),
                se=np.sqrt(np.nanvar(estimates, ddof=1) * (1 + 1 / n_imputations)),
                ci_lower=np.nan,
                ci_upper=np.nan,
                df=n_imputations - 1,
                n_imputations=n_imputations,
                fraction_missing_info=1.0,
            )
            # Compute CI
            if not np.isnan(mi_result.se):
                mi_result.ci_lower, mi_result.ci_upper = confidence_interval(
                    mi_result.estimate, mi_result.se
                )
        else:
            mi_result = combine_mi_results_rubins_rules(
                list(estimates),
                list(variances),
            )

        combined.append(
            {
                "group": group,
                "program": program,
                "estimate": mi_result.estimate,
                "se": mi_result.se,
                "ci_lower": mi_result.ci_lower,
                "ci_upper": mi_result.ci_upper,
                "within_var": mi_result.within_variance,
                "between_var": mi_result.between_variance,
                "total_var": mi_result.total_variance,
                "fmi": mi_result.fraction_missing_info,
                "n_unweighted": group_df["n_unweighted"].mean(),
                "n_weighted": group_df["n_weighted"].mean(),
            }
        )

    return pd.DataFrame(combined)


def compute_observable_rates(
    df: pd.DataFrame,
    programs: list[str],
    weight_col: str = "PWGTP",
) -> pd.DataFrame:
    """
    Compute rates by observable status (without imputation).

    Uses EstimandSpec to determine correct weights for each program.
    Household-level SNAP estimands use WGTP weights, person-level use PWGTP.

    Args:
        df: DataFrame with observable status
        programs: List of program keys (from WELFARE_PROGRAMS or indicator columns)
        weight_col: Default weight column (used for non-estimand programs)

    Returns:
        DataFrame with rate estimates
    """
    results = []
    status_col = "observable_status"

    for status_value in ["US_BORN", "NATURALIZED", "NONCITIZEN"]:
        for program in programs:
            # Check if this is a defined estimand in WELFARE_PROGRAMS
            if program in config.WELFARE_PROGRAMS:
                try:
                    estimand = get_estimand_spec(program)

                    # Use estimand-aware computation with correct weights
                    rate, variance, se, n_unweighted, n_weighted = compute_rate_for_estimand(
                        df,
                        estimand=estimand,
                        status_col=status_col,
                        status_value=status_value,
                    )

                    # Confidence interval
                    if not np.isnan(se):
                        ci_low, ci_high = confidence_interval(rate, se)
                        cv = coefficient_of_variation(rate, se)
                    else:
                        ci_low, ci_high = np.nan, np.nan
                        cv = np.nan

                    results.append(
                        {
                            "group": status_value,
                            "program": program,
                            "estimate": rate,
                            "se": se,
                            "ci_lower": ci_low,
                            "ci_upper": ci_high,
                            "cv": cv,
                            "n_unweighted": n_unweighted,
                            "n_weighted": n_weighted,
                            "source": "observable",
                            "unit": estimand.unit,
                            "estimand_id": estimand.estimand_id,
                        }
                    )
                    continue

                except Exception as e:
                    logger.warning(f"Error computing observable estimand {program}: {e}")

            # Fall back to legacy behavior for non-estimand programs
            if program not in df.columns:
                continue

            mask = df[status_col] == status_value
            n_unweighted = mask.sum()
            n_weighted = df.loc[mask, weight_col].sum()

            rate, variance, se = compute_rate_with_replicate_variance(
                df,
                indicator_col=program,
                status_col=status_col,
                status_value=status_value,
                weight_col=weight_col,
            )

            # Confidence interval
            if not np.isnan(se):
                ci_low, ci_high = confidence_interval(rate, se)
                cv = coefficient_of_variation(rate, se)
            else:
                ci_low, ci_high = np.nan, np.nan
                cv = np.nan

            results.append(
                {
                    "group": status_value,
                    "program": program,
                    "estimate": rate,
                    "se": se,
                    "ci_lower": ci_low,
                    "ci_upper": ci_high,
                    "cv": cv,
                    "n_unweighted": n_unweighted,
                    "n_weighted": n_weighted,
                    "source": "observable",
                    "unit": "person",
                    "estimand_id": program,
                }
            )

    return pd.DataFrame(results)


def apply_suppression_rules(
    df: pd.DataFrame,
    min_n: int = config.MIN_UNWEIGHTED_N,
    max_cv: float = config.MAX_COEFFICIENT_OF_VARIATION,
) -> pd.DataFrame:
    """
    Apply cell suppression rules for reliability.

    Args:
        df: DataFrame with estimates
        min_n: Minimum unweighted sample size
        max_cv: Maximum coefficient of variation

    Returns:
        DataFrame with suppression flags
    """
    df = df.copy()

    df["reliable"] = True
    df["suppressed"] = False

    # Sample size check
    small_n = df["n_unweighted"] < min_n
    df.loc[small_n, "reliable"] = False
    df.loc[small_n, "suppressed"] = True

    # CV check
    df["cv"] = np.where(
        df["estimate"] != 0,
        np.abs(df["se"] / df["estimate"]),
        np.inf,
    )
    high_cv = df["cv"] > max_cv
    df.loc[high_cv, "reliable"] = False

    # Log suppressions
    n_suppressed = df["suppressed"].sum()
    n_unreliable = (~df["reliable"]).sum()
    logger.info(f"Suppressed cells (n < {min_n}): {n_suppressed}")
    logger.info(f"Unreliable cells (CV > {max_cv:.0%}): {n_unreliable}")

    return df


def format_output_table(
    df: pd.DataFrame,
    year: int,
    source: str = "imputed",
) -> pd.DataFrame:
    """
    Format results for output.

    Args:
        df: DataFrame with estimates
        year: ACS year
        source: Data source description

    Returns:
        Formatted DataFrame
    """
    df = df.copy()

    df["year"] = year
    # Use unit from data if available, otherwise default to "person"
    if "unit" not in df.columns:
        df["unit"] = "person"
    df["source"] = source

    # Rename programs to labels
    program_labels = {
        "medicaid": "Medicaid/Medi-Cal",
        "snap": "SNAP/CalFresh",
        "snap_person": "SNAP/CalFresh (% Persons in HH)",
        "snap_household_householder": "SNAP/CalFresh (HH Rate, Householder)",
        "snap_household_highest_risk": "SNAP/CalFresh (HH Rate, Highest-Risk)",
        "ssi": "SSI",
        "public_assistance": "Public Assistance",
        "any_benefit": "Any Benefit",
        "any_cash": "Any Cash Benefit",
    }
    df["program_label"] = df["program"].map(program_labels).fillna(df["program"])

    # Reorder columns
    cols = [
        "year",
        "unit",
        "group",
        "program",
        "program_label",
        "estimate",
        "se",
        "ci_lower",
        "ci_upper",
        "cv",
        "n_unweighted",
        "n_weighted",
        "reliable",
        "suppressed",
        "source",
    ]
    cols = [c for c in cols if c in df.columns]

    return df[cols]


def save_results(
    df: pd.DataFrame,
    year: int,
    suffix: str = "",
) -> Path:
    """
    Save results to CSV.

    Args:
        df: Results DataFrame
        year: ACS year
        suffix: Filename suffix

    Returns:
        Path to saved file
    """
    filename = f"ca_rates_by_group_program_{year}{suffix}.csv"
    output_path = config.TABLES_DIR / filename

    df.to_csv(output_path, index=False)
    logger.info(f"Saved results: {output_path}")

    return output_path


def print_summary_table(df: pd.DataFrame) -> None:
    """
    Print formatted summary table to console.

    Args:
        df: Results DataFrame
    """
    logger.info("\n" + "=" * 80)
    logger.info("WELFARE PARTICIPATION RATES BY IMMIGRATION STATUS")
    logger.info("=" * 80)

    programs = df["program"].unique()

    for program in programs:
        prog_df = df[df["program"] == program]
        label = prog_df["program_label"].iloc[0] if "program_label" in prog_df.columns else program

        logger.info(f"\n{label}:")
        logger.info("-" * 50)

        for _, row in prog_df.iterrows():
            group = row["group"]
            est = row["estimate"] * 100
            ci_low = row["ci_lower"] * 100 if pd.notna(row["ci_lower"]) else np.nan
            ci_high = row["ci_upper"] * 100 if pd.notna(row["ci_upper"]) else np.nan

            if row.get("suppressed", False):
                logger.info(f"  {group:20s}: [SUPPRESSED]")
            elif pd.isna(est):
                logger.info(f"  {group:20s}: N/A")
            else:
                reliable_marker = "" if row.get("reliable", True) else " *"
                logger.info(
                    f"  {group:20s}: {est:5.1f}% ({ci_low:5.1f}% - {ci_high:5.1f}%){reliable_marker}"
                )

    logger.info("\n* Estimate may be unreliable (high CV)")


def main():
    """Main entry point for rate estimation."""
    parser = argparse.ArgumentParser(description="Estimate welfare participation rates")
    parser.add_argument(
        "--year",
        type=int,
        default=2023,
        help="ACS year (default: 2023)",
    )
    parser.add_argument(
        "--observable-only",
        action="store_true",
        help="Only compute observable status rates (no imputation)",
    )
    parser.add_argument(
        "--n-imputations",
        type=int,
        default=config.N_IMPUTATIONS,
        help="Number of imputations to use",
    )
    parser.add_argument(
        "--include-household",
        action="store_true",
        help="Include household-level SNAP estimands (using WGTP weights)",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"Rate Estimation - {args.year}")
    logger.info("=" * 60)

    # Define programs - use legacy keys for backward compatibility
    programs = ["medicaid", "snap", "ssi", "public_assistance", "any_benefit", "any_cash"]

    # Add household-level SNAP estimands if requested
    if args.include_household:
        programs.extend(
            [
                "snap_household_householder",
                "snap_household_highest_risk",
                "snap_person",  # Explicit person-in-SNAP-household
            ]
        )
        logger.info("Including household-level SNAP estimands (using WGTP weights)")

    # Determine which file to load
    if args.observable_only:
        file_path = config.PROCESSED_DATA_DIR / f"acs_{args.year}_ca_processed.parquet"
    else:
        file_path = config.PROCESSED_DATA_DIR / f"acs_{args.year}_ca_imputed.parquet"

    if not file_path.exists():
        logger.error(f"Data file not found: {file_path}")
        if args.observable_only:
            logger.error("Run 'python -m src.01_clean_acs' first")
        else:
            logger.error("Run 'python -m src.03_impute_status_acs' first")
        return 1

    df = pd.read_parquet(file_path)
    logger.info(f"Loaded {len(df):,} records from {file_path}")

    # Always compute observable rates
    logger.info("\nComputing observable status rates...")
    observable_results = compute_observable_rates(df, programs)
    observable_results = apply_suppression_rules(observable_results)
    observable_results = format_output_table(observable_results, args.year, source="observable")
    save_results(observable_results, args.year, suffix="_observable")

    print_summary_table(observable_results)

    # Compute imputed rates if available
    if not args.observable_only and "status_agg_0" in df.columns:
        logger.info("\nComputing imputed status rates...")

        # Collect per-imputation results
        all_results = []
        for i in range(args.n_imputations):
            if f"status_agg_{i}" not in df.columns:
                logger.warning(f"Missing imputation {i}, stopping at {i} imputations")
                break
            imp_results = compute_rates_single_imputation(df, i, programs)
            all_results.append(imp_results)

        if all_results:
            all_results_df = pd.concat(all_results, ignore_index=True)

            # Combine using Rubin's rules
            combined_results = combine_imputations(all_results_df, len(all_results))
            combined_results = apply_suppression_rules(combined_results)
            combined_results = format_output_table(combined_results, args.year, source="imputed")

            save_results(combined_results, args.year, suffix="_imputed")

            logger.info("\n" + "=" * 80)
            logger.info("IMPUTED STATUS RESULTS")
            logger.info("=" * 80)
            print_summary_table(combined_results)

    # Save combined output (main results file)
    if not args.observable_only and "status_agg_0" in df.columns:
        main_results = combined_results
    else:
        main_results = observable_results

    save_results(main_results, args.year)

    # Also save population totals
    pop_results = []
    for group in main_results["group"].unique():
        row = main_results[main_results["group"] == group].iloc[0]
        pop_results.append(
            {
                "year": args.year,
                "group": group,
                "n_unweighted": row["n_unweighted"],
                "n_weighted": row["n_weighted"],
            }
        )

    pop_df = pd.DataFrame(pop_results)
    pop_path = config.TABLES_DIR / f"ca_population_by_group_{args.year}.csv"
    pop_df.to_csv(pop_path, index=False)
    logger.info(f"Saved population totals: {pop_path}")

    logger.info("\nRate estimation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
