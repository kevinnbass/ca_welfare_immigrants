"""
Pooled-year ACS analysis for welfare participation rates.

This script:
1. Loads cleaned ACS data for multiple years
2. Harmonizes variables across years
3. Constructs pooled weights (divides by number of years)
4. Estimates rates by status for the pooled period
5. Outputs pooled estimates with proper variance estimation

Usage:
    python -m src.06_pooled_analysis [--years 2019 2020 2021 2022 2023]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import config
from .utils.analysis_units import get_estimand_spec, compute_rate_for_estimand
from .utils.imputation import combine_mi_results_rubins_rules
from .utils.weights import (
    weighted_proportion,
    coefficient_of_variation,
    confidence_interval,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_year_data(year: int, use_imputed: bool = True) -> Optional[pd.DataFrame]:
    """
    Load ACS data for a single year.

    Args:
        year: ACS year to load
        use_imputed: If True, load imputed file; otherwise load processed file

    Returns:
        DataFrame or None if file not found
    """
    if use_imputed:
        file_path = config.PROCESSED_DATA_DIR / f"acs_{year}_ca_imputed.parquet"
    else:
        file_path = config.PROCESSED_DATA_DIR / f"acs_{year}_ca_processed.parquet"

    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return None

    logger.info(f"Loading {year}: {file_path}")
    df = pd.read_parquet(file_path)
    df["year"] = year
    logger.info(f"  Loaded {len(df):,} records")

    return df


def harmonize_variables(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Harmonize variable names for a specific year.

    Some ACS variables change names across years. This function
    applies year-specific mappings from config.VARIABLE_HARMONIZATION.

    Args:
        df: DataFrame for a single year
        year: Year of the data

    Returns:
        DataFrame with harmonized variable names
    """
    df = df.copy()

    if year in config.VARIABLE_HARMONIZATION:
        mappings = config.VARIABLE_HARMONIZATION[year]
        for old_name, new_name in mappings.items():
            if old_name in df.columns and new_name not in df.columns:
                df = df.rename(columns={old_name: new_name})
                logger.debug(f"  Year {year}: Renamed {old_name} -> {new_name}")

    return df


def load_and_harmonize_years(
    years: list[int],
    use_imputed: bool = True,
) -> pd.DataFrame:
    """
    Load and harmonize ACS data across multiple years.

    Args:
        years: List of years to load
        use_imputed: If True, load imputed files

    Returns:
        Combined DataFrame with all years

    Raises:
        ValueError: If no data could be loaded
    """
    dfs = []

    for year in years:
        df = load_year_data(year, use_imputed=use_imputed)
        if df is not None:
            df = harmonize_variables(df, year)
            dfs.append(df)

    if not dfs:
        raise ValueError(f"No data found for years: {years}")

    logger.info(f"Loaded {len(dfs)} years of data")

    # Concatenate all years
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined dataset: {len(combined):,} total records")

    return combined


def construct_pooled_weights(
    df: pd.DataFrame,
    n_years: int,
) -> pd.DataFrame:
    """
    Divide all weight columns by number of years pooled.

    Per Census Bureau guidance for pooling ACS 1-year microdata:
    - Divide person weights (PWGTP, PWGTP1-80) by k
    - Divide household weights (WGTP, WGTP1-80) by k
    This preserves population totals while reducing variance.

    Args:
        df: Combined multi-year DataFrame
        n_years: Number of years pooled

    Returns:
        DataFrame with adjusted weights
    """
    df = df.copy()

    # Person weights
    person_weight_cols = [config.ACS_PERSON_WEIGHT]
    person_weight_cols += [
        f"{config.ACS_REP_WEIGHT_PREFIX_PERSON}{i}"
        for i in range(1, config.N_REPLICATE_WEIGHTS + 1)
    ]

    for col in person_weight_cols:
        if col in df.columns:
            df[col] = df[col] / n_years

    # Household weights
    hh_weight_cols = [config.ACS_HH_WEIGHT]
    hh_weight_cols += [
        f"{config.ACS_REP_WEIGHT_PREFIX_HH}{i}"
        for i in range(1, config.N_REPLICATE_WEIGHTS + 1)
    ]

    for col in hh_weight_cols:
        if col in df.columns:
            df[col] = df[col] / n_years

    logger.info(f"Divided weights by {n_years} for pooled analysis")

    return df


def estimate_pooled_rates(
    df: pd.DataFrame,
    programs: list[str],
    status_col: str = "observable_status",
    stratify_by: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Estimate welfare rates using pooled data.

    Args:
        df: Pooled DataFrame with adjusted weights
        programs: List of program keys to analyze
        status_col: Status column to use
        stratify_by: Optional list of columns to stratify by

    Returns:
        DataFrame with pooled rate estimates
    """
    results = []

    # Determine status values based on column
    if "status_agg" in status_col or "imputed" in status_col:
        status_values = ["US_BORN", "LEGAL_IMMIGRANT", "ILLEGAL"]
    else:
        status_values = ["US_BORN", "NATURALIZED", "NONCITIZEN"]

    for status_value in status_values:
        for program in programs:
            # Check if this is a defined estimand
            if program in config.WELFARE_PROGRAMS:
                try:
                    estimand = get_estimand_spec(program)

                    rate, variance, se, n_unweighted, n_weighted = compute_rate_for_estimand(
                        df,
                        estimand=estimand,
                        status_col=status_col,
                        status_value=status_value,
                    )

                    if not np.isnan(se):
                        ci_low, ci_high = confidence_interval(rate, se)
                        cv = coefficient_of_variation(rate, se)
                    else:
                        ci_low, ci_high = np.nan, np.nan
                        cv = np.nan

                    results.append({
                        "group": status_value,
                        "program": program,
                        "estimate": rate,
                        "se": se,
                        "ci_lower": ci_low,
                        "ci_upper": ci_high,
                        "cv": cv,
                        "n_unweighted": n_unweighted,
                        "n_weighted": n_weighted,
                        "unit": estimand.unit,
                        "estimand_id": estimand.estimand_id,
                    })
                    continue

                except Exception as e:
                    logger.warning(f"Error computing estimand {program}: {e}")

            # Fall back for non-estimand programs
            if program not in df.columns:
                continue

            mask = df[status_col] == status_value
            df_group = df.loc[mask]
            n_unweighted = mask.sum()
            n_weighted = df_group[config.ACS_PERSON_WEIGHT].sum()

            rate = weighted_proportion(
                df_group[program],
                df_group[config.ACS_PERSON_WEIGHT],
                validate_binary=False,
            )

            # Compute variance using replicate weights (SDR method)
            rate_reps = []
            for i in range(1, config.N_REPLICATE_WEIGHTS + 1):
                rep_col = f"{config.ACS_REP_WEIGHT_PREFIX_PERSON}{i}"
                if rep_col in df_group.columns:
                    rate_r = weighted_proportion(
                        df_group[program],
                        df_group[rep_col],
                        validate_binary=False,
                    )
                    rate_reps.append(rate_r)

            if rate_reps:
                rate_reps = np.array(rate_reps)
                # SDR variance (4/80 factor for ACS successive difference replication)
                variance = (4.0 / 80.0) * np.sum((rate_reps - rate) ** 2)
                se = np.sqrt(variance)
                ci_low, ci_high = confidence_interval(rate, se)
                cv = coefficient_of_variation(rate, se)
            else:
                se = np.nan
                ci_low, ci_high = np.nan, np.nan
                cv = np.nan

            results.append({
                "group": status_value,
                "program": program,
                "estimate": rate,
                "se": se,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "cv": cv,
                "n_unweighted": n_unweighted,
                "n_weighted": n_weighted,
                "unit": "person",
                "estimand_id": program,
            })

    return pd.DataFrame(results)


def estimate_pooled_rates_with_mi(
    df: pd.DataFrame,
    programs: list[str],
    n_imputations: int = 10,
) -> pd.DataFrame:
    """
    Estimate pooled rates combining multiple imputations.

    Args:
        df: Pooled DataFrame
        programs: List of program keys
        n_imputations: Number of imputations to combine

    Returns:
        DataFrame with combined pooled estimates
    """
    # Collect estimates from each imputation
    all_imp_results = []

    for i in range(n_imputations):
        status_col = f"status_agg_{i}"
        if status_col not in df.columns:
            logger.warning(f"Missing imputation {i}")
            break

        imp_results = estimate_pooled_rates(
            df, programs, status_col=status_col
        )
        imp_results["imputation"] = i
        all_imp_results.append(imp_results)

    if not all_imp_results:
        return pd.DataFrame()

    all_results = pd.concat(all_imp_results, ignore_index=True)

    # Combine using Rubin's rules
    combined = []
    for (group, program), gdf in all_results.groupby(["group", "program"]):
        estimates = gdf["estimate"].values
        variances = gdf["se"].values ** 2  # Convert SE to variance

        # Handle NaN variances
        if np.any(np.isnan(variances)):
            mean_est = np.nanmean(estimates)
            between_var = np.nanvar(estimates, ddof=1)
            total_var = between_var * (1 + 1 / len(estimates))
            se = np.sqrt(total_var)
        else:
            mi_result = combine_mi_results_rubins_rules(
                list(estimates), list(variances)
            )
            mean_est = mi_result.estimate
            se = mi_result.se

        if not np.isnan(se):
            ci_low, ci_high = confidence_interval(mean_est, se)
            cv = coefficient_of_variation(mean_est, se)
        else:
            ci_low, ci_high = np.nan, np.nan
            cv = np.nan

        combined.append({
            "group": group,
            "program": program,
            "estimate": mean_est,
            "se": se,
            "ci_lower": ci_low,
            "ci_upper": ci_high,
            "cv": cv,
            "n_unweighted": gdf["n_unweighted"].mean(),
            "n_weighted": gdf["n_weighted"].mean(),
            "unit": gdf["unit"].iloc[0],
            "estimand_id": gdf["estimand_id"].iloc[0],
        })

    return pd.DataFrame(combined)


def save_pooled_results(
    df: pd.DataFrame,
    years: list[int],
    source: str = "pooled",
) -> Path:
    """
    Save pooled analysis results.

    Args:
        df: Results DataFrame
        years: List of years in the pooled analysis
        source: Source description

    Returns:
        Path to saved directory
    """
    start_year = min(years)
    end_year = max(years)

    # Create output directory
    output_dir = config.POOLED_OUTPUTS_DIR / f"{start_year}-{end_year}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add metadata columns
    df = df.copy()
    df["period"] = f"{start_year}-{end_year}"
    df["n_years"] = len(years)
    df["source"] = source

    # Save CSV
    csv_path = output_dir / "rate_estimates.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV: {csv_path}")

    # Save Parquet
    parquet_path = output_dir / "rate_estimates.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info(f"Saved Parquet: {parquet_path}")

    # Save methodology documentation
    methodology_path = output_dir / "methodology.md"
    with open(methodology_path, "w") as f:
        f.write(f"""# Pooled-Year Analysis Methodology

## Period
{start_year} - {end_year} ({len(years)} years)

## Weight Adjustment

For pooling k years of ACS 1-year microdata, weights are divided by k.

- Person weights (PWGTP, PWGTP1-80) divided by {len(years)}
- Household weights (WGTP, WGTP1-80) divided by {len(years)}

This approach:
- Preserves approximate population totals
- Reduces sampling variance by pooling multiple years
- Is consistent with Census Bureau guidance for ACS multi-year estimates

## Variable Harmonization

Year-specific variable name changes were harmonized using mappings in config.py.

## Multiple Imputation

If imputed status was available, results were combined using Rubin's rules:
- Within-imputation variance: Average of per-imputation variances
- Between-imputation variance: Variance of point estimates across imputations
- Total variance: Within + (1 + 1/M) * Between

## Limitations

1. Pooled estimates represent average rates over the period, not trends
2. Policy or eligibility changes within the period are not accounted for
3. The division-by-k approach assumes constant population structure
4. Survey design changes across years may affect comparability

## Citation

Analysis conducted using California Welfare Immigrants analysis pipeline.
Data source: American Community Survey Public Use Microdata Sample (ACS PUMS).
""")
    logger.info(f"Saved methodology: {methodology_path}")

    return output_dir


def main():
    """Main entry point for pooled analysis."""
    parser = argparse.ArgumentParser(
        description="Pooled-year ACS analysis for welfare rates"
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=config.POOLED_YEARS,
        help=f"Years to pool (default: {config.POOLED_YEARS})",
    )
    parser.add_argument(
        "--observable-only",
        action="store_true",
        help="Only use observable status (no imputation)",
    )
    parser.add_argument(
        "--n-imputations",
        type=int,
        default=config.N_IMPUTATIONS,
        help="Number of imputations to combine",
    )
    parser.add_argument(
        "--include-household",
        action="store_true",
        help="Include household-level SNAP estimands",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"Pooled Analysis: {min(args.years)}-{max(args.years)}")
    logger.info("=" * 60)

    # Define programs
    programs = ["medicaid", "snap", "ssi", "public_assistance", "any_benefit"]

    if args.include_household:
        programs.extend([
            "snap_household_householder",
            "snap_household_highest_risk",
            "snap_person",
        ])

    # Load and harmonize data
    try:
        pooled_df = load_and_harmonize_years(
            args.years,
            use_imputed=not args.observable_only,
        )
    except ValueError as e:
        logger.error(f"Could not load data: {e}")
        return 1

    # Construct pooled weights
    pooled_df = construct_pooled_weights(pooled_df, n_years=len(args.years))

    # Estimate rates
    if args.observable_only:
        logger.info("\nEstimating observable status rates...")
        results = estimate_pooled_rates(
            pooled_df,
            programs,
            status_col="observable_status",
        )
        source = "pooled_observable"
    else:
        logger.info("\nEstimating imputed status rates with MI...")
        results = estimate_pooled_rates_with_mi(
            pooled_df,
            programs,
            n_imputations=args.n_imputations,
        )
        source = "pooled_imputed"

    # Save results
    output_dir = save_pooled_results(results, args.years, source=source)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("POOLED RESULTS SUMMARY")
    logger.info("=" * 60)

    for program in results["program"].unique():
        prog_df = results[results["program"] == program]
        logger.info(f"\n{program}:")
        logger.info("-" * 40)

        for _, row in prog_df.iterrows():
            est = row["estimate"] * 100
            ci_low = row["ci_lower"] * 100 if pd.notna(row["ci_lower"]) else np.nan
            ci_high = row["ci_upper"] * 100 if pd.notna(row["ci_upper"]) else np.nan

            if pd.isna(est):
                logger.info(f"  {row['group']:20s}: N/A")
            else:
                logger.info(
                    f"  {row['group']:20s}: {est:5.1f}% ({ci_low:5.1f}% - {ci_high:5.1f}%)"
                )

    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("\nPooled analysis complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
