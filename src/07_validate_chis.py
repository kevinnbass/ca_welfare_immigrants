"""
Validate ACS-based welfare estimates against CHIS 2023 data.

This script:
1. Loads CHIS 2023 Adult Public Use File
2. Computes weighted welfare program rates by citizenship status
3. Compares to ACS-based estimates
4. Outputs validation table

Usage:
    python -m src.07_validate_chis [--year YEAR]
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from . import config
from .utils.chis_loader import CHISLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_acs_estimates(year: int) -> pd.DataFrame:
    """
    Load ACS-based estimates for comparison.

    Args:
        year: Analysis year

    Returns:
        DataFrame with ACS estimates
    """
    # Try imputed first, then observable
    for suffix in ["_imputed", "_observable", ""]:
        file_path = config.TABLES_DIR / f"ca_rates_by_group_program_{year}{suffix}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            df["source"] = "ACS"
            logger.info(f"Loaded ACS estimates from {file_path}")
            return df

    raise FileNotFoundError(f"No ACS estimates found for year {year}")


def create_comparison_table(
    chis_df: pd.DataFrame,
    acs_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create side-by-side comparison table.

    Args:
        chis_df: CHIS estimates
        acs_df: ACS estimates

    Returns:
        Comparison DataFrame
    """
    # Map CHIS programs to ACS programs
    program_mapping = {
        "medicaid": "medicaid",
        "food_stamps": "snap",
    }

    # Map CHIS groups to comparable ACS groups
    group_mapping = {
        "US_BORN": "US_BORN",
        "NATURALIZED": None,  # Part of LEGAL_IMMIGRANT in ACS
        "NONCITIZEN": None,  # Mixed legal/illegal in ACS
    }

    comparisons = []

    for _, chis_row in chis_df.iterrows():
        if chis_row["group"] == "ALL":
            continue

        chis_program = chis_row["program"]
        acs_program = program_mapping.get(chis_program)

        if not acs_program:
            continue

        chis_group = chis_row["group"]

        # Find comparable ACS estimate
        if chis_group == "US_BORN":
            acs_match = acs_df[
                (acs_df["program"] == acs_program) &
                (acs_df["group"] == "US_BORN")
            ]
        elif chis_group == "NATURALIZED":
            # Naturalized is part of LEGAL_IMMIGRANT in ACS
            acs_match = acs_df[
                (acs_df["program"] == acs_program) &
                (acs_df["group"] == "LEGAL_IMMIGRANT")
            ]
        elif chis_group == "NONCITIZEN":
            # Non-citizens in CHIS = legal + illegal in ACS
            # Get both for reference
            acs_match = acs_df[
                (acs_df["program"] == acs_program) &
                (acs_df["group"].isin(["LEGAL_IMMIGRANT", "ILLEGAL"]))
            ]
        else:
            continue

        comparison = {
            "program": chis_program,
            "chis_group": chis_group,
            "chis_estimate": chis_row["estimate"],
            "chis_se": chis_row["se"],
            "chis_n": chis_row["n_unweighted"],
        }

        if len(acs_match) == 1:
            comparison["acs_group"] = acs_match.iloc[0]["group"]
            comparison["acs_estimate"] = acs_match.iloc[0]["estimate"]
            comparison["acs_se"] = acs_match.iloc[0].get("se", None)
            comparison["difference"] = chis_row["estimate"] - acs_match.iloc[0]["estimate"]
        elif len(acs_match) > 1:
            # For NONCITIZEN, show range
            comparison["acs_group"] = "LEGAL+ILLEGAL"
            comparison["acs_estimate"] = f"{acs_match['estimate'].min():.3f}-{acs_match['estimate'].max():.3f}"
            comparison["acs_se"] = None
            comparison["difference"] = None

        comparisons.append(comparison)

    return pd.DataFrame(comparisons)


def print_validation_summary(
    chis_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> None:
    """Print formatted validation summary."""
    print("\n" + "=" * 70)
    print("CHIS vs ACS Validation Summary - California 2023")
    print("=" * 70)

    print("\n### CHIS 2023 Estimates by Citizenship Status ###\n")

    for program in chis_df["program"].unique():
        prog_df = chis_df[
            (chis_df["program"] == program) &
            (chis_df["group"] != "ALL")
        ]
        program_label = {
            "medicaid": "Medi-Cal",
            "food_stamps": "Food Stamps/CalFresh",
        }.get(program, program)

        print(f"{program_label}:")
        for _, row in prog_df.iterrows():
            est_pct = row["estimate"] * 100
            se_pct = row["se"] * 100
            ci_low = row["ci_lower"] * 100
            ci_high = row["ci_upper"] * 100
            print(f"  {row['group']:15} {est_pct:5.1f}% (SE: {se_pct:.1f}%, 95% CI: {ci_low:.1f}-{ci_high:.1f}%, n={row['n_unweighted']:,})")
        print()

    print("\n### Comparison to ACS Estimates ###\n")
    print(f"{'Program':<15} {'CHIS Group':<15} {'CHIS':<10} {'ACS Group':<18} {'ACS':<12} {'Diff':<8}")
    print("-" * 78)

    for _, row in comparison_df.iterrows():
        chis_est = f"{row['chis_estimate']*100:.1f}%"
        if isinstance(row.get("acs_estimate"), str):
            acs_est = row["acs_estimate"]
            diff = "N/A"
        elif pd.notna(row.get("acs_estimate")):
            acs_est = f"{row['acs_estimate']*100:.1f}%"
            if pd.notna(row.get("difference")):
                diff = f"{row['difference']*100:+.1f}pp"
            else:
                diff = "N/A"
        else:
            acs_est = "N/A"
            diff = "N/A"

        print(f"{row['program']:<15} {row['chis_group']:<15} {chis_est:<10} {row.get('acs_group', 'N/A'):<18} {acs_est:<12} {diff:<8}")

    print("\n### Key Notes ###")
    print("- CHIS NONCITIZEN includes both legal and illegal immigrants (cannot distinguish)")
    print("- CHIS Medi-Cal may capture only full-scope coverage, not emergency/restricted")
    print("- Food stamps question asked only of income-eligible subset in CHIS")
    print("- Differences expected due to survey methodology, question wording, reference periods")
    print()


def main():
    """Main entry point for CHIS validation."""
    parser = argparse.ArgumentParser(description="Validate against CHIS data")
    parser.add_argument(
        "--year",
        type=int,
        default=2023,
        help="Analysis year (default: 2023)",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"CHIS Validation - {args.year}")
    logger.info("=" * 60)

    # Load CHIS data
    chis_data_dir = config.RAW_DATA_DIR / "chis"
    loader = CHISLoader(chis_data_dir)

    try:
        loader.load(args.year)
    except FileNotFoundError as e:
        logger.error(f"CHIS data not found: {e}")
        logger.error("Download CHIS Adult PUF from https://healthpolicy.ucla.edu/our-work/public-use-files")
        return 1

    # Compute CHIS rates
    logger.info("Computing CHIS rates...")
    chis_df = loader.compute_all_rates()

    # Load ACS estimates
    try:
        acs_df = load_acs_estimates(args.year)
    except FileNotFoundError as e:
        logger.error(f"ACS estimates not found: {e}")
        logger.error("Run the main pipeline first: python -m src.run_all")
        return 1

    # Create comparison
    comparison_df = create_comparison_table(chis_df, acs_df)

    # Save results
    output_path = config.TABLES_DIR / f"chis_validation_{args.year}.csv"
    chis_df.to_csv(output_path, index=False)
    logger.info(f"Saved CHIS estimates: {output_path}")

    comparison_path = config.TABLES_DIR / f"chis_acs_comparison_{args.year}.csv"
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"Saved comparison: {comparison_path}")

    # Print summary
    print_validation_summary(chis_df, comparison_df)

    logger.info("CHIS validation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
