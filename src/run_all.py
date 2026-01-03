"""
Main pipeline orchestrator for California welfare participation analysis.

This script runs all pipeline steps in sequence:
1. Fetch data (ACS PUMS, SIPP, Pew benchmarks)
2. Clean ACS data
3. Train status model from SIPP
4. Impute status in ACS
5. Estimate rates
6. Generate report

Usage:
    python -m src.run_all [--year YEAR] [--skip-fetch] [--force]
"""

import argparse
import logging
import subprocess
import sys

from . import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_step(step_name: str, command: list[str], critical: bool = True) -> bool:
    """
    Run a pipeline step.

    Args:
        step_name: Name of the step
        command: Command to run
        critical: If True, pipeline stops on failure

    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 60)
    logger.info(f"STEP: {step_name}")
    logger.info("=" * 60)
    logger.info(f"Command: {' '.join(command)}")

    try:
        subprocess.run(
            command,
            check=True,
            capture_output=False,
        )
        logger.info(f"Step '{step_name}' completed successfully")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Step '{step_name}' failed with exit code {e.returncode}")
        if critical:
            logger.error("PIPELINE STOPPED due to critical step failure")
            return False
        else:
            logger.warning("Continuing despite non-critical step failure")
            return True

    except FileNotFoundError as e:
        logger.error(f"Step '{step_name}' failed - command not found: {e}")
        if critical:
            return False
        return True
    except PermissionError as e:
        logger.error(f"Step '{step_name}' failed - permission denied: {e}")
        if critical:
            return False
        return True
    except OSError as e:
        logger.error(f"Step '{step_name}' failed - OS error: {e}")
        if critical:
            return False
        return True


def check_prerequisites() -> bool:
    """
    Check that required tools are available.

    Returns:
        True if all prerequisites met
    """
    logger.info("Checking prerequisites...")

    # Check Python version (sys already imported at module level)
    if sys.version_info < (3, 11):
        logger.error(f"Python 3.11+ required, found {sys.version}")
        return False

    # Check required packages
    required = ["pandas", "numpy", "sklearn", "requests", "tqdm"]
    missing = []

    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        logger.error(f"Missing required packages: {missing}")
        logger.error("Run: pip install -r requirements.txt")
        return False

    logger.info("Prerequisites check passed")
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run California welfare participation analysis pipeline"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2023,
        help="ACS year to analyze (default: 2023)",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip data fetching (assume data already downloaded)",
    )
    parser.add_argument(
        "--skip-sipp",
        action="store_true",
        help="Skip SIPP model training (use existing model)",
    )
    parser.add_argument(
        "--observable-only",
        action="store_true",
        help="Only compute observable status rates (skip imputation)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run of all steps",
    )
    parser.add_argument(
        "--n-imputations",
        type=int,
        default=config.N_IMPUTATIONS,
        help=f"Number of imputations (default: {config.N_IMPUTATIONS})",
    )

    args = parser.parse_args()

    # Validate year
    try:
        config.validate_year(args.year)
    except ValueError as e:
        logger.error(str(e))
        return 1

    # Ensure required directories exist
    config.ensure_directories()

    logger.info("#" * 60)
    logger.info("# CALIFORNIA WELFARE PARTICIPATION ANALYSIS PIPELINE")
    logger.info("#" * 60)
    logger.info(f"Year: {args.year}")
    logger.info(f"Skip fetch: {args.skip_fetch}")
    logger.info(f"Observable only: {args.observable_only}")
    logger.info("")

    # Check prerequisites
    if not check_prerequisites():
        return 1

    python = sys.executable

    # Step 1: Fetch data
    if not args.skip_fetch:
        success = run_step(
            "Fetch Data",
            [python, "-m", "src.00_fetch_data", "--year", str(args.year)],
            critical=True,
        )
        if not success:
            return 1

    # Step 2: Clean ACS data
    acs_processed = config.PROCESSED_DATA_DIR / f"acs_{args.year}_ca_processed.parquet"

    if args.force or not acs_processed.exists():
        success = run_step(
            "Clean ACS Data",
            [python, "-m", "src.01_clean_acs", "--year", str(args.year), "--validate"],
            critical=True,
        )
        if not success:
            return 1
    else:
        logger.info(f"Skipping ACS cleaning (file exists: {acs_processed})")

    # Step 3: Train status model from SIPP
    if not args.observable_only and not args.skip_sipp:
        model_files = list(config.MODELS_DIR.glob("status_model_*.joblib"))

        if args.force or not model_files:
            success = run_step(
                "Train Status Model",
                [python, "-m", "src.02_train_status_model", "--model", "both"],
                critical=True,  # This is critical per user requirement
            )
            if not success:
                logger.error("")
                logger.error("SIPP model training failed.")
                logger.error("The pipeline cannot proceed without a trained model.")
                logger.error("")
                logger.error("Options:")
                logger.error("1. Ensure SIPP data is downloaded correctly")
                logger.error("2. Check SIPP codebook for legal status variables")
                logger.error("3. Use --observable-only to skip imputation")
                logger.error("")
                return 1
        else:
            logger.info("Skipping SIPP model training (models exist)")

    # Step 4: Impute status in ACS
    if not args.observable_only:
        acs_imputed = config.PROCESSED_DATA_DIR / f"acs_{args.year}_ca_imputed.parquet"

        if args.force or not acs_imputed.exists():
            success = run_step(
                "Impute Status",
                [
                    python,
                    "-m",
                    "src.03_impute_status_acs",
                    "--year",
                    str(args.year),
                    "--n-imputations",
                    str(args.n_imputations),
                ],
                critical=True,
            )
            if not success:
                return 1
        else:
            logger.info(f"Skipping imputation (file exists: {acs_imputed})")

    # Step 5: Estimate rates
    rate_args = [python, "-m", "src.04_estimate_rates", "--year", str(args.year)]
    if args.observable_only:
        rate_args.append("--observable-only")
    else:
        rate_args.extend(["--n-imputations", str(args.n_imputations)])

    success = run_step(
        "Estimate Rates",
        rate_args,
        critical=True,
    )
    if not success:
        return 1

    # Step 6: Generate report
    success = run_step(
        "Generate Report",
        [python, "-m", "src.05_report", "--year", str(args.year)],
        critical=False,
    )

    # Summary
    logger.info("")
    logger.info("#" * 60)
    logger.info("# PIPELINE COMPLETE")
    logger.info("#" * 60)
    logger.info("")
    logger.info("Output files:")

    # List output files
    for pattern in ["*.csv", "*.parquet"]:
        for f in config.TABLES_DIR.glob(pattern):
            logger.info(f"  - {f}")

    for f in config.FIGURES_DIR.glob("*.png"):
        logger.info(f"  - {f}")

    for f in config.REPORTS_DIR.glob("*.md"):
        logger.info(f"  - {f}")

    logger.info("")
    logger.info("See reports/ for the final analysis report.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
