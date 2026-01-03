"""
Data fetching script for California welfare participation analysis.

Downloads:
1. ACS PUMS (person and housing files) for California
2. SIPP data for legal status model training
3. Pew Research benchmark data for calibration

Usage:
    python -m src.00_fetch_data [--year YEAR] [--overwrite]
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import requests
import yaml

from . import config
from .utils.download import (
    ACSPUMSDownloader,
    PewDataDownloader,
    SIPPDownloader,
    check_url_exists,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def fetch_acs_pums(
    years: list[int],
    state: str = "ca",
    survey: str = "1-Year",
    overwrite: bool = False,
) -> dict[int, dict[str, Path]]:
    """
    Fetch ACS PUMS data for specified years.

    Args:
        years: List of years to download
        state: State abbreviation
        survey: Survey type ('1-Year' or '5-Year')
        overwrite: Overwrite existing files

    Returns:
        Dict of year -> {file_type: path}
    """
    downloader = ACSPUMSDownloader(config.RAW_DATA_DIR)
    results = {}

    for year in years:
        logger.info(f"Fetching ACS PUMS {year} {survey} for {state.upper()}...")

        try:
            files = downloader.download_state_data(
                year=year,
                state=state,
                survey=survey,
                overwrite=overwrite,
            )

            if files:
                results[year] = files
                logger.info(f"Successfully downloaded ACS {year}: {list(files.keys())}")
            else:
                logger.warning(f"No files downloaded for ACS {year}")

        except requests.HTTPError as e:
            logger.error(f"HTTP error downloading ACS {year}: {e}")
        except requests.Timeout as e:
            logger.error(f"Timeout downloading ACS {year}: {e}")
        except requests.ConnectionError as e:
            logger.error(f"Connection error downloading ACS {year}: {e}")
        except OSError as e:
            logger.error(f"File system error for ACS {year}: {e}")

    return results


def fetch_sipp(
    years: list[int],
    overwrite: bool = False,
) -> dict[int, Path]:
    """
    Fetch SIPP data for specified years.

    Args:
        years: List of years to try
        overwrite: Overwrite existing files

    Returns:
        Dict of year -> data directory path
    """
    downloader = SIPPDownloader(config.RAW_DATA_DIR)
    results = {}

    for year in years:
        logger.info(f"Fetching SIPP {year}...")

        try:
            data_dir = downloader.download_sipp_data(year=year, overwrite=overwrite)

            if data_dir:
                results[year] = data_dir
                logger.info(f"Successfully downloaded SIPP {year}")
            else:
                logger.warning(f"SIPP {year} not found or download failed")

        except requests.HTTPError as e:
            logger.error(f"HTTP error downloading SIPP {year}: {e}")
        except requests.Timeout as e:
            logger.error(f"Timeout downloading SIPP {year}: {e}")
        except requests.ConnectionError as e:
            logger.error(f"Connection error downloading SIPP {year}: {e}")
        except OSError as e:
            logger.error(f"File system error for SIPP {year}: {e}")

    return results


def fetch_pew_benchmarks(overwrite: bool = False) -> dict[str, Path]:
    """
    Fetch Pew Research benchmark data.

    Args:
        overwrite: Overwrite existing files

    Returns:
        Dict of file type -> path
    """
    downloader = PewDataDownloader(config.EXTERNAL_DATA_DIR)
    results = {}

    # State trends
    logger.info("Fetching Pew state trends data...")
    state_trends = downloader.download_state_trends(
        url=config.PEW_STATE_TRENDS_URL,
        overwrite=overwrite,
    )
    if state_trends:
        results["state_trends"] = state_trends
        logger.info(f"Downloaded state trends: {state_trends}")
    else:
        logger.warning("Could not download Pew state trends. Manual download may be required.")

    # Labor force by state
    logger.info("Fetching Pew labor force data...")
    labor_force = downloader.download_labor_force(
        url=config.PEW_LABOR_FORCE_URL,
        overwrite=overwrite,
    )
    if labor_force:
        results["labor_force"] = labor_force
        logger.info(f"Downloaded labor force data: {labor_force}")
    else:
        logger.warning("Could not download Pew labor force data.")

    return results


def check_data_availability() -> dict:
    """
    Check availability of data sources before downloading.

    Returns:
        Dict with availability status for each source
    """
    logger.info("Checking data source availability...")

    availability = {
        "acs_pums": {},
        "sipp": {},
        "pew": {},
    }

    # Check ACS PUMS
    downloader = ACSPUMSDownloader(config.RAW_DATA_DIR)
    for year in config.PRIMARY_YEARS:
        url = downloader.get_download_url(year, "person", config.CA_STATE_ABBR)
        available = check_url_exists(url)
        availability["acs_pums"][year] = available
        logger.info(f"ACS PUMS {year}: {'Available' if available else 'Not available'}")

    # Check Pew
    pew_available = check_url_exists(config.PEW_STATE_TRENDS_URL)
    availability["pew"]["state_trends"] = pew_available
    logger.info(f"Pew state trends: {'Available' if pew_available else 'Not available'}")

    return availability


def update_data_inventory(
    acs_files: dict,
    sipp_files: dict,
    pew_files: dict,
) -> None:
    """
    Update data inventory YAML with download information.

    Args:
        acs_files: Downloaded ACS files
        sipp_files: Downloaded SIPP files
        pew_files: Downloaded Pew files
    """
    inventory_path = config.DOCS_DIR / "data_inventory.yaml"

    # Read existing inventory
    if inventory_path.exists():
        try:
            with open(inventory_path) as f:
                inventory = yaml.safe_load(f)
            if inventory is None:
                inventory = {"download_log": []}
        except yaml.YAMLError as e:
            logger.warning(f"Could not parse existing inventory YAML: {e}")
            logger.warning("Creating new inventory file")
            inventory = {"download_log": []}
    else:
        inventory = {"download_log": []}

    # Ensure download_log exists and is a list
    if "download_log" not in inventory or inventory["download_log"] is None:
        inventory["download_log"] = []

    timestamp = datetime.now().isoformat()

    # Add ACS downloads
    for year, files in acs_files.items():
        for file_type, path in files.items():
            inventory["download_log"].append({
                "dataset": f"acs_pums_{year}_{file_type}",
                "download_date": timestamp,
                "file_path": str(path),
                "file_size": path.stat().st_size if path.exists() else 0,
            })

    # Add SIPP downloads
    for year, path in sipp_files.items():
        inventory["download_log"].append({
            "dataset": f"sipp_{year}",
            "download_date": timestamp,
            "file_path": str(path),
        })

    # Add Pew downloads
    for file_type, path in pew_files.items():
        if path:
            inventory["download_log"].append({
                "dataset": f"pew_{file_type}",
                "download_date": timestamp,
                "file_path": str(path),
                "file_size": path.stat().st_size if path.exists() else 0,
            })

    # Write updated inventory
    with open(inventory_path, "w") as f:
        yaml.dump(inventory, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Updated data inventory: {inventory_path}")


def main():
    """Main entry point for data fetching."""
    parser = argparse.ArgumentParser(
        description="Fetch data for California welfare participation analysis"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Specific ACS year to fetch (default: try 2024, then 2023)",
    )
    parser.add_argument(
        "--survey",
        choices=["1-Year", "5-Year"],
        default="1-Year",
        help="ACS survey type (default: 1-Year)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check availability, don't download",
    )
    parser.add_argument(
        "--skip-sipp",
        action="store_true",
        help="Skip SIPP download",
    )
    parser.add_argument(
        "--skip-pew",
        action="store_true",
        help="Skip Pew benchmark download",
    )

    args = parser.parse_args()

    # Validate year if provided
    if args.year is not None:
        try:
            config.validate_year(args.year)
        except ValueError as e:
            logger.error(str(e))
            return 1

    logger.info("=" * 60)
    logger.info("California Welfare Analysis - Data Fetching")
    logger.info("=" * 60)

    # Check availability first
    availability = check_data_availability()

    if args.check_only:
        logger.info("Check-only mode, exiting.")
        return 0

    # Determine which ACS years to fetch
    if args.year:
        acs_years = [args.year]
    else:
        # Find first available year from preferences
        acs_years = []
        for year in config.PRIMARY_YEARS:
            if availability["acs_pums"].get(year, False):
                acs_years.append(year)
                break

        if not acs_years:
            logger.error("No ACS PUMS data available for preferred years")
            return 1

    # Fetch ACS PUMS
    logger.info(f"\nFetching ACS PUMS for years: {acs_years}")
    acs_files = fetch_acs_pums(
        years=acs_years,
        state=config.CA_STATE_ABBR,
        survey=args.survey,
        overwrite=args.overwrite,
    )

    if not acs_files:
        logger.error("Failed to download any ACS PUMS data")
        return 1

    # Fetch SIPP
    sipp_files = {}
    if not args.skip_sipp:
        logger.info(f"\nFetching SIPP for years: {config.SIPP_YEARS}")
        sipp_files = fetch_sipp(
            years=config.SIPP_YEARS,
            overwrite=args.overwrite,
        )

        if not sipp_files:
            logger.warning("No SIPP data downloaded. Status imputation may not be possible.")
    else:
        logger.info("Skipping SIPP download")

    # Fetch Pew benchmarks
    pew_files = {}
    if not args.skip_pew:
        logger.info("\nFetching Pew Research benchmark data...")
        pew_files = fetch_pew_benchmarks(overwrite=args.overwrite)

        if not pew_files:
            logger.warning(
                "Pew data not downloaded. Using fallback calibration values. "
                "See docs/manual_download.md for manual download instructions."
            )
    else:
        logger.info("Skipping Pew download")

    # Update data inventory
    update_data_inventory(acs_files, sipp_files, pew_files)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"ACS PUMS: {len(acs_files)} year(s) downloaded")
    for year, files in acs_files.items():
        for file_type, path in files.items():
            logger.info(f"  - {year} {file_type}: {path}")

    logger.info(f"SIPP: {len(sipp_files)} year(s) downloaded")
    for year, path in sipp_files.items():
        logger.info(f"  - {year}: {path}")

    logger.info(f"Pew: {len(pew_files)} file(s) downloaded")
    for file_type, path in pew_files.items():
        logger.info(f"  - {file_type}: {path}")

    logger.info("\nData fetching complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
