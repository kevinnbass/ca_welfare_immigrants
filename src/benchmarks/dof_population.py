"""
California Department of Finance population estimates.

Sources:
- E-1: City/County Population Estimates (January 1)
- E-5: Population/Housing Estimates for Cities, Counties, State
- E-6: Population Projections

This replaces hard-coded values in src/utils/validation.py.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# DOF Data URLs (may need periodic updates)
DOF_BASE_URL = "https://dof.ca.gov/forecasting/demographics/estimates"


@dataclass
class PopulationEstimate:
    """A single population estimate."""

    year: int
    geography: str  # "California" or county name
    population: int
    source: str = "DOF"
    vintage: Optional[str] = None  # e.g., "January 2024"
    notes: str = ""


# Fallback population estimates (used when DOF fetch fails)
# Source: Census Bureau population estimates / DOF historical data
FALLBACK_POPULATION: dict[int, int] = {
    2019: 39_461_588,
    2020: 39_538_223,
    2021: 39_237_836,
    2022: 38_965_193,
    2023: 39_000_000,  # Approximate
    2024: 39_100_000,  # Approximate
    2025: 39_200_000,  # Projected
}


class DOFPopulationFetcher:
    """
    Fetch and parse California DOF population estimates.

    Replaces hard-coded CA_POPULATION_2023, CA_POPULATION_2024 in validation.py.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize fetcher.

        Args:
            cache_dir: Directory to cache downloaded files.
                      Defaults to data/external/dof relative to project root.
        """
        if cache_dir is None:
            from .. import config

            cache_dir = config.EXTERNAL_DATA_DIR / "dof"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._estimates: dict[tuple[int, str], PopulationEstimate] = {}

    def fetch_dof_e1_population(
        self,
        year: int,
        overwrite: bool = False,
    ) -> Optional[PopulationEstimate]:
        """
        Fetch E-1 population estimate for a specific year.

        Args:
            year: Estimate year (e.g., 2023, 2024)
            overwrite: Force re-download

        Returns:
            PopulationEstimate or None if unavailable
        """
        cache_key = (year, "California")
        if not overwrite and cache_key in self._estimates:
            return self._estimates[cache_key]

        try:
            # Construct URL for year-specific E-1 file
            url = self._get_e1_url(year)
            dest_path = self.cache_dir / f"dof_e1_{year}.xlsx"

            if not overwrite and dest_path.exists():
                logger.debug(f"Using cached DOF E-1: {dest_path}")
            else:
                self._download_file(url, dest_path)

            # Parse Excel file
            population = self._parse_e1_file(dest_path, year)

            estimate = PopulationEstimate(
                year=year,
                geography="California",
                population=population,
                source="DOF E-1",
                vintage=f"January {year}",
            )

            self._estimates[cache_key] = estimate
            return estimate

        except requests.HTTPError as e:
            logger.warning(f"Could not fetch DOF E-1 for {year}: {e}")
            return None
        except FileNotFoundError as e:
            logger.warning(f"DOF E-1 file not found: {e}")
            return None
        except (KeyError, ValueError, IndexError) as e:
            logger.warning(f"Could not parse DOF E-1 for {year}: {e}")
            return None

    def fetch_latest_dof_population_estimates(self) -> dict[int, PopulationEstimate]:
        """
        Fetch all available recent DOF estimates.

        Returns:
            Dict of year -> PopulationEstimate for California
        """
        current_year = datetime.now().year
        years_to_try = [current_year, current_year - 1, current_year - 2]

        results = {}
        for year in years_to_try:
            estimate = self.fetch_dof_e1_population(year)
            if estimate:
                results[year] = estimate

        return results

    def get_population(
        self,
        year: int,
        geography: str = "California",
        use_fallback: bool = True,
    ) -> Optional[int]:
        """
        Get population for year/geography, with fallback.

        This is the main interface for replacing hard-coded values.

        Args:
            year: Target year
            geography: Geographic area (currently only "California" supported)
            use_fallback: Whether to use fallback values if DOF unavailable

        Returns:
            Population count, or None if unavailable
        """
        # Try cached first
        cache_key = (year, geography)
        if cache_key in self._estimates:
            return self._estimates[cache_key].population

        # Try to fetch from DOF
        if geography == "California":
            estimate = self.fetch_dof_e1_population(year)
            if estimate:
                return estimate.population

        # Fallback to hard-coded estimates
        if use_fallback and year in FALLBACK_POPULATION:
            logger.info(
                f"Using fallback population for {year}: {FALLBACK_POPULATION[year]:,}. "
                "DOF data not available."
            )
            return FALLBACK_POPULATION[year]

        return None

    def _get_e1_url(self, year: int) -> str:
        """
        Construct URL for E-1 file by year.

        Note: DOF URL structure may change; update as needed.
        """
        # DOF typically uses URLs like:
        # https://dof.ca.gov/wp-content/uploads/sites/352/Forecasting/Demographics/E-1/E-1_2024_Internet_Version.xlsx
        return (
            f"https://dof.ca.gov/wp-content/uploads/sites/352/Forecasting/"
            f"Demographics/E-1/E-1_{year}_Internet_Version.xlsx"
        )

    def _download_file(self, url: str, dest_path: Path) -> None:
        """Download file with basic error handling."""
        logger.info(f"Downloading DOF E-1 from {url}")

        response = requests.get(url, timeout=60)
        response.raise_for_status()

        with open(dest_path, "wb") as f:
            f.write(response.content)

        logger.info(f"Downloaded: {dest_path}")

    def _parse_e1_file(self, file_path: Path, year: int) -> int:
        """
        Parse E-1 Excel to extract statewide total.

        The E-1 file structure varies by year but typically has:
        - Header rows with title information
        - A row for "California" with the state total
        - County-level rows

        Args:
            file_path: Path to downloaded E-1 Excel file
            year: Year of the estimate

        Returns:
            California population total

        Raises:
            ValueError: If California total cannot be found
        """
        # Try different parsing strategies based on file structure
        try:
            # Try reading with header row detection
            df = pd.read_excel(file_path, sheet_name=0, header=None)

            # Look for "California" or "CALIFORNIA" in any row
            for idx, row in df.iterrows():
                row_str = " ".join(str(v).lower() for v in row.values if pd.notna(v))
                if "california" in row_str and "total" not in row_str.lower():
                    # Found California row - extract numeric value
                    for val in row.values:
                        if isinstance(val, (int, float)) and not pd.isna(val):
                            if val > 1_000_000:  # Sanity check for population
                                return int(val)

            # Alternative: look for state total in a specific location
            # E-1 files often have the state total in a consistent position
            raise ValueError(f"Could not find California total in E-1 for {year}")

        except Exception as e:
            logger.warning(f"Error parsing E-1 file {file_path}: {e}")
            raise


def get_ca_population(year: int, use_fallback: bool = True) -> int:
    """
    Get California population for a given year.

    Drop-in replacement for hard-coded values.

    Usage in validation.py:
        from ..benchmarks.dof_population import get_ca_population
        expected_pop = get_ca_population(self.year)

    Args:
        year: Target year
        use_fallback: Whether to use fallback if DOF unavailable

    Returns:
        California population estimate

    Raises:
        ValueError: If no estimate available and use_fallback=False
    """
    from .. import config

    fetcher = DOFPopulationFetcher(cache_dir=config.EXTERNAL_DATA_DIR / "dof")
    population = fetcher.get_population(year, use_fallback=use_fallback)

    if population is None:
        if use_fallback:
            # Interpolate/extrapolate from nearest known year
            known_years = sorted(FALLBACK_POPULATION.keys())
            if year < known_years[0]:
                population = FALLBACK_POPULATION[known_years[0]]
            elif year > known_years[-1]:
                population = FALLBACK_POPULATION[known_years[-1]]
            else:
                # Linear interpolation
                lower_year = max(y for y in known_years if y <= year)
                upper_year = min(y for y in known_years if y >= year)
                if lower_year == upper_year:
                    population = FALLBACK_POPULATION[lower_year]
                else:
                    ratio = (year - lower_year) / (upper_year - lower_year)
                    population = int(
                        FALLBACK_POPULATION[lower_year]
                        + ratio
                        * (
                            FALLBACK_POPULATION[upper_year]
                            - FALLBACK_POPULATION[lower_year]
                        )
                    )
        else:
            raise ValueError(f"No population estimate available for {year}")

    return population


def get_expected_foreign_born_pct() -> float:
    """Get expected percentage of foreign-born population in California."""
    return 0.27  # ~27% foreign-born in CA


def get_expected_noncitizen_pct() -> float:
    """Get expected percentage of noncitizens in California."""
    return 0.13  # ~13% noncitizens in CA
