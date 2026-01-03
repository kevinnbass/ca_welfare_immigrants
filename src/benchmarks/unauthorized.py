"""
Multi-source unauthorized immigrant population benchmarks.

Sources:
- Pew Research Center (current primary)
- Migration Policy Institute (MPI) CA Profile
- Center for Migration Studies (CMS)
- DHS estimates

Enables sensitivity analysis by providing alternative calibration targets.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class UnauthorizedEstimate:
    """A single unauthorized population estimate."""

    source: str
    year: int
    estimate: int  # Point estimate
    low: Optional[int] = None  # Lower bound (if CI provided)
    high: Optional[int] = None  # Upper bound
    geography: str = "California"
    notes: str = ""
    url: Optional[str] = None
    methodology: Optional[str] = None


# Known estimates by source and year
# These serve as fallbacks when live fetching fails
KNOWN_ESTIMATES: dict[str, dict[int, UnauthorizedEstimate]] = {
    "pew": {
        2021: UnauthorizedEstimate(
            source="pew",
            year=2021,
            estimate=2_200_000,
            low=2_000_000,
            high=2_400_000,
            notes="Pew Research Center state trends",
            url="https://www.pewresearch.org/short-reads/2023/11/16/what-we-know-about-unauthorized-immigrants-living-in-the-u-s/",
            methodology="Residual method applied to ACS/CPS",
        ),
        2022: UnauthorizedEstimate(
            source="pew",
            year=2022,
            estimate=2_200_000,
            notes="Pew Research Center state trends",
            methodology="Residual method applied to ACS/CPS",
        ),
        2023: UnauthorizedEstimate(
            source="pew",
            year=2023,
            estimate=2_200_000,
            notes="Pew Research Center, August 2025 report",
            methodology="Residual method applied to ACS/CPS",
        ),
    },
    "mpi": {
        2019: UnauthorizedEstimate(
            source="mpi",
            year=2019,
            estimate=2_300_000,
            notes="MPI California unauthorized profile",
            url="https://www.migrationpolicy.org/data/unauthorized-immigrant-population/state/CA",
            methodology="ACS-based estimates with demographic adjustment",
        ),
        2021: UnauthorizedEstimate(
            source="mpi",
            year=2021,
            estimate=2_350_000,
            notes="MPI California unauthorized profile",
            methodology="ACS-based estimates with demographic adjustment",
        ),
        2022: UnauthorizedEstimate(
            source="mpi",
            year=2022,
            estimate=2_400_000,
            notes="MPI California unauthorized profile",
            methodology="ACS-based estimates with demographic adjustment",
        ),
    },
    "cms": {
        2019: UnauthorizedEstimate(
            source="cms",
            year=2019,
            estimate=2_280_000,
            notes="CMS state estimates",
            url="https://cmsny.org/publications/essay-2021-undocumented-and-eligible/",
            methodology="ACS residual with longitudinal adjustments",
        ),
        2021: UnauthorizedEstimate(
            source="cms",
            year=2021,
            estimate=2_320_000,
            notes="CMS state estimates",
            methodology="ACS residual with longitudinal adjustments",
        ),
    },
    "dhs": {
        2018: UnauthorizedEstimate(
            source="dhs",
            year=2018,
            estimate=2_050_000,
            notes="DHS Office of Immigration Statistics",
            url="https://www.dhs.gov/immigration-statistics/population-estimates/unauthorized-resident",
            methodology="Administrative data with survey adjustments",
        ),
        2019: UnauthorizedEstimate(
            source="dhs",
            year=2019,
            estimate=2_100_000,
            notes="DHS Office of Immigration Statistics",
            methodology="Administrative data with survey adjustments",
        ),
    },
}


class UnauthorizedBenchmarkFetcher:
    """
    Fetch and manage unauthorized immigrant benchmarks from multiple sources.
    """

    SOURCES: dict[str, dict[str, str]] = {
        "pew": {
            "name": "Pew Research Center",
            "url": "https://www.pewresearch.org/short-reads/",
            "methodology": "Residual method applied to ACS/CPS",
        },
        "mpi": {
            "name": "Migration Policy Institute",
            "url": "https://www.migrationpolicy.org/data/unauthorized-immigrant-population/state/CA",
            "methodology": "ACS-based estimates with demographic adjustment",
        },
        "cms": {
            "name": "Center for Migration Studies",
            "url": "https://cmsny.org/",
            "methodology": "ACS residual with longitudinal adjustments",
        },
        "dhs": {
            "name": "Department of Homeland Security",
            "url": "https://www.dhs.gov/immigration-statistics",
            "methodology": "Administrative data with survey adjustments",
        },
    }

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize fetcher.

        Args:
            cache_dir: Directory to cache downloaded files.
                      Defaults to data/external/benchmarks relative to project root.
        """
        if cache_dir is None:
            from .. import config

            cache_dir = config.EXTERNAL_DATA_DIR / "benchmarks"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_all_sources(self, year: int) -> dict[str, UnauthorizedEstimate]:
        """
        Fetch unauthorized estimates from all sources for a given year.

        Args:
            year: Target year

        Returns:
            Dict of source -> estimate (may be empty for some sources)
        """
        results = {}

        # Pew (primary - integrate with existing downloader if available)
        pew_estimate = self._fetch_pew(year)
        if pew_estimate:
            results["pew"] = pew_estimate

        # MPI
        mpi_estimate = self._fetch_mpi(year)
        if mpi_estimate:
            results["mpi"] = mpi_estimate

        # CMS
        cms_estimate = self._fetch_cms(year)
        if cms_estimate:
            results["cms"] = cms_estimate

        # DHS
        dhs_estimate = self._fetch_dhs(year)
        if dhs_estimate:
            results["dhs"] = dhs_estimate

        return results

    def get_benchmark_summary(self, year: int) -> dict[str, dict]:
        """
        Get summary of all benchmarks for sensitivity analysis.

        Args:
            year: Target year

        Returns:
            Dictionary with benchmark details and relative differences
        """
        estimates = self.fetch_all_sources(year)

        # Pew is reference
        pew_estimate = estimates.get("pew")
        pew_value = pew_estimate.estimate if pew_estimate else 0

        summary = {}
        for source, est in estimates.items():
            rel_diff = (est.estimate - pew_value) / pew_value if pew_value > 0 else 0
            summary[source] = {
                "name": self.SOURCES[source]["name"],
                "year": est.year,
                "estimate": est.estimate,
                "low": est.low,
                "high": est.high,
                "notes": est.notes,
                "methodology": est.methodology,
                "relative_diff_from_pew": rel_diff,
            }

        return summary

    def get_estimate(
        self,
        source: str,
        year: int,
        fallback_to_nearest: bool = True,
    ) -> Optional[UnauthorizedEstimate]:
        """
        Get estimate from a specific source.

        Args:
            source: Source key ('pew', 'mpi', 'cms', 'dhs')
            year: Target year
            fallback_to_nearest: If exact year not available, use nearest

        Returns:
            UnauthorizedEstimate or None
        """
        if source not in KNOWN_ESTIMATES:
            logger.warning(f"Unknown source: {source}")
            return None

        source_estimates = KNOWN_ESTIMATES[source]

        if year in source_estimates:
            return source_estimates[year]

        if fallback_to_nearest and source_estimates:
            # Find nearest year
            available_years = sorted(source_estimates.keys())
            nearest_year = min(available_years, key=lambda y: abs(y - year))
            estimate = source_estimates[nearest_year]
            logger.info(
                f"Using {source} estimate from {nearest_year} (nearest to requested {year})"
            )
            return estimate

        return None

    def get_primary_estimate(self, year: int) -> Optional[UnauthorizedEstimate]:
        """
        Get primary (Pew) estimate for a year.

        Args:
            year: Target year

        Returns:
            Pew estimate or fallback
        """
        return self._fetch_pew(year)

    def _fetch_pew(self, year: int) -> Optional[UnauthorizedEstimate]:
        """Fetch Pew estimate."""
        # Try to get from downloaded Pew data first
        try:
            from .. import config

            trends_path = self.cache_dir / "pew_state_trends.xlsx"

            if trends_path.exists():
                df = pd.read_excel(trends_path)
                # Parse California row for target year
                ca_rows = df[df.apply(lambda r: "california" in str(r.values).lower(), axis=1)]
                if not ca_rows.empty and str(year) in df.columns:
                    estimate_val = ca_rows[str(year)].iloc[0]
                    if pd.notna(estimate_val):
                        # Pew typically reports in thousands
                        estimate = int(estimate_val * 1000)
                        return UnauthorizedEstimate(
                            source="pew",
                            year=year,
                            estimate=estimate,
                            notes="Parsed from Pew state trends file",
                            url=config.PEW_STATE_TRENDS_URL,
                            methodology="Residual method applied to ACS/CPS",
                        )

        except Exception as e:
            logger.debug(f"Could not parse Pew file: {e}")

        # Fall back to known estimates
        return self.get_estimate("pew", year, fallback_to_nearest=True)

    def _fetch_mpi(self, year: int) -> Optional[UnauthorizedEstimate]:
        """Fetch MPI estimate."""
        # MPI data requires scraping their profile page
        # For now, use known estimates
        return self.get_estimate("mpi", year, fallback_to_nearest=True)

    def _fetch_cms(self, year: int) -> Optional[UnauthorizedEstimate]:
        """Fetch CMS estimate."""
        return self.get_estimate("cms", year, fallback_to_nearest=True)

    def _fetch_dhs(self, year: int) -> Optional[UnauthorizedEstimate]:
        """Fetch DHS estimate."""
        return self.get_estimate("dhs", year, fallback_to_nearest=True)


def get_calibration_target(
    year: int,
    source: str = "pew",
    fallback_to_config: bool = True,
) -> int:
    """
    Get unauthorized population calibration target.

    Convenience function for use in imputation pipeline.

    Args:
        year: Target year
        source: Preferred source ('pew', 'mpi', 'cms', 'dhs')
        fallback_to_config: Fall back to config constant if unavailable

    Returns:
        Calibration target (population count)

    Raises:
        ValueError: If no estimate available
    """
    from .. import config

    fetcher = UnauthorizedBenchmarkFetcher()
    estimate = fetcher.get_estimate(source, year, fallback_to_nearest=True)

    if estimate:
        return estimate.estimate

    if fallback_to_config:
        logger.warning(
            f"Using fallback calibration target from config: {config.PEW_CA_UNAUTHORIZED_2023:,}"
        )
        return config.PEW_CA_UNAUTHORIZED_2023

    raise ValueError(f"No calibration target available for {year} from {source}")


def list_available_benchmarks() -> dict[str, list[int]]:
    """
    List all available benchmark estimates.

    Returns:
        Dict of source -> list of available years
    """
    return {source: sorted(years.keys()) for source, years in KNOWN_ESTIMATES.items()}
