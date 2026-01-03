"""
Administrative data source registry.

Provides unified interface for fetching all administrative data sources
with consistent error handling and reporting.
"""

import logging
from pathlib import Path
from typing import Optional

from .base import BaseDataSourceDownloader, DownloadOutcome, DownloadResult
from .calfresh import CalFreshDownloader
from .calworks import CalWORKsDownloader
from .medi_cal import MediCalDownloader
from .ssi import SSIDownloader

logger = logging.getLogger(__name__)


class AdminDataSourceRegistry:
    """
    Registry for managing administrative data sources.

    Provides unified interface for fetching all sources with
    consistent error handling and reporting.
    """

    SOURCES: dict[str, type[BaseDataSourceDownloader]] = {
        "calfresh": CalFreshDownloader,
        "calworks": CalWORKsDownloader,
        "medi_cal": MediCalDownloader,
        "ssi": SSIDownloader,
    }

    def __init__(
        self,
        data_dir: Path,
        strict: bool = False,
        ssi_year: int = 2023,
    ):
        """
        Initialize registry.

        Args:
            data_dir: Base directory for admin data storage
            strict: If True, raise exceptions on failures
            ssi_year: Year for SSI data download
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.strict = strict
        self.ssi_year = ssi_year

        self._downloaders: dict[str, BaseDataSourceDownloader] = {}
        self._results: dict[str, DownloadOutcome] = {}

    def _create_downloader(self, name: str) -> BaseDataSourceDownloader:
        """Create downloader instance for a data source."""
        source_dir = self.data_dir / name
        source_dir.mkdir(parents=True, exist_ok=True)

        if name == "ssi":
            return SSIDownloader(source_dir, year=self.ssi_year)
        else:
            return self.SOURCES[name](source_dir)

    def fetch_all(
        self,
        overwrite: bool = False,
        sources: Optional[list[str]] = None,
    ) -> dict[str, DownloadOutcome]:
        """
        Fetch all registered data sources.

        Args:
            overwrite: Force re-download even if cached
            sources: Optional list of source names to fetch (default: all)

        Returns:
            Dict of source_name -> DownloadOutcome
        """
        sources_to_fetch = sources or list(self.SOURCES.keys())

        for name in sources_to_fetch:
            if name not in self.SOURCES:
                logger.warning(f"Unknown source: {name}")
                continue

            logger.info(f"Fetching {name}...")

            try:
                downloader = self._create_downloader(name)
                self._downloaders[name] = downloader

                outcome = downloader.fetch(overwrite=overwrite, strict=self.strict)
                self._results[name] = outcome

                if outcome.status == DownloadResult.SUCCESS:
                    logger.info(f"  {name}: Downloaded successfully")
                elif outcome.status == DownloadResult.SKIPPED_CACHED:
                    logger.info(f"  {name}: Using cached data")
                elif outcome.status == DownloadResult.SKIPPED_UNAVAILABLE:
                    logger.warning(f"  {name}: Skipped (unavailable)")
                elif outcome.status == DownloadResult.FAILED:
                    logger.warning(f"  {name}: Failed - {outcome.error_message}")

            except Exception as e:
                logger.error(f"Error fetching {name}: {e}")
                self._results[name] = DownloadOutcome(
                    status=DownloadResult.FAILED,
                    error_message=str(e),
                )
                if self.strict:
                    raise

        return self._results

    def fetch_one(
        self,
        name: str,
        overwrite: bool = False,
    ) -> DownloadOutcome:
        """
        Fetch a single data source.

        Args:
            name: Source name ('calfresh', 'calworks', 'medi_cal', 'ssi')
            overwrite: Force re-download even if cached

        Returns:
            DownloadOutcome for the source
        """
        if name not in self.SOURCES:
            raise ValueError(f"Unknown source: {name}. Available: {list(self.SOURCES.keys())}")

        results = self.fetch_all(overwrite=overwrite, sources=[name])
        return results[name]

    def get_summary(self) -> dict[str, dict]:
        """
        Return summary of fetch results.

        Returns:
            Dict with status, file path, and error for each source
        """
        return {
            name: {
                "status": result.status.value,
                "file_path": str(result.file_path) if result.file_path else None,
                "error": result.error_message,
            }
            for name, result in self._results.items()
        }

    def clean_all(self) -> dict[str, "pd.DataFrame"]:
        """
        Clean all successfully downloaded data sources.

        Returns:
            Dict of source_name -> cleaned DataFrame
        """
        import pandas as pd

        cleaned = {}

        for name, outcome in self._results.items():
            if outcome.status in (DownloadResult.SUCCESS, DownloadResult.SKIPPED_CACHED):
                if outcome.file_path and outcome.file_path.exists():
                    try:
                        downloader = self._downloaders.get(name)
                        if downloader:
                            df = downloader.clean(outcome.file_path)
                            cleaned[name] = df
                            logger.info(f"Cleaned {name}: {len(df)} rows")
                    except Exception as e:
                        logger.warning(f"Could not clean {name}: {e}")

        return cleaned

    def get_available_sources(self) -> list[str]:
        """Return list of available data source names."""
        return list(self.SOURCES.keys())
