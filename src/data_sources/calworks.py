"""
CalWORKs data downloader.

Downloads CalWORKs (TANF) program data from California Data Portal.
Source: CDSS via data.ca.gov / catalog.data.gov

Data includes:
- Monthly applications and approvals
- Case counts by county
- Benefit amounts
"""

import logging
from pathlib import Path

import pandas as pd

from .base import (
    CkanDownloader,
    DataSourceMetadata,
    DownloadOutcome,
    DownloadResult,
)

logger = logging.getLogger(__name__)

# Known CalWORKs dataset ID
CALWORKS_DATASET_ID = "calworks"
CALWORKS_URL = "https://catalog.data.gov/dataset/calworks"


class CalWORKsDownloader(CkanDownloader):
    """
    Download CalWORKs monthly program data.

    Source: CDSS via catalog.data.gov
    Contains: Monthly applications, cases, benefit amounts by county
    """

    def __init__(self, data_dir: Path, **kwargs):
        """
        Initialize CalWORKs downloader.

        Args:
            data_dir: Directory to store downloaded files
            **kwargs: Additional arguments for CkanDownloader
        """
        super().__init__(
            data_dir=data_dir,
            base_url="https://catalog.data.gov",
            dataset_id=CALWORKS_DATASET_ID,
            **kwargs,
        )

    def fetch(
        self,
        overwrite: bool = False,
        strict: bool = False,
    ) -> DownloadOutcome:
        """
        Fetch CalWORKs data.

        Args:
            overwrite: Force re-download even if cached
            strict: Raise exception on failure if True

        Returns:
            DownloadOutcome with status and file path
        """
        dest_path = self.data_dir / "calworks.csv"

        # Check cache
        if not overwrite and self._is_cache_valid(dest_path):
            logger.info(f"Using cached CalWORKs data: {dest_path}")
            return DownloadOutcome(
                status=DownloadResult.SKIPPED_CACHED,
                file_path=dest_path,
            )

        try:
            return super().fetch(overwrite=overwrite, strict=strict)
        except Exception as e:
            logger.warning(f"CalWORKs fetch failed: {e}")
            if strict:
                raise
            return DownloadOutcome(
                status=DownloadResult.SKIPPED_UNAVAILABLE,
                error_message=str(e),
            )

    def clean(self, raw_path: Path) -> pd.DataFrame:
        """
        Clean CalWORKs data into standardized format.

        Output columns:
        - year, month: Time period
        - county: County name
        - applications: Number of applications
        - approvals: Number of approvals
        - cases: Active case count
        - recipients: Total recipients
        """
        df = pd.read_csv(raw_path)

        # Standardize column names
        column_mapping = {
            "County": "county",
            "Month": "month",
            "Year": "year",
            "Applications": "applications",
            "Approvals": "approvals",
            "Cases": "cases",
            "Recipients": "recipients",
        }

        for source_col in list(df.columns):
            for target_key, target_col in column_mapping.items():
                if target_key.lower() in source_col.lower():
                    df = df.rename(columns={source_col: target_col})
                    break

        return df

    def get_metadata(self) -> DataSourceMetadata:
        """Return metadata for CalWORKs data source."""
        return DataSourceMetadata(
            name="calworks",
            source_url=CALWORKS_URL,
            description="CalWORKs Monthly Applications and Cases",
            update_frequency="monthly",
            file_format="csv",
            notes="Administrative data from California Department of Social Services",
        )
