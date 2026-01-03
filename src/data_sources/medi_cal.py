"""
Medi-Cal administrative data downloader.

Downloads Medi-Cal enrollment data from California Health and Human Services.
Source: CHHS Open Data Portal (data.chhs.ca.gov)

Data includes:
- Adult Full Scope Expansion Programs
- County-by-month enrollment
- Age group breakdowns
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

# Known Medi-Cal dataset IDs on CHHS data portal
MEDI_CAL_DATASET_ID = "medi-cal-monthly-enrollment"
MEDI_CAL_URL = "https://data.chhs.ca.gov/dataset/medi-cal-monthly-enrollment"


class MediCalDownloader(CkanDownloader):
    """
    Download Medi-Cal administrative enrollment data.

    Source: CHHS via data.chhs.ca.gov
    Contains: Monthly enrollment by county, eligibility group, age
    """

    def __init__(self, data_dir: Path, **kwargs):
        """
        Initialize Medi-Cal downloader.

        Args:
            data_dir: Directory to store downloaded files
            **kwargs: Additional arguments for CkanDownloader
        """
        super().__init__(
            data_dir=data_dir,
            base_url="https://data.chhs.ca.gov",
            dataset_id=MEDI_CAL_DATASET_ID,
            **kwargs,
        )

    def fetch(
        self,
        overwrite: bool = False,
        strict: bool = False,
    ) -> DownloadOutcome:
        """
        Fetch Medi-Cal data.

        Args:
            overwrite: Force re-download even if cached
            strict: Raise exception on failure if True

        Returns:
            DownloadOutcome with status and file path
        """
        dest_path = self.data_dir / "medi_cal_enrollment.csv"

        # Check cache
        if not overwrite and self._is_cache_valid(dest_path):
            logger.info(f"Using cached Medi-Cal data: {dest_path}")
            return DownloadOutcome(
                status=DownloadResult.SKIPPED_CACHED,
                file_path=dest_path,
            )

        try:
            return super().fetch(overwrite=overwrite, strict=strict)
        except Exception as e:
            logger.warning(f"Medi-Cal fetch failed: {e}")
            if strict:
                raise
            return DownloadOutcome(
                status=DownloadResult.SKIPPED_UNAVAILABLE,
                error_message=str(e),
            )

    def clean(self, raw_path: Path) -> pd.DataFrame:
        """
        Clean Medi-Cal data into standardized format.

        Output columns:
        - year, month: Time period
        - county: County name
        - eligibility_group: Eligibility category
        - age_group: Age category
        - enrollment: Number enrolled
        """
        df = pd.read_csv(raw_path)

        # Standardize column names
        column_mapping = {
            "County": "county",
            "Month": "month",
            "Year": "year",
            "Eligibility": "eligibility_group",
            "Age": "age_group",
            "Enrollment": "enrollment",
            "Enrollees": "enrollment",
        }

        for source_col in list(df.columns):
            for target_key, target_col in column_mapping.items():
                if target_key.lower() in source_col.lower():
                    df = df.rename(columns={source_col: target_col})
                    break

        return df

    def get_metadata(self) -> DataSourceMetadata:
        """Return metadata for Medi-Cal data source."""
        return DataSourceMetadata(
            name="medi_cal",
            source_url=MEDI_CAL_URL,
            description="Medi-Cal Monthly Enrollment by County and Eligibility",
            update_frequency="monthly",
            file_format="csv",
            notes="Administrative data from California Department of Health Care Services",
        )
