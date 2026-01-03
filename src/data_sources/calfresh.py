"""
CalFresh DFA256 data downloader.

Downloads CalFresh (SNAP) participation data from California Data Portal.
Source: CDSS via data.ca.gov

Data includes:
- Monthly participation counts
- Benefit issuance by county
- Household counts
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .base import (
    CkanDownloader,
    DataSourceMetadata,
    DownloadOutcome,
    DownloadResult,
)

logger = logging.getLogger(__name__)

# Known CalFresh dataset IDs on data.ca.gov
# These may need updating as the portal structure changes
CALFRESH_DATASET_ID = "calfresh-data"
CALFRESH_DFA256_URL = (
    "https://data.ca.gov/dataset/"
    "calfresh-dfa-256-monthly-participation-and-benefit-issuance"
)


class CalFreshDownloader(CkanDownloader):
    """
    Download CalFresh DFA256 monthly participation data.

    Source: CDSS via data.ca.gov
    Contains: Monthly participation counts, benefit issuance by county
    """

    def __init__(self, data_dir: Path, **kwargs):
        """
        Initialize CalFresh downloader.

        Args:
            data_dir: Directory to store downloaded files
            **kwargs: Additional arguments for CkanDownloader
        """
        super().__init__(
            data_dir=data_dir,
            base_url="https://data.ca.gov",
            dataset_id=CALFRESH_DATASET_ID,
            **kwargs,
        )
        self._direct_url: Optional[str] = None

    def fetch(
        self,
        overwrite: bool = False,
        strict: bool = False,
    ) -> DownloadOutcome:
        """
        Fetch CalFresh data with graceful degradation.

        Args:
            overwrite: Force re-download even if cached
            strict: Raise exception on failure if True

        Returns:
            DownloadOutcome with status and file path
        """
        dest_path = self.data_dir / "calfresh_dfa256.csv"

        # Check cache
        if not overwrite and self._is_cache_valid(dest_path):
            logger.info(f"Using cached CalFresh data: {dest_path}")
            return DownloadOutcome(
                status=DownloadResult.SKIPPED_CACHED,
                file_path=dest_path,
            )

        # Try CKAN API first, fall back to direct URL if needed
        try:
            return super().fetch(overwrite=overwrite, strict=strict)
        except Exception as e:
            logger.warning(f"CKAN fetch failed, trying direct URL: {e}")

            if self._direct_url:
                try:
                    self._download_with_retry(
                        url=self._direct_url,
                        dest_path=dest_path,
                    )
                    metadata = self.get_metadata()
                    metadata.sha256_hash = self._compute_sha256(dest_path)
                    self._save_metadata(metadata)

                    return DownloadOutcome(
                        status=DownloadResult.SUCCESS,
                        file_path=dest_path,
                        metadata=metadata,
                    )
                except Exception as e2:
                    logger.warning(f"Direct URL fetch also failed: {e2}")

            if strict:
                raise
            return DownloadOutcome(
                status=DownloadResult.SKIPPED_UNAVAILABLE,
                error_message=str(e),
            )

    def clean(self, raw_path: Path) -> pd.DataFrame:
        """
        Clean CalFresh data into standardized format.

        Output columns:
        - year, month: Time period
        - county: County name (or "California" for statewide)
        - participants: Number of participants
        - households: Number of households
        - benefits_issued: Total benefits in dollars
        """
        df = pd.read_csv(raw_path)

        # Standardize column names
        # Actual mapping depends on source format - adjust as needed
        column_mapping = {
            "County": "county",
            "Month": "month",
            "Year": "year",
            "Persons": "participants",
            "Households": "households",
            "Issuance": "benefits_issued",
        }

        # Find matching columns (case-insensitive)
        for source_col in list(df.columns):
            for target_key, target_col in column_mapping.items():
                if target_key.lower() in source_col.lower():
                    df = df.rename(columns={source_col: target_col})
                    break

        # Add year/month if not present
        if "year" not in df.columns and "date" in df.columns:
            df["year"] = pd.to_datetime(df["date"]).dt.year
            df["month"] = pd.to_datetime(df["date"]).dt.month

        return df

    def get_metadata(self) -> DataSourceMetadata:
        """Return metadata for CalFresh data source."""
        return DataSourceMetadata(
            name="calfresh_dfa256",
            source_url=CALFRESH_DFA256_URL,
            description="CalFresh DFA256 Monthly Participation and Benefit Issuance",
            update_frequency="monthly",
            file_format="csv",
            notes="Administrative data from California Department of Social Services",
        )
