"""
SSA SSI recipient data downloader.

Downloads SSI (Supplemental Security Income) recipient counts from
Social Security Administration.

Source: SSA Statistical Supplements
https://www.ssa.gov/policy/docs/statcomps/ssi_sc/

Data includes:
- State and county recipient counts
- Age and category breakdowns
- Annual data
"""

import logging
from pathlib import Path

import pandas as pd

from .base import (
    DataSourceMetadata,
    DownloadOutcome,
    DownloadResult,
    PortalDownloader,
)

logger = logging.getLogger(__name__)

# SSA SSI data URLs (update annually)
SSI_BASE_URL = "https://www.ssa.gov/policy/docs/statcomps/ssi_sc/"
SSI_STATE_COUNTY_URL = "https://www.ssa.gov/policy/docs/statcomps/ssi_sc/2023/ca.html"


class SSIDownloader(PortalDownloader):
    """
    Download SSA SSI recipient data.

    Source: SSA Statistical Supplements
    Contains: Annual recipient counts by state and county
    """

    def __init__(self, data_dir: Path, year: int = 2023, **kwargs):
        """
        Initialize SSI downloader.

        Args:
            data_dir: Directory to store downloaded files
            year: Year of data to download
            **kwargs: Additional arguments for PortalDownloader
        """
        self.year = year
        # SSA provides data in HTML tables - we'll need to parse these
        url = f"https://www.ssa.gov/policy/docs/statcomps/ssi_sc/{year}/ca.html"

        super().__init__(
            data_dir=data_dir,
            url=url,
            filename=f"ssi_ca_{year}.html",
            name="ssi",
            description="SSI Recipients by State and County",
            expected_content_types=["text/html", "text/plain"],
            **kwargs,
        )

    def fetch(
        self,
        overwrite: bool = False,
        strict: bool = False,
    ) -> DownloadOutcome:
        """
        Fetch SSI data.

        Args:
            overwrite: Force re-download even if cached
            strict: Raise exception on failure if True

        Returns:
            DownloadOutcome with status and file path
        """
        dest_path = self.data_dir / self.filename

        # Check cache
        if not overwrite and self._is_cache_valid(dest_path):
            logger.info(f"Using cached SSI data: {dest_path}")
            return DownloadOutcome(
                status=DownloadResult.SKIPPED_CACHED,
                file_path=dest_path,
            )

        try:
            return super().fetch(overwrite=overwrite, strict=strict)
        except Exception as e:
            logger.warning(f"SSI fetch failed: {e}")
            if strict:
                raise
            return DownloadOutcome(
                status=DownloadResult.SKIPPED_UNAVAILABLE,
                error_message=str(e),
            )

    def clean(self, raw_path: Path) -> pd.DataFrame:
        """
        Clean SSI data into standardized format.

        Parses HTML tables from SSA website.

        Output columns:
        - year: Data year
        - county: County name (or "California" for statewide)
        - recipients_total: Total SSI recipients
        - recipients_aged: Aged recipients
        - recipients_blind_disabled: Blind/disabled recipients
        """
        try:
            # Try to parse HTML tables
            tables = pd.read_html(raw_path)

            if not tables:
                logger.warning("No tables found in SSI HTML file")
                return pd.DataFrame()

            # The first table typically has the state/county data
            df = tables[0]

            # Add year column
            df["year"] = self.year

            # Standardize column names
            column_mapping = {
                "County": "county",
                "Total": "recipients_total",
                "Aged": "recipients_aged",
                "Blind": "recipients_blind_disabled",
            }

            for source_col in list(df.columns):
                if isinstance(source_col, str):
                    for target_key, target_col in column_mapping.items():
                        if target_key.lower() in source_col.lower():
                            df = df.rename(columns={source_col: target_col})
                            break

            return df

        except Exception as e:
            logger.warning(f"Could not parse SSI HTML: {e}")
            return pd.DataFrame()

    def get_metadata(self) -> DataSourceMetadata:
        """Return metadata for SSI data source."""
        return DataSourceMetadata(
            name=f"ssi_{self.year}",
            source_url=self.url,
            description=f"SSI Recipients in California by County ({self.year})",
            update_frequency="annual",
            file_format="html",
            notes="Administrative data from Social Security Administration",
        )
