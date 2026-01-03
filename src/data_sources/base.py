"""
Base classes for administrative data source downloaders.

Provides abstract interfaces and common functionality for fetching,
validating, and caching data from various portals and APIs.
"""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class DownloadResult(Enum):
    """Result status for download operations."""

    SUCCESS = "success"
    SKIPPED_CACHED = "skipped_cached"
    SKIPPED_UNAVAILABLE = "skipped_unavailable"
    FAILED = "failed"


@dataclass
class DataSourceMetadata:
    """Metadata for a data source."""

    name: str
    source_url: str
    description: str = ""
    update_frequency: str = "unknown"  # monthly, quarterly, annual
    last_updated: Optional[datetime] = None
    sha256_hash: Optional[str] = None
    file_format: str = "csv"
    file_size_bytes: Optional[int] = None
    row_count: Optional[int] = None
    column_names: list[str] = field(default_factory=list)
    notes: str = ""
    downloaded_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert datetime objects to ISO format strings
        for key in ["last_updated", "downloaded_at"]:
            if result[key] is not None:
                result[key] = result[key].isoformat()
        return result


@dataclass
class DownloadOutcome:
    """Outcome of a download operation."""

    status: DownloadResult
    file_path: Optional[Path] = None
    error_message: Optional[str] = None
    metadata: Optional[DataSourceMetadata] = None


class BaseDataSourceDownloader(ABC):
    """
    Abstract base class for data source downloaders.

    Implements: retries/backoff, content-type validation, sha256 hashing, caching.
    Following existing pattern from ACSPUMSDownloader, SIPPDownloader.
    """

    def __init__(
        self,
        data_dir: Path,
        max_retries: int = 3,
        base_delay: float = 1.0,
        timeout: int = 300,
        cache_days: int = 7,
    ):
        """
        Initialize downloader.

        Args:
            data_dir: Directory to store downloaded files
            max_retries: Maximum retry attempts for failed downloads
            base_delay: Base delay in seconds for exponential backoff
            timeout: Request timeout in seconds
            cache_days: Number of days to consider cached files valid
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.timeout = timeout
        self.cache_days = cache_days

    @abstractmethod
    def get_download_url(self, **kwargs) -> str:
        """Construct download URL for this data source."""
        pass

    @abstractmethod
    def fetch(
        self,
        overwrite: bool = False,
        strict: bool = False,
    ) -> DownloadOutcome:
        """
        Fetch data from source.

        Args:
            overwrite: Force re-download even if cached
            strict: If True, raise exception on failure; if False, log warning and skip

        Returns:
            DownloadOutcome with status and file path
        """
        pass

    @abstractmethod
    def clean(self, raw_path: Path) -> pd.DataFrame:
        """Clean raw data into standardized format."""
        pass

    @abstractmethod
    def get_metadata(self) -> DataSourceMetadata:
        """Return metadata for this data source."""
        pass

    def _is_cache_valid(self, file_path: Path) -> bool:
        """Check if cached file is still valid."""
        if not file_path.exists():
            return False

        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
        age_days = (datetime.now() - mtime).days
        return age_days < self.cache_days

    def _validate_content_type(
        self,
        response: requests.Response,
        expected_types: list[str],
    ) -> None:
        """
        Validate response content type.

        Args:
            response: HTTP response
            expected_types: List of acceptable content type substrings

        Raises:
            ValueError: If content type doesn't match any expected type
        """
        content_type = response.headers.get("content-type", "").lower()
        if not any(expected in content_type for expected in expected_types):
            raise ValueError(
                f"Unexpected content type: {content_type}. "
                f"Expected one of: {expected_types}"
            )

    def _compute_sha256(self, file_path: Path) -> str:
        """Compute SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _save_metadata(self, metadata: DataSourceMetadata) -> Path:
        """Save metadata JSON alongside data file."""
        metadata_path = self.data_dir / f"{metadata.name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)
        logger.debug(f"Saved metadata: {metadata_path}")
        return metadata_path

    def _download_with_retry(
        self,
        url: str,
        dest_path: Path,
        expected_content_types: Optional[list[str]] = None,
    ) -> None:
        """
        Download file with retry logic and exponential backoff.

        Args:
            url: URL to download from
            dest_path: Destination file path
            expected_content_types: Optional list of valid content types

        Raises:
            requests.HTTPError: On HTTP errors after retries exhausted
            requests.Timeout: On timeout after retries exhausted
            requests.ConnectionError: On connection errors after retries exhausted
        """
        import time

        last_exception: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    url,
                    timeout=self.timeout,
                    stream=True,
                )
                response.raise_for_status()

                if expected_content_types:
                    self._validate_content_type(response, expected_content_types)

                # Write to file
                with open(dest_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                logger.info(f"Downloaded: {dest_path}")
                return

            except (requests.HTTPError, requests.Timeout, requests.ConnectionError) as e:
                last_exception = e
                delay = self.base_delay * (2**attempt)
                logger.warning(
                    f"Download attempt {attempt + 1}/{self.max_retries} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)

        # All retries exhausted
        if last_exception:
            raise last_exception


class CkanDownloader(BaseDataSourceDownloader):
    """
    Generic CKAN API downloader for data.gov/data.ca.gov datasets.

    CKAN API endpoints:
    - Package metadata: {base}/api/3/action/package_show?id={dataset_id}
    - Resource download: Direct URL from resource metadata
    """

    def __init__(
        self,
        data_dir: Path,
        base_url: str,  # e.g., "https://catalog.data.gov" or "https://data.ca.gov"
        dataset_id: str,
        resource_name: Optional[str] = None,  # If multiple resources, specify which
        **kwargs,
    ):
        """
        Initialize CKAN downloader.

        Args:
            data_dir: Directory to store downloaded files
            base_url: Base URL of CKAN portal (e.g., "https://data.ca.gov")
            dataset_id: CKAN dataset/package ID
            resource_name: Optional resource name filter
            **kwargs: Additional arguments for BaseDataSourceDownloader
        """
        super().__init__(data_dir, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.dataset_id = dataset_id
        self.resource_name = resource_name
        self._package_metadata: Optional[dict] = None

    def get_package_metadata(self, refresh: bool = False) -> dict:
        """
        Fetch CKAN package metadata.

        Args:
            refresh: Force refresh of cached metadata

        Returns:
            Package metadata dictionary

        Raises:
            requests.HTTPError: On HTTP errors
            ValueError: If API returns error
        """
        if self._package_metadata is not None and not refresh:
            return self._package_metadata

        url = f"{self.base_url}/api/3/action/package_show?id={self.dataset_id}"
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()

        result = response.json()
        if not result.get("success"):
            raise ValueError(f"CKAN API error: {result.get('error')}")

        self._package_metadata = result["result"]
        return self._package_metadata

    def get_download_url(self, **kwargs) -> str:
        """
        Get resource download URL from CKAN metadata.

        Returns:
            Direct download URL for the resource

        Raises:
            ValueError: If no suitable resource found
        """
        metadata = self.get_package_metadata()
        resources = metadata.get("resources", [])

        if not resources:
            raise ValueError(f"No resources found for dataset: {self.dataset_id}")

        # Find matching resource
        if self.resource_name:
            for res in resources:
                if self.resource_name.lower() in res.get("name", "").lower():
                    return res["url"]
            raise ValueError(f"Resource '{self.resource_name}' not found")

        # Default to first CSV/Excel resource
        for res in resources:
            fmt = res.get("format", "").lower()
            if fmt in ("csv", "xlsx", "xls"):
                return res["url"]

        # Fall back to first resource
        return resources[0]["url"]

    def fetch(
        self,
        overwrite: bool = False,
        strict: bool = False,
    ) -> DownloadOutcome:
        """Fetch data from CKAN portal."""
        dest_path = self.data_dir / f"{self.dataset_id}.csv"

        # Check cache
        if not overwrite and self._is_cache_valid(dest_path):
            logger.info(f"Using cached data: {dest_path}")
            return DownloadOutcome(
                status=DownloadResult.SKIPPED_CACHED,
                file_path=dest_path,
            )

        try:
            url = self.get_download_url()
            self._download_with_retry(
                url=url,
                dest_path=dest_path,
                expected_content_types=[
                    "text/csv",
                    "application/csv",
                    "application/vnd.ms-excel",
                    "application/vnd.openxmlformats",
                ],
            )

            # Compute and save metadata
            sha256 = self._compute_sha256(dest_path)
            metadata = self.get_metadata()
            metadata.sha256_hash = sha256
            metadata.downloaded_at = datetime.now()
            metadata.file_size_bytes = dest_path.stat().st_size
            self._save_metadata(metadata)

            return DownloadOutcome(
                status=DownloadResult.SUCCESS,
                file_path=dest_path,
                metadata=metadata,
            )

        except requests.HTTPError as e:
            if hasattr(e, "response") and e.response is not None:
                if e.response.status_code == 403:
                    logger.warning(f"Access forbidden (403) for {self.dataset_id}: {e}")
                elif e.response.status_code == 404:
                    logger.warning(f"Not found (404) for {self.dataset_id}: URL may have changed")
                else:
                    logger.warning(f"HTTP error for {self.dataset_id}: {e}")
            else:
                logger.warning(f"HTTP error for {self.dataset_id}: {e}")

            if strict:
                raise
            return DownloadOutcome(
                status=DownloadResult.SKIPPED_UNAVAILABLE,
                error_message=str(e),
            )

        except requests.Timeout as e:
            logger.warning(f"Timeout fetching {self.dataset_id}: {e}")
            if strict:
                raise
            return DownloadOutcome(
                status=DownloadResult.FAILED,
                error_message=str(e),
            )

        except requests.ConnectionError as e:
            logger.warning(f"Connection error fetching {self.dataset_id}: {e}")
            if strict:
                raise
            return DownloadOutcome(
                status=DownloadResult.FAILED,
                error_message=str(e),
            )

        except ValueError as e:
            logger.warning(f"Validation error for {self.dataset_id}: {e}")
            if strict:
                raise
            return DownloadOutcome(
                status=DownloadResult.FAILED,
                error_message=str(e),
            )

    def clean(self, raw_path: Path) -> pd.DataFrame:
        """
        Clean raw data into standardized format.

        Override in subclasses for source-specific cleaning logic.
        """
        return pd.read_csv(raw_path)

    def get_metadata(self) -> DataSourceMetadata:
        """Return metadata for this data source."""
        try:
            pkg = self.get_package_metadata()
            return DataSourceMetadata(
                name=self.dataset_id,
                source_url=f"{self.base_url}/dataset/{self.dataset_id}",
                description=pkg.get("notes", ""),
                update_frequency=pkg.get("update_frequency", "unknown"),
            )
        except Exception:
            return DataSourceMetadata(
                name=self.dataset_id,
                source_url=f"{self.base_url}/dataset/{self.dataset_id}",
            )


class PortalDownloader(BaseDataSourceDownloader):
    """
    Generic portal downloader for state agency websites.

    Handles direct file downloads with content-type validation.
    """

    def __init__(
        self,
        data_dir: Path,
        url: str,
        filename: str,
        name: str,
        description: str = "",
        expected_content_types: Optional[list[str]] = None,
        **kwargs,
    ):
        """
        Initialize portal downloader.

        Args:
            data_dir: Directory to store downloaded files
            url: Direct download URL
            filename: Local filename to save as
            name: Data source name
            description: Data source description
            expected_content_types: List of valid content types
            **kwargs: Additional arguments for BaseDataSourceDownloader
        """
        super().__init__(data_dir, **kwargs)
        self.url = url
        self.filename = filename
        self.name = name
        self.description = description
        self.expected_content_types = expected_content_types or [
            "text/csv",
            "application/csv",
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/octet-stream",
        ]

    def get_download_url(self, **kwargs) -> str:
        """Return the configured URL."""
        return self.url

    def fetch(
        self,
        overwrite: bool = False,
        strict: bool = False,
    ) -> DownloadOutcome:
        """Fetch data from portal."""
        dest_path = self.data_dir / self.filename

        # Check cache
        if not overwrite and self._is_cache_valid(dest_path):
            logger.info(f"Using cached data: {dest_path}")
            return DownloadOutcome(
                status=DownloadResult.SKIPPED_CACHED,
                file_path=dest_path,
            )

        try:
            self._download_with_retry(
                url=self.url,
                dest_path=dest_path,
                expected_content_types=self.expected_content_types,
            )

            # Compute and save metadata
            sha256 = self._compute_sha256(dest_path)
            metadata = self.get_metadata()
            metadata.sha256_hash = sha256
            metadata.downloaded_at = datetime.now()
            metadata.file_size_bytes = dest_path.stat().st_size
            self._save_metadata(metadata)

            return DownloadOutcome(
                status=DownloadResult.SUCCESS,
                file_path=dest_path,
                metadata=metadata,
            )

        except requests.HTTPError as e:
            if hasattr(e, "response") and e.response is not None:
                if e.response.status_code == 403:
                    logger.warning(f"Access forbidden (403) for {self.name}: {e}")
                elif e.response.status_code == 404:
                    logger.warning(f"Not found (404) for {self.name}: URL may have changed")
                else:
                    logger.warning(f"HTTP error for {self.name}: {e}")
            else:
                logger.warning(f"HTTP error for {self.name}: {e}")

            if strict:
                raise
            return DownloadOutcome(
                status=DownloadResult.SKIPPED_UNAVAILABLE,
                error_message=str(e),
            )

        except requests.Timeout as e:
            logger.warning(f"Timeout fetching {self.name}: {e}")
            if strict:
                raise
            return DownloadOutcome(
                status=DownloadResult.FAILED,
                error_message=str(e),
            )

        except requests.ConnectionError as e:
            logger.warning(f"Connection error fetching {self.name}: {e}")
            if strict:
                raise
            return DownloadOutcome(
                status=DownloadResult.FAILED,
                error_message=str(e),
            )

    def clean(self, raw_path: Path) -> pd.DataFrame:
        """
        Clean raw data into standardized format.

        Override in subclasses for source-specific cleaning logic.
        """
        if raw_path.suffix.lower() in (".xlsx", ".xls"):
            return pd.read_excel(raw_path)
        return pd.read_csv(raw_path)

    def get_metadata(self) -> DataSourceMetadata:
        """Return metadata for this data source."""
        return DataSourceMetadata(
            name=self.name,
            source_url=self.url,
            description=self.description,
        )
