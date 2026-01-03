"""
Data download utilities for fetching Census and other public data.
"""

import hashlib
import logging
import time
import zipfile
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


def download_with_retry(
    url: str,
    dest_path: Path,
    max_retries: int = 3,
    base_delay: float = 1.0,
    chunk_size: int = 8192,
    timeout: int = 300,
    overwrite: bool = False,
) -> Path:
    """
    Download a file with exponential backoff retry on failure.

    Args:
        url: URL to download from
        dest_path: Destination file path
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds (doubles each retry)
        chunk_size: Download chunk size in bytes
        timeout: Request timeout in seconds
        overwrite: If True, overwrite existing file

    Returns:
        Path to downloaded file

    Raises:
        requests.HTTPError: If download fails after all retries
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            return download_file(
                url=url,
                dest_path=dest_path,
                chunk_size=chunk_size,
                timeout=timeout,
                overwrite=overwrite,
            )
        except (requests.HTTPError, requests.Timeout, requests.ConnectionError) as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                logger.warning(
                    f"Download failed (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {delay:.1f}s: {e}"
                )
                time.sleep(delay)
            else:
                logger.error(f"Download failed after {max_retries} attempts: {e}")

    # Clean up partial download if it exists
    if dest_path.exists():
        try:
            dest_path.unlink()
            logger.info(f"Cleaned up partial download: {dest_path}")
        except OSError as e:
            logger.error(f"Failed to clean up partial download {dest_path}: {e}")
            # Re-raise to ensure caller knows cleanup failed
            raise RuntimeError(
                f"Download failed and cleanup of {dest_path} also failed: {e}"
            ) from last_exception

    raise last_exception


def download_file(
    url: str,
    dest_path: Path,
    chunk_size: int = 8192,
    timeout: int = 300,
    overwrite: bool = False,
) -> Path:
    """
    Download a file from URL with progress bar.

    Args:
        url: URL to download from
        dest_path: Destination file path
        chunk_size: Download chunk size in bytes
        timeout: Request timeout in seconds
        overwrite: If True, overwrite existing file

    Returns:
        Path to downloaded file

    Raises:
        requests.HTTPError: If download fails
    """
    dest_path = Path(dest_path)

    if dest_path.exists() and not overwrite:
        logger.info(f"File already exists, skipping: {dest_path}")
        return dest_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading: {url}")

    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f:
        with tqdm(
            total=total_size,
            unit="iB",
            unit_scale=True,
            desc=dest_path.name,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                size = f.write(chunk)
                pbar.update(size)

    logger.info(f"Downloaded: {dest_path} ({dest_path.stat().st_size:,} bytes)")
    return dest_path


def extract_zip(
    zip_path: Path,
    dest_dir: Path,
    remove_zip: bool = False,
) -> list[Path]:
    """
    Extract a ZIP file.

    Args:
        zip_path: Path to ZIP file
        dest_dir: Destination directory
        remove_zip: If True, delete ZIP after extraction

    Returns:
        List of extracted file paths
    """
    zip_path = Path(zip_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting: {zip_path} -> {dest_dir}")

    extracted = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            # Security check: prevent path traversal attacks
            member_path = (dest_dir / member).resolve()
            if not str(member_path).startswith(str(dest_dir.resolve())):
                raise ValueError(f"Path traversal detected in ZIP member: {member}")
            zf.extract(member, dest_dir)
            extracted.append(dest_dir / member)

    if remove_zip:
        zip_path.unlink()
        logger.info(f"Removed ZIP: {zip_path}")

    return extracted


def compute_checksum(file_path: Path, algorithm: str = "sha256") -> str:
    """
    Compute file checksum.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm (sha256, md5, etc.)

    Returns:
        Hex digest of checksum
    """
    hasher = hashlib.new(algorithm)

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def check_url_exists(url: str, timeout: int = 30) -> bool:
    """
    Check if a URL exists (returns 200 for HEAD request).

    Args:
        url: URL to check
        timeout: Request timeout in seconds

    Returns:
        True if URL exists, False otherwise
    """
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        return response.status_code == 200
    except requests.RequestException:
        return False


class ACSPUMSDownloader:
    """Download ACS PUMS data from Census Bureau FTP."""

    BASE_URL = "https://www2.census.gov/programs-surveys/acs/data/pums"

    def __init__(self, raw_data_dir: Path):
        self.raw_data_dir = Path(raw_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

    def get_download_url(
        self,
        year: int,
        file_type: str,
        state: str = "ca",
        survey: str = "1-Year",
    ) -> str:
        """
        Construct download URL for ACS PUMS file.

        Args:
            year: Survey year
            file_type: 'person' or 'housing'
            state: State abbreviation (lowercase)
            survey: '1-Year' or '5-Year'

        Returns:
            Full download URL
        """
        prefix = "p" if file_type == "person" else "h"
        filename = f"csv_{prefix}{state}.zip"
        return f"{self.BASE_URL}/{year}/{survey}/{filename}"

    def download_state_data(
        self,
        year: int,
        state: str = "ca",
        survey: str = "1-Year",
        overwrite: bool = False,
    ) -> dict[str, Path]:
        """
        Download person and housing files for a state.

        Args:
            year: Survey year
            state: State abbreviation
            survey: '1-Year' or '5-Year'
            overwrite: Overwrite existing files

        Returns:
            Dict with 'person' and 'housing' file paths
        """
        result = {}

        for file_type in ["person", "housing"]:
            url = self.get_download_url(year, file_type, state, survey)

            # Check if URL exists
            if not check_url_exists(url):
                logger.warning(f"URL not found: {url}")
                continue

            # Download
            zip_name = f"acs_{year}_{survey.lower().replace('-', '')}_{file_type}_{state}.zip"
            zip_path = self.raw_data_dir / zip_name

            download_file(url, zip_path, overwrite=overwrite)

            # Extract
            extract_dir = (
                self.raw_data_dir / f"acs_{year}_{survey.lower().replace('-', '')}_{state}"
            )
            extracted = extract_zip(zip_path, extract_dir)

            # Find CSV file
            csv_files = [f for f in extracted if f.suffix.lower() == ".csv"]
            if csv_files:
                result[file_type] = csv_files[0]
            else:
                logger.warning(f"No CSV found in: {zip_path}")

        return result

    def find_available_year(
        self,
        preferred_years: list[int],
        state: str = "ca",
        survey: str = "1-Year",
    ) -> Optional[int]:
        """
        Find first available year from preference list.

        Args:
            preferred_years: Years to try in order
            state: State abbreviation
            survey: Survey type

        Returns:
            First available year, or None if none found
        """
        for year in preferred_years:
            url = self.get_download_url(year, "person", state, survey)
            if check_url_exists(url):
                logger.info(f"Found available ACS PUMS year: {year}")
                return year

        logger.warning(f"No ACS PUMS data found for years: {preferred_years}")
        return None


class SIPPDownloader:
    """Download SIPP data from Census Bureau."""

    BASE_URL = "https://www2.census.gov/programs-surveys/sipp/data/datasets"

    def __init__(self, raw_data_dir: Path):
        self.raw_data_dir = Path(raw_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

    def list_available_years(self) -> list[int]:
        """List available SIPP years (requires parsing directory listing)."""
        # This would need to parse the FTP directory
        # For now, return known recent years
        return [2024, 2023, 2022, 2021, 2020, 2019, 2018]

    def download_sipp_data(
        self,
        year: int,
        overwrite: bool = False,
    ) -> Optional[Path]:
        """
        Download SIPP data for a given year.

        Note: SIPP file structure varies by year. This handles recent years.

        Args:
            year: SIPP panel year
            overwrite: Overwrite existing files

        Returns:
            Path to downloaded/extracted data, or None if not found
        """
        # SIPP 2022+ uses a different structure
        # Example: https://www2.census.gov/programs-surveys/sipp/data/datasets/2022/

        year_url = f"{self.BASE_URL}/{year}/"

        # Try to find the main data file
        # Common patterns: pu2022.zip, sipp2022.zip, etc.
        possible_names = [
            f"pu{year}.zip",
            f"sipp{year}.zip",
            f"pu{year}_csv.zip",
            f"sipp{year}_csv.zip",
        ]

        for name in possible_names:
            url = f"{year_url}{name}"
            if check_url_exists(url):
                dest_path = self.raw_data_dir / f"sipp_{year}" / name
                download_file(url, dest_path, overwrite=overwrite)

                # Extract if ZIP
                if name.endswith(".zip"):
                    extract_dir = dest_path.parent
                    extract_zip(dest_path, extract_dir)

                return dest_path.parent

        logger.warning(f"Could not find SIPP data for year {year}")
        return None


class PewDataDownloader:
    """Download Pew Research unauthorized immigrant estimates."""

    def __init__(self, external_data_dir: Path):
        self.external_data_dir = Path(external_data_dir)
        self.external_data_dir.mkdir(parents=True, exist_ok=True)

    def download_state_trends(
        self,
        url: str,
        overwrite: bool = False,
    ) -> Optional[Path]:
        """
        Download Pew state trends Excel file.

        Args:
            url: Direct URL to Excel file
            overwrite: Overwrite existing file

        Returns:
            Path to downloaded file, or None if failed
        """
        dest_path = self.external_data_dir / "pew_state_trends.xlsx"

        try:
            download_file(url, dest_path, overwrite=overwrite)
            return dest_path
        except requests.HTTPError as e:
            logger.error(f"Failed to download Pew data: {e}")
            logger.info("Manual download may be required. See docs/manual_download.md")
            return None

    def download_labor_force(
        self,
        url: str,
        overwrite: bool = False,
    ) -> Optional[Path]:
        """
        Download Pew labor force Excel file.

        Args:
            url: Direct URL to Excel file
            overwrite: Overwrite existing file

        Returns:
            Path to downloaded file, or None if failed
        """
        dest_path = self.external_data_dir / "pew_labor_force_by_state.xlsx"

        try:
            download_file(url, dest_path, overwrite=overwrite)
            return dest_path
        except requests.HTTPError as e:
            logger.error(f"Failed to download Pew labor force data: {e}")
            return None
