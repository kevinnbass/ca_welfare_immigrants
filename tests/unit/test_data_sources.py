"""
Tests for administrative data source downloaders.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data_sources.base import (
    BaseDataSourceDownloader,
    CkanDownloader,
    DataSourceMetadata,
    DownloadOutcome,
    DownloadResult,
    PortalDownloader,
)


class TestDataSourceMetadata:
    """Tests for DataSourceMetadata dataclass."""

    def test_metadata_creation(self):
        """Test basic metadata creation."""
        meta = DataSourceMetadata(
            name="test_dataset",
            source_url="https://example.com/data.csv",
            description="Test dataset",
        )
        assert meta.name == "test_dataset"
        assert meta.source_url == "https://example.com/data.csv"
        assert meta.update_frequency == "unknown"

    def test_metadata_to_dict(self):
        """Test metadata serialization."""
        meta = DataSourceMetadata(
            name="test",
            source_url="https://example.com",
        )
        result = meta.to_dict()
        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["last_updated"] is None


class TestDownloadResult:
    """Tests for DownloadResult enum."""

    def test_result_values(self):
        """Test enum values."""
        assert DownloadResult.SUCCESS.value == "success"
        assert DownloadResult.SKIPPED_CACHED.value == "skipped_cached"
        assert DownloadResult.FAILED.value == "failed"


class TestBaseDataSourceDownloader:
    """Tests for BaseDataSourceDownloader."""

    def test_cache_validation_missing_file(self):
        """Test cache validation for missing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a concrete implementation for testing
            class TestDownloader(BaseDataSourceDownloader):
                def get_download_url(self, **kwargs):
                    return "https://example.com"

                def fetch(self, overwrite=False, strict=False):
                    return DownloadOutcome(status=DownloadResult.SUCCESS)

                def clean(self, raw_path):
                    return None

                def get_metadata(self):
                    return DataSourceMetadata(name="test", source_url="https://example.com")

            downloader = TestDownloader(Path(tmpdir))
            nonexistent = Path(tmpdir) / "nonexistent.csv"
            assert not downloader._is_cache_valid(nonexistent)

    def test_sha256_computation(self):
        """Test SHA256 hash computation."""
        with tempfile.TemporaryDirectory() as tmpdir:

            class TestDownloader(BaseDataSourceDownloader):
                def get_download_url(self, **kwargs):
                    return "https://example.com"

                def fetch(self, overwrite=False, strict=False):
                    return DownloadOutcome(status=DownloadResult.SUCCESS)

                def clean(self, raw_path):
                    return None

                def get_metadata(self):
                    return DataSourceMetadata(name="test", source_url="https://example.com")

            downloader = TestDownloader(Path(tmpdir))

            # Create a test file
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("test content")

            hash_result = downloader._compute_sha256(test_file)
            assert isinstance(hash_result, str)
            assert len(hash_result) == 64  # SHA256 produces 64-char hex string

    def test_content_type_validation_valid(self):
        """Test content type validation with valid type."""
        with tempfile.TemporaryDirectory() as tmpdir:

            class TestDownloader(BaseDataSourceDownloader):
                def get_download_url(self, **kwargs):
                    return "https://example.com"

                def fetch(self, overwrite=False, strict=False):
                    return DownloadOutcome(status=DownloadResult.SUCCESS)

                def clean(self, raw_path):
                    return None

                def get_metadata(self):
                    return DataSourceMetadata(name="test", source_url="https://example.com")

            downloader = TestDownloader(Path(tmpdir))

            mock_response = MagicMock()
            mock_response.headers = {"content-type": "text/csv; charset=utf-8"}

            # Should not raise
            downloader._validate_content_type(mock_response, ["text/csv", "application/csv"])

    def test_content_type_validation_invalid(self):
        """Test content type validation with invalid type."""
        with tempfile.TemporaryDirectory() as tmpdir:

            class TestDownloader(BaseDataSourceDownloader):
                def get_download_url(self, **kwargs):
                    return "https://example.com"

                def fetch(self, overwrite=False, strict=False):
                    return DownloadOutcome(status=DownloadResult.SUCCESS)

                def clean(self, raw_path):
                    return None

                def get_metadata(self):
                    return DataSourceMetadata(name="test", source_url="https://example.com")

            downloader = TestDownloader(Path(tmpdir))

            mock_response = MagicMock()
            mock_response.headers = {"content-type": "text/html"}

            with pytest.raises(ValueError, match="Unexpected content type"):
                downloader._validate_content_type(mock_response, ["text/csv"])


class TestCkanDownloader:
    """Tests for CkanDownloader."""

    def test_url_construction(self):
        """Test CKAN URL construction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = CkanDownloader(
                data_dir=Path(tmpdir),
                base_url="https://data.ca.gov",
                dataset_id="test-dataset",
            )
            assert downloader.base_url == "https://data.ca.gov"
            assert downloader.dataset_id == "test-dataset"

    def test_trailing_slash_removal(self):
        """Test trailing slash is removed from base URL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = CkanDownloader(
                data_dir=Path(tmpdir),
                base_url="https://data.ca.gov/",
                dataset_id="test",
            )
            assert downloader.base_url == "https://data.ca.gov"

    @patch("src.data_sources.base.requests.get")
    def test_fetch_uses_cache(self, mock_get):
        """Test fetch uses cached file when valid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = CkanDownloader(
                data_dir=Path(tmpdir),
                base_url="https://data.ca.gov",
                dataset_id="test-dataset",
                cache_days=30,
            )

            # Create cached file
            cached_file = Path(tmpdir) / "test-dataset.csv"
            cached_file.write_text("cached,data\n1,2")

            result = downloader.fetch(overwrite=False)
            assert result.status == DownloadResult.SKIPPED_CACHED
            mock_get.assert_not_called()


class TestPortalDownloader:
    """Tests for PortalDownloader."""

    def test_basic_creation(self):
        """Test portal downloader creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = PortalDownloader(
                data_dir=Path(tmpdir),
                url="https://example.com/data.csv",
                filename="data.csv",
                name="test_source",
            )
            assert downloader.url == "https://example.com/data.csv"
            assert downloader.filename == "data.csv"
            assert downloader.name == "test_source"

    def test_get_download_url(self):
        """Test get_download_url returns configured URL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = PortalDownloader(
                data_dir=Path(tmpdir),
                url="https://example.com/data.csv",
                filename="data.csv",
                name="test",
            )
            assert downloader.get_download_url() == "https://example.com/data.csv"

    def test_metadata_creation(self):
        """Test metadata generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = PortalDownloader(
                data_dir=Path(tmpdir),
                url="https://example.com/data.csv",
                filename="data.csv",
                name="test_source",
                description="Test description",
            )
            meta = downloader.get_metadata()
            assert meta.name == "test_source"
            assert meta.source_url == "https://example.com/data.csv"
            assert meta.description == "Test description"


class TestRegistryIntegration:
    """Tests for AdminDataSourceRegistry."""

    def test_registry_import(self):
        """Test registry can be imported."""
        from src.data_sources.registry import AdminDataSourceRegistry

        assert AdminDataSourceRegistry is not None

    @patch("src.data_sources.base.requests.get")
    def test_registry_continues_on_error(self, mock_get):
        """Test registry continues processing when one source fails."""
        from src.data_sources.registry import AdminDataSourceRegistry

        # Mock HTTP 403 error
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.raise_for_status.side_effect = Exception("Forbidden")
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = AdminDataSourceRegistry(Path(tmpdir), strict=False)

            # Should not raise even with all sources failing
            results = registry.fetch_all(overwrite=True)
            assert isinstance(results, dict)
