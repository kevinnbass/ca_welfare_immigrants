"""Tests for download utilities."""

import zipfile
from pathlib import Path

import pytest

from src.utils.download import extract_zip, download_with_retry


class TestExtractZip:
    """Tests for ZIP extraction with security checks."""

    def test_extract_normal_zip(self, tmp_zip_file, tmp_path):
        """Test extracting a normal ZIP file succeeds."""
        dest_dir = tmp_path / "extracted"
        extracted = extract_zip(tmp_zip_file, dest_dir)

        assert len(extracted) == 1
        assert (dest_dir / "test_file.txt").exists()

    def test_path_traversal_blocked(self, tmp_path):
        """Test that path traversal attempts are blocked."""
        # Create a malicious ZIP with path traversal
        zip_path = tmp_path / "malicious.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            # Attempt path traversal
            zf.writestr("../../../outside.txt", "malicious content")

        dest_dir = tmp_path / "safe_dir"
        dest_dir.mkdir()

        with pytest.raises(ValueError, match="Path traversal detected"):
            extract_zip(zip_path, dest_dir)

    def test_path_traversal_with_dots_blocked(self, tmp_path):
        """Test various path traversal patterns are blocked."""
        patterns = [
            "../secret.txt",
            "foo/../../secret.txt",
            "..\\..\\secret.txt",  # Windows-style
        ]

        for pattern in patterns:
            zip_path = tmp_path / f"test_{hash(pattern)}.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(pattern, "content")

            dest_dir = tmp_path / "dest"
            dest_dir.mkdir(exist_ok=True)

            with pytest.raises(ValueError, match="Path traversal detected"):
                extract_zip(zip_path, dest_dir)

    def test_nested_directory_allowed(self, tmp_path):
        """Test that nested directories within dest are allowed."""
        zip_path = tmp_path / "nested.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("subdir/nested/file.txt", "content")

        dest_dir = tmp_path / "dest"
        extracted = extract_zip(zip_path, dest_dir)

        assert len(extracted) > 0
        assert (dest_dir / "subdir" / "nested" / "file.txt").exists()


class TestDownloadWithRetry:
    """Tests for download retry logic."""

    def test_successful_download_no_retry(self, mocker, tmp_path):
        """Test successful download on first attempt."""
        mock_download = mocker.patch(
            "src.utils.download.download_file",
            return_value=tmp_path / "file.txt"
        )

        result = download_with_retry(
            "http://example.com/file.txt",
            tmp_path / "file.txt"
        )

        assert mock_download.call_count == 1
        assert result == tmp_path / "file.txt"

    def test_retry_on_timeout(self, mocker, tmp_path):
        """Test retry on timeout error."""
        import requests

        # Fail twice, succeed on third
        mock_download = mocker.patch(
            "src.utils.download.download_file",
            side_effect=[
                requests.Timeout("timeout"),
                requests.Timeout("timeout"),
                tmp_path / "file.txt"
            ]
        )

        result = download_with_retry(
            "http://example.com/file.txt",
            tmp_path / "file.txt",
            max_retries=3
        )

        assert mock_download.call_count == 3
        assert result == tmp_path / "file.txt"

    def test_exhaust_retries(self, mocker, tmp_path):
        """Test exception raised after exhausting retries."""
        import requests

        mocker.patch(
            "src.utils.download.download_file",
            side_effect=requests.ConnectionError("connection failed")
        )

        with pytest.raises(requests.ConnectionError):
            download_with_retry(
                "http://example.com/file.txt",
                tmp_path / "file.txt",
                max_retries=3
            )
