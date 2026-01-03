"""Tests for configuration module."""

import pandas as pd
import pytest

from src import config


class TestValidateYear:
    """Tests for year validation."""

    def test_valid_year_in_range(self):
        """Test that valid years pass validation."""
        config.validate_year(2020)
        config.validate_year(2023)
        config.validate_year(2024)

    def test_year_at_lower_bound(self):
        """Test year at lower bound."""
        config.validate_year(config.VALID_YEAR_RANGE[0])

    def test_year_at_upper_bound(self):
        """Test year at upper bound."""
        config.validate_year(config.VALID_YEAR_RANGE[1])

    def test_year_below_range_raises(self):
        """Test that year below range raises ValueError."""
        with pytest.raises(ValueError, match="outside valid range"):
            config.validate_year(2010)

    def test_year_above_range_raises(self):
        """Test that year above range raises ValueError."""
        with pytest.raises(ValueError, match="outside valid range"):
            config.validate_year(2050)


class TestValidateRequiredColumns:
    """Tests for required column validation."""

    def test_all_columns_present(self):
        """Test passes when all columns exist."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4], "col3": [5, 6]})
        # Should not raise
        config.validate_required_columns(df, ["col1", "col2"])

    def test_single_missing_column_raises(self):
        """Test raises when single column is missing."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        with pytest.raises(KeyError, match="Missing required columns"):
            config.validate_required_columns(df, ["col1", "col3"])

    def test_multiple_missing_columns_raises(self):
        """Test raises when multiple columns are missing."""
        df = pd.DataFrame({"col1": [1, 2]})
        with pytest.raises(KeyError) as exc_info:
            config.validate_required_columns(df, ["col1", "col2", "col3"])
        assert "col2" in str(exc_info.value)
        assert "col3" in str(exc_info.value)

    def test_context_included_in_error(self):
        """Test that context is included in error message."""
        df = pd.DataFrame({"col1": [1, 2]})
        with pytest.raises(KeyError, match="in test_context"):
            config.validate_required_columns(df, ["col1", "col2"], context="test_context")

    def test_empty_required_columns_passes(self):
        """Test passes when no columns required."""
        df = pd.DataFrame({"col1": [1, 2]})
        config.validate_required_columns(df, [])


class TestStatusGroupEnum:
    """Tests for StatusGroup enum."""

    def test_all_values_accessible(self):
        """Test that all status values are accessible."""
        assert config.StatusGroup.US_BORN.value == "US_BORN"
        assert config.StatusGroup.NATURALIZED.value == "NATURALIZED"
        assert config.StatusGroup.NONCITIZEN.value == "NONCITIZEN"
        assert config.StatusGroup.LEGAL_IMMIGRANT.value == "LEGAL_IMMIGRANT"
        assert config.StatusGroup.LEGAL_NONCITIZEN.value == "LEGAL_NONCITIZEN"
        assert config.StatusGroup.ILLEGAL.value == "ILLEGAL"
        assert config.StatusGroup.UNKNOWN.value == "UNKNOWN"

    def test_string_conversion(self):
        """Test that enum values can be used as strings."""
        status = config.StatusGroup.US_BORN
        assert str(status.value) == "US_BORN"
        assert f"{status.value}" == "US_BORN"

    def test_enum_comparison(self):
        """Test enum comparison."""
        assert config.StatusGroup.US_BORN == config.StatusGroup.US_BORN
        assert config.StatusGroup.US_BORN != config.StatusGroup.NATURALIZED

    def test_enum_from_string(self):
        """Test creating enum from string value."""
        status = config.StatusGroup("US_BORN")
        assert status == config.StatusGroup.US_BORN


class TestEnsureDirectories:
    """Tests for ensure_directories function."""

    def test_creates_missing_dirs(self, tmp_path, monkeypatch):
        """Test that missing directories are created."""
        # Create a temporary project structure
        test_data_dir = tmp_path / "data"
        test_raw_dir = test_data_dir / "raw"

        # Temporarily override the directory constants
        monkeypatch.setattr(config, "_REQUIRED_DIRS", [test_raw_dir])

        # Directory shouldn't exist yet
        assert not test_raw_dir.exists()

        # Call ensure_directories
        config.ensure_directories()

        # Now it should exist
        assert test_raw_dir.exists()

    def test_existing_dirs_no_error(self, tmp_path, monkeypatch):
        """Test that existing directories don't cause errors."""
        # Create the directory first
        test_dir = tmp_path / "existing"
        test_dir.mkdir(parents=True)

        monkeypatch.setattr(config, "_REQUIRED_DIRS", [test_dir])

        # Should not raise
        config.ensure_directories()

        # Should still exist
        assert test_dir.exists()


class TestConfigConstants:
    """Tests for configuration constants."""

    def test_age_bins_and_labels_match(self):
        """Test that AGE_BINS and AGE_LABELS have matching lengths."""
        # AGE_BINS has N+1 edges for N labels
        assert len(config.AGE_BINS) == len(config.AGE_LABELS) + 1

    def test_valid_year_range_is_tuple(self):
        """Test that VALID_YEAR_RANGE is a proper tuple."""
        assert isinstance(config.VALID_YEAR_RANGE, tuple)
        assert len(config.VALID_YEAR_RANGE) == 2
        assert config.VALID_YEAR_RANGE[0] < config.VALID_YEAR_RANGE[1]

    def test_welfare_programs_structure(self):
        """Test WELFARE_PROGRAMS dictionary structure."""
        for program, details in config.WELFARE_PROGRAMS.items():
            assert "variable" in details
            assert "condition" in details
            assert "level" in details
            assert details["level"] in ["person", "household"]

    def test_n_replicate_weights_is_80(self):
        """Test that N_REPLICATE_WEIGHTS is 80."""
        assert config.N_REPLICATE_WEIGHTS == 80

    def test_statistical_thresholds(self):
        """Test statistical threshold values are reasonable."""
        assert config.MIN_UNWEIGHTED_N > 0
        assert 0 < config.MAX_COEFFICIENT_OF_VARIATION < 1
        assert 0 < config.MIN_MODEL_AUC < 1
