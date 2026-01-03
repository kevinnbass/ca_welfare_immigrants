"""
Tests for analysis_units module.

Tests household vs person level estimation utilities.
"""

import numpy as np
import pandas as pd
import pytest

from src.utils.analysis_units import (
    EstimandSpec,
    get_estimand_spec,
    prepare_household_level_data,
    validate_weight_columns,
    get_all_estimand_specs,
    classify_household_status,
    STATUS_RISK_ORDER,
)
from src import config


class TestEstimandSpec:
    """Tests for EstimandSpec dataclass and get_estimand_spec function."""

    def test_get_medicaid_spec(self):
        """Test getting Medicaid estimand spec."""
        spec = get_estimand_spec("medicaid")

        assert spec.unit == "person"
        assert spec.weight_col == "PWGTP"
        assert spec.weight_prefix == "PWGTP"
        assert spec.estimand_id == "medicaid_person_rate"

    def test_get_snap_household_householder_spec(self):
        """Test getting SNAP household (householder) estimand spec."""
        spec = get_estimand_spec("snap_household_householder")

        assert spec.unit == "household"
        assert spec.weight_col == "WGTP"
        assert spec.weight_prefix == "WGTP"
        assert spec.hh_status_rule == "householder"
        assert spec.estimand_id == "snap_household_rate_householder"

    def test_get_snap_household_highest_risk_spec(self):
        """Test getting SNAP household (highest risk) estimand spec."""
        spec = get_estimand_spec("snap_household_highest_risk")

        assert spec.unit == "household"
        assert spec.weight_col == "WGTP"
        assert spec.weight_prefix == "WGTP"
        assert spec.hh_status_rule == "highest_risk"
        assert spec.estimand_id == "snap_household_rate_highest_risk"

    def test_get_snap_person_spec(self):
        """Test getting SNAP person estimand spec."""
        spec = get_estimand_spec("snap_person")

        assert spec.unit == "person"
        assert spec.weight_col == "PWGTP"
        assert spec.weight_prefix == "PWGTP"

    def test_unknown_program_raises(self):
        """Test that unknown program key raises KeyError."""
        with pytest.raises(KeyError):
            get_estimand_spec("unknown_program")

    def test_deprecated_snap_warns(self):
        """Test that deprecated 'snap' key emits warning."""
        with pytest.warns(DeprecationWarning):
            spec = get_estimand_spec("snap")

        assert spec.deprecated is True


class TestPrepareHouseholdLevelData:
    """Tests for prepare_household_level_data function."""

    @pytest.fixture
    def sample_person_df(self):
        """Create sample person-level DataFrame."""
        return pd.DataFrame({
            "SERIALNO": [1, 1, 1, 2, 2, 3],
            "SPORDER": [1, 2, 3, 1, 2, 1],  # 1 = householder
            "status": [
                "US_BORN", "ILLEGAL", "US_BORN",
                "NATURALIZED", "ILLEGAL",
                "LEGAL_NONCITIZEN"
            ],
            "PWGTP": [100, 80, 50, 120, 90, 150],
            "WGTP": [230, 230, 230, 210, 210, 150],
            "snap": [1, 1, 1, 0, 0, 1],
        })

    def test_householder_rule(self, sample_person_df):
        """Test filtering to householders only."""
        hh_df = prepare_household_level_data(
            sample_person_df,
            status_col="status",
            hh_status_rule="householder",
        )

        # Should have 3 households (one row per household)
        assert len(hh_df) == 3
        # All rows should have SPORDER == 1
        assert (hh_df["SPORDER"] == 1).all()
        # Check household statuses
        assert list(hh_df["status"]) == ["US_BORN", "NATURALIZED", "LEGAL_NONCITIZEN"]

    def test_highest_risk_rule(self, sample_person_df):
        """Test selecting highest-risk member per household."""
        hh_df = prepare_household_level_data(
            sample_person_df,
            status_col="status",
            hh_status_rule="highest_risk",
        )

        # Should have 3 households
        assert len(hh_df) == 3

        # HH 1: ILLEGAL should be selected (highest risk)
        hh1 = hh_df[hh_df["SERIALNO"] == 1]
        assert hh1["status"].iloc[0] == "ILLEGAL"

        # HH 2: ILLEGAL should be selected
        hh2 = hh_df[hh_df["SERIALNO"] == 2]
        assert hh2["status"].iloc[0] == "ILLEGAL"

        # HH 3: LEGAL_NONCITIZEN (only member)
        hh3 = hh_df[hh_df["SERIALNO"] == 3]
        assert hh3["status"].iloc[0] == "LEGAL_NONCITIZEN"

    def test_invalid_rule_raises(self, sample_person_df):
        """Test that invalid rule raises ValueError."""
        with pytest.raises(ValueError, match="Unknown hh_status_rule"):
            prepare_household_level_data(
                sample_person_df,
                status_col="status",
                hh_status_rule="invalid_rule",
            )


class TestValidateWeightColumns:
    """Tests for validate_weight_columns function."""

    @pytest.fixture
    def sample_df_with_weights(self):
        """Create DataFrame with full replicate weights."""
        df = pd.DataFrame({
            "PWGTP": [100],
            "WGTP": [200],
        })
        # Add 80 person replicate weights
        for i in range(1, 81):
            df[f"PWGTP{i}"] = 100 + i
            df[f"WGTP{i}"] = 200 + i
        return df

    def test_person_weights_valid(self, sample_df_with_weights):
        """Test validation passes for person-level estimand with PWGTP weights."""
        spec = EstimandSpec(
            estimand_id="test",
            program_key="test",
            program_variable="test_var",
            unit="person",
            weight_col="PWGTP",
            weight_prefix="PWGTP",
            indicator_col="test",
        )

        # Should not raise
        validate_weight_columns(sample_df_with_weights, spec)

    def test_household_weights_valid(self, sample_df_with_weights):
        """Test validation passes for household-level estimand with WGTP weights."""
        spec = EstimandSpec(
            estimand_id="test",
            program_key="test",
            program_variable="test_var",
            unit="household",
            weight_col="WGTP",
            weight_prefix="WGTP",
            indicator_col="test",
            hh_status_rule="householder",
        )

        # Should not raise
        validate_weight_columns(sample_df_with_weights, spec)

    def test_missing_weights_raises(self):
        """Test that missing weight columns raises KeyError."""
        df = pd.DataFrame({"PWGTP": [100]})  # Missing replicate weights

        spec = EstimandSpec(
            estimand_id="test",
            program_key="test",
            program_variable="test_var",
            unit="person",
            weight_col="PWGTP",
            weight_prefix="PWGTP",
            indicator_col="test",
        )

        with pytest.raises(KeyError, match="Missing weight columns"):
            validate_weight_columns(df, spec)


class TestGetAllEstimandSpecs:
    """Tests for get_all_estimand_specs function."""

    def test_returns_non_deprecated(self):
        """Test that only non-deprecated estimands are returned."""
        specs = get_all_estimand_specs()

        # Should not include deprecated 'snap' key
        assert "snap_legacy" not in specs

        # Should include non-deprecated estimands
        assert "medicaid_person_rate" in specs
        assert "snap_household_rate_householder" in specs
        assert "snap_household_rate_highest_risk" in specs


class TestClassifyHouseholdStatus:
    """Tests for classify_household_status function."""

    @pytest.fixture
    def sample_person_df(self):
        """Create sample person-level DataFrame."""
        return pd.DataFrame({
            "SERIALNO": [1, 1, 2, 2, 3],
            "SPORDER": [1, 2, 1, 2, 1],
            "status": [
                "US_BORN", "ILLEGAL",
                "NATURALIZED", "LEGAL_NONCITIZEN",
                "ILLEGAL"
            ],
        })

    def test_both_methods(self, sample_person_df):
        """Test adding both classification columns."""
        result = classify_household_status(
            sample_person_df,
            status_col="status",
            method="both",
        )

        assert "hh_status_householder" in result.columns
        assert "hh_status_highest_risk" in result.columns

        # Check HH 1: householder=US_BORN, highest_risk=ILLEGAL
        hh1 = result[result["SERIALNO"] == 1]
        assert (hh1["hh_status_householder"] == "US_BORN").all()
        assert (hh1["hh_status_highest_risk"] == "ILLEGAL").all()

    def test_householder_only(self, sample_person_df):
        """Test adding only householder classification."""
        result = classify_household_status(
            sample_person_df,
            status_col="status",
            method="householder",
        )

        assert "hh_status_householder" in result.columns
        assert "hh_status_highest_risk" not in result.columns


class TestStatusRiskOrder:
    """Tests for STATUS_RISK_ORDER constant."""

    def test_illegal_highest_risk(self):
        """Test that ILLEGAL has lowest (highest risk) value."""
        assert STATUS_RISK_ORDER["ILLEGAL"] < STATUS_RISK_ORDER["US_BORN"]
        assert STATUS_RISK_ORDER["ILLEGAL"] < STATUS_RISK_ORDER["NATURALIZED"]

    def test_us_born_lowest_risk(self):
        """Test that US_BORN has highest (lowest risk) value among common statuses."""
        assert STATUS_RISK_ORDER["US_BORN"] > STATUS_RISK_ORDER["ILLEGAL"]
        assert STATUS_RISK_ORDER["US_BORN"] > STATUS_RISK_ORDER["LEGAL_NONCITIZEN"]


class TestSNAPWeightSelection:
    """Integration tests verifying SNAP uses correct weights."""

    def test_snap_household_uses_wgtp(self):
        """Verify SNAP household estimand uses WGTP weight column."""
        spec = get_estimand_spec("snap_household_householder")

        assert spec.weight_col == "WGTP", "SNAP household should use WGTP"
        assert spec.weight_prefix == "WGTP", "SNAP household replicates should be WGTP"

    def test_snap_person_uses_pwgtp(self):
        """Verify SNAP person estimand uses PWGTP weight column."""
        spec = get_estimand_spec("snap_person")

        assert spec.weight_col == "PWGTP", "SNAP person should use PWGTP"
        assert spec.weight_prefix == "PWGTP", "SNAP person replicates should be PWGTP"

    def test_medicaid_uses_pwgtp(self):
        """Verify Medicaid (person-level) uses PWGTP weights."""
        spec = get_estimand_spec("medicaid")

        assert spec.weight_col == "PWGTP", "Medicaid should use PWGTP"
        assert spec.unit == "person"
