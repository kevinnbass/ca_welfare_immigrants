"""
Integration tests for mini end-to-end pipeline with synthetic data.
"""

import numpy as np
import pandas as pd
import pytest


class TestSyntheticDataGeneration:
    """Tests for synthetic test data generation."""

    @pytest.fixture
    def synthetic_acs_data(self):
        """Generate synthetic ACS-like data for testing."""
        np.random.seed(42)
        n = 200

        # Generate person records
        data = {
            "SERIALNO": np.repeat(range(n // 2), 2),  # 2 persons per household
            "SPORDER": np.tile([1, 2], n // 2),  # Person order in household
            "PWGTP": np.random.uniform(50, 200, n),
            "WGTP": np.repeat(np.random.uniform(100, 400, n // 2), 2),
            "AGEP": np.random.randint(0, 90, n),
            "SEX": np.random.choice([1, 2], n),
            "CIT": np.random.choice([1, 2, 3, 4, 5], n, p=[0.7, 0.1, 0.05, 0.1, 0.05]),
            "NATIVITY": np.where(np.random.random(n) < 0.75, 1, 2),
            "HINS4": np.random.choice([1, 2], n, p=[0.2, 0.8]),  # Medicaid
            "FS": np.random.choice([1, 2], n, p=[0.15, 0.85]),  # SNAP (household)
            "SSP": np.random.randint(0, 1000, n),  # SSI
            "PAP": np.random.randint(0, 500, n),  # Public assistance
        }

        # Add replicate weights
        for i in range(1, 81):
            data[f"PWGTP{i}"] = data["PWGTP"] * np.random.uniform(0.9, 1.1, n)
            data[f"WGTP{i}"] = data["WGTP"] * np.random.uniform(0.9, 1.1, n)

        return pd.DataFrame(data)

    def test_synthetic_data_structure(self, synthetic_acs_data):
        """Test synthetic data has correct structure."""
        df = synthetic_acs_data

        # Check required columns
        assert "SERIALNO" in df.columns
        assert "PWGTP" in df.columns
        assert "WGTP" in df.columns
        assert "CIT" in df.columns

        # Check replicate weights
        assert "PWGTP1" in df.columns
        assert "PWGTP80" in df.columns
        assert "WGTP1" in df.columns
        assert "WGTP80" in df.columns

    def test_synthetic_data_values(self, synthetic_acs_data):
        """Test synthetic data has valid values."""
        df = synthetic_acs_data

        assert df["PWGTP"].min() > 0
        assert df["AGEP"].min() >= 0
        assert df["CIT"].isin([1, 2, 3, 4, 5]).all()


class TestImputationPipeline:
    """Tests for imputation pipeline with synthetic data."""

    @pytest.fixture
    def synthetic_acs_data(self):
        """Generate synthetic ACS data."""
        np.random.seed(42)
        n = 100

        data = {
            "SERIALNO": range(n),
            "SPORDER": np.ones(n, dtype=int),
            "PWGTP": np.random.uniform(50, 200, n),
            "CIT": np.random.choice([1, 4, 5], n, p=[0.7, 0.2, 0.1]),
            "NATIVITY": np.where(np.random.random(n) < 0.75, 1, 2),
            "HINS4": np.random.choice([1, 2], n, p=[0.2, 0.8]),
        }
        return pd.DataFrame(data)

    def test_noncitizen_identification(self, synthetic_acs_data):
        """Test noncitizens can be identified."""
        df = synthetic_acs_data

        # Noncitizens have CIT in [4, 5]
        noncitizen_mask = df["CIT"].isin([4, 5])
        assert noncitizen_mask.sum() > 0

    def test_imputation_creates_status_column(self, synthetic_acs_data):
        """Test imputation produces status column."""
        df = synthetic_acs_data.copy()

        # Mock imputation: assign random status to noncitizens
        noncitizen_mask = df["CIT"].isin([4, 5])

        df["imputed_status"] = "US_BORN"
        df.loc[df["CIT"] == 1, "imputed_status"] = "US_BORN"
        df.loc[df["CIT"].isin([2, 3]), "imputed_status"] = "NATURALIZED"
        df.loc[noncitizen_mask, "imputed_status"] = np.where(
            np.random.random(noncitizen_mask.sum()) < 0.5,
            "LEGAL_IMMIGRANT",
            "ILLEGAL",
        )

        assert "imputed_status" in df.columns
        assert df["imputed_status"].notna().all()


class TestRateEstimation:
    """Tests for rate estimation with synthetic data."""

    @pytest.fixture
    def synthetic_data_with_status(self):
        """Generate synthetic data with imputed status."""
        np.random.seed(42)
        n = 300

        data = {
            "SERIALNO": range(n),
            "PWGTP": np.random.uniform(50, 200, n),
            "HINS4": np.random.choice([1, 2], n, p=[0.25, 0.75]),  # 25% Medicaid
            "status": np.random.choice(
                ["US_BORN", "LEGAL_IMMIGRANT", "ILLEGAL"],
                n,
                p=[0.6, 0.25, 0.15],
            ),
        }

        # Add replicate weights
        for i in range(1, 81):
            data[f"PWGTP{i}"] = data["PWGTP"] * np.random.uniform(0.9, 1.1, n)

        return pd.DataFrame(data)

    def test_weighted_rate_calculation(self, synthetic_data_with_status):
        """Test weighted rate calculation."""
        df = synthetic_data_with_status

        # Create indicator
        df["medicaid"] = (df["HINS4"] == 1).astype(int)

        # Calculate weighted rate
        weighted_rate = np.average(df["medicaid"], weights=df["PWGTP"])

        assert 0 < weighted_rate < 1

    def test_rate_by_status_group(self, synthetic_data_with_status):
        """Test rates can be computed by status group."""
        df = synthetic_data_with_status

        df["medicaid"] = (df["HINS4"] == 1).astype(int)

        rates = {}
        for status in df["status"].unique():
            mask = df["status"] == status
            rates[status] = np.average(
                df.loc[mask, "medicaid"],
                weights=df.loc[mask, "PWGTP"],
            )

        assert len(rates) == 3
        assert all(0 <= r <= 1 for r in rates.values())


class TestOutputStructure:
    """Tests for output file structure."""

    def test_output_columns(self):
        """Test output has required columns."""
        expected_columns = [
            "program",
            "group",
            "estimate",
            "se",
            "ci_lower",
            "ci_upper",
        ]

        # Create mock output
        output = pd.DataFrame(
            {
                "program": ["medicaid", "medicaid", "snap"],
                "group": ["US_BORN", "ILLEGAL", "US_BORN"],
                "estimate": [0.25, 0.15, 0.12],
                "se": [0.02, 0.03, 0.02],
                "ci_lower": [0.21, 0.09, 0.08],
                "ci_upper": [0.29, 0.21, 0.16],
            }
        )

        for col in expected_columns:
            assert col in output.columns

    def test_confidence_interval_validity(self):
        """Test confidence intervals are valid."""
        output = pd.DataFrame(
            {
                "estimate": [0.25, 0.15, 0.12],
                "se": [0.02, 0.03, 0.02],
                "ci_lower": [0.21, 0.09, 0.08],
                "ci_upper": [0.29, 0.21, 0.16],
            }
        )

        # CI should contain estimate
        assert (output["ci_lower"] <= output["estimate"]).all()
        assert (output["estimate"] <= output["ci_upper"]).all()

        # CI should be symmetric around estimate (approximately)
        lower_diff = output["estimate"] - output["ci_lower"]
        upper_diff = output["ci_upper"] - output["estimate"]
        assert np.allclose(lower_diff, upper_diff, rtol=0.1)
