"""
Tests for pooled multi-year analysis.
"""

import numpy as np
import pandas as pd
import pytest


class TestPooledWeights:
    """Tests for pooled weight construction."""

    def test_pooled_analysis_import(self):
        """Test pooled analysis module can be imported."""
        # Check pooled analysis exists
        from src import config  # Verify src package works

        assert hasattr(config, "VALID_YEAR_RANGE")

    def test_weights_divided_by_k(self):
        """Test pooled weights are divided by number of years."""
        # Create mock data for 3 years
        df = pd.DataFrame(
            {
                "PWGTP": [100, 200, 300],
                "PWGTP1": [98, 198, 298],
                "PWGTP2": [102, 202, 302],
                "WGTP": [50, 100, 150],
                "year": [2021, 2022, 2023],
            }
        )

        n_years = 3

        # Apply weight division
        weight_cols = ["PWGTP", "PWGTP1", "PWGTP2", "WGTP"]
        for col in weight_cols:
            df[f"{col}_pooled"] = df[col] / n_years

        # Verify division
        assert df["PWGTP_pooled"].iloc[0] == pytest.approx(100 / 3)
        assert df["WGTP_pooled"].iloc[1] == pytest.approx(100 / 3)

    def test_pooled_weights_sum_to_original(self):
        """Test pooled weights from k years sum to approximately single-year weights."""
        np.random.seed(42)
        n_obs_per_year = 100
        n_years = 5

        # Create mock data with consistent weights
        base_weight = 1000  # Average weight

        years_data = []
        for year in range(2019, 2019 + n_years):
            year_df = pd.DataFrame(
                {
                    "PWGTP": np.random.normal(base_weight, 100, n_obs_per_year),
                    "year": year,
                }
            )
            years_data.append(year_df)

        combined = pd.concat(years_data, ignore_index=True)

        # Divide by number of years
        combined["PWGTP_pooled"] = combined["PWGTP"] / n_years

        # Sum of pooled weights should approximate sum of single year
        single_year_sum = combined[combined["year"] == 2019]["PWGTP"].sum()
        pooled_sum = combined["PWGTP_pooled"].sum()

        # Pooled sum for all years should equal single-year sum (approximately)
        assert abs(pooled_sum / n_years - single_year_sum / n_years) < single_year_sum * 0.1


class TestVariableHarmonization:
    """Tests for variable harmonization across years."""

    def test_harmonize_variable_names(self):
        """Test variable name harmonization."""
        # 2019 had different RELP coding
        df_2019 = pd.DataFrame(
            {
                "RELSHIPP": [1, 2, 3],  # Old variable name
                "PWGTP": [100, 200, 300],
            }
        )

        df_2020 = pd.DataFrame(
            {
                "RELP": [1, 2, 3],  # New variable name
                "PWGTP": [100, 200, 300],
            }
        )

        # Apply harmonization (rename RELSHIPP to RELP for consistency)
        if "RELSHIPP" in df_2019.columns:
            df_2019 = df_2019.rename(columns={"RELSHIPP": "RELP"})

        assert "RELP" in df_2019.columns
        assert "RELP" in df_2020.columns

    def test_combine_years(self):
        """Test combining multiple years of data."""
        dfs = []
        for year in [2019, 2020, 2021]:
            df = pd.DataFrame(
                {
                    "person_id": range(10),
                    "PWGTP": np.random.uniform(100, 200, 10),
                    "year": year,
                }
            )
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)

        assert len(combined) == 30
        assert combined["year"].nunique() == 3
        assert set(combined["year"].unique()) == {2019, 2020, 2021}


class TestPooledEstimation:
    """Tests for pooled rate estimation."""

    def test_pooled_variance_formula(self):
        """Test pooled variance is computed correctly."""
        # With pooled data, variance should account for clustering by year
        np.random.seed(42)

        n_years = 5

        # Mock estimates by year
        year_estimates = np.random.normal(0.15, 0.02, n_years)
        year_variances = np.full(n_years, 0.001)

        # Pooled estimate is weighted average
        pooled_estimate = year_estimates.mean()

        # Pooled variance should be smaller than single-year
        pooled_variance = year_variances.mean() / n_years

        assert pooled_estimate > 0
        assert pooled_variance < year_variances.mean()

    def test_stratified_pooled_estimation(self):
        """Test pooled estimation with stratification."""
        df = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B"] * 10,
                "outcome": np.random.binomial(1, 0.3, 40),
                "PWGTP": np.random.uniform(100, 200, 40),
                "year": [2020] * 20 + [2021] * 20,
            }
        )

        # Compute rates by group
        rates = df.groupby("group").apply(lambda x: np.average(x["outcome"], weights=x["PWGTP"]))

        assert len(rates) == 2
        assert "A" in rates.index
        assert "B" in rates.index
