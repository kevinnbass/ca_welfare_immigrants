"""Tests for multiple imputation utilities."""

import numpy as np
import pandas as pd
import pytest

from src.utils.imputation import (
    create_bernoulli_imputations,
    combine_mi_results_rubins_rules,
    calibrate_to_total,
    calibrate_to_total_by_raking,
    MIResult,
)


class TestBernoulliImputations:
    """Tests for Bernoulli imputation creation."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        probs = np.array([0.3, 0.5, 0.7, 0.9])
        n_imputations = 10

        result = create_bernoulli_imputations(probs, n_imputations)

        assert result.shape == (4, 10)

    def test_binary_values(self):
        """Test that all values are 0 or 1."""
        probs = np.random.uniform(0, 1, 100)
        result = create_bernoulli_imputations(probs, n_imputations=5)

        assert set(result.flatten()) <= {0, 1}

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        probs = np.array([0.3, 0.5, 0.7])

        result1 = create_bernoulli_imputations(probs, n_imputations=5, random_state=42)
        result2 = create_bernoulli_imputations(probs, n_imputations=5, random_state=42)

        np.testing.assert_array_equal(result1, result2)

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        probs = np.array([0.3, 0.5, 0.7])

        result1 = create_bernoulli_imputations(probs, n_imputations=5, random_state=42)
        result2 = create_bernoulli_imputations(probs, n_imputations=5, random_state=123)

        # With different seeds, results should differ (almost certainly)
        assert not np.array_equal(result1, result2)

    def test_extreme_probabilities(self):
        """Test behavior with extreme probabilities."""
        probs = np.array([0.0, 1.0])
        result = create_bernoulli_imputations(probs, n_imputations=100)

        # Prob=0 should always produce 0
        assert np.all(result[0, :] == 0)

        # Prob=1 should always produce 1
        assert np.all(result[1, :] == 1)


class TestRubinsRules:
    """Tests for Rubin's rules combination."""

    def test_basic_combination(self):
        """Test basic Rubin's rules calculation."""
        estimates = [0.3, 0.32, 0.28, 0.31, 0.29]
        variances = [0.01, 0.01, 0.01, 0.01, 0.01]

        result = combine_mi_results_rubins_rules(estimates, variances)

        assert isinstance(result, MIResult)
        assert result.estimate == pytest.approx(0.30, abs=0.01)
        assert result.n_imputations == 5
        assert result.se > 0

    def test_single_imputation(self):
        """Test edge case with single imputation (m=1)."""
        estimates = [0.5]
        variances = [0.01]

        result = combine_mi_results_rubins_rules(estimates, variances)

        assert result.estimate == 0.5
        assert result.n_imputations == 1

    def test_zero_within_variance(self):
        """Test handling when within-imputation variance is zero."""
        estimates = [0.3, 0.5, 0.4]
        variances = [0.0, 0.0, 0.0]

        result = combine_mi_results_rubins_rules(estimates, variances)

        # Should still produce valid result using between-imputation variance
        assert not np.isnan(result.estimate)
        assert result.within_variance == 0.0
        assert result.between_variance > 0

    def test_zero_between_variance(self):
        """Test handling when between-imputation variance is zero."""
        estimates = [0.5, 0.5, 0.5]  # All same
        variances = [0.01, 0.01, 0.01]

        result = combine_mi_results_rubins_rules(estimates, variances)

        assert result.between_variance == pytest.approx(0.0)
        assert result.fraction_missing_info == pytest.approx(0.0)

    def test_confidence_interval_contains_estimate(self):
        """Test that CI contains point estimate."""
        estimates = [0.3, 0.35, 0.28]
        variances = [0.01, 0.01, 0.01]

        result = combine_mi_results_rubins_rules(estimates, variances)

        assert result.ci_lower <= result.estimate <= result.ci_upper

    def test_length_mismatch_raises_error(self):
        """Test that mismatched lengths raise error."""
        estimates = [0.3, 0.35]
        variances = [0.01, 0.01, 0.01]

        with pytest.raises(ValueError):
            combine_mi_results_rubins_rules(estimates, variances)

    def test_fraction_missing_info_bounds(self):
        """Test that FMI is in valid range [0, 1]."""
        for _ in range(50):
            n = np.random.randint(3, 20)
            estimates = list(np.random.uniform(0.2, 0.8, n))
            variances = list(np.random.uniform(0.001, 0.1, n))

            result = combine_mi_results_rubins_rules(estimates, variances)

            assert 0 <= result.fraction_missing_info <= 1


class TestProbabilityValidation:
    """Tests for probability validation in create_bernoulli_imputations."""

    def test_probabilities_out_of_bounds_raises(self):
        """Test that probabilities > 1 raise ValueError."""
        probs = np.array([0.5, 1.5, 0.3])
        with pytest.raises(ValueError, match="must be in"):
            create_bernoulli_imputations(probs, n_imputations=5)

    def test_negative_probabilities_raises(self):
        """Test that negative probabilities raise ValueError."""
        probs = np.array([0.5, -0.1, 0.3])
        with pytest.raises(ValueError, match="must be in"):
            create_bernoulli_imputations(probs, n_imputations=5)

    def test_nan_probabilities_raises(self):
        """Test that NaN probabilities raise ValueError."""
        probs = np.array([0.5, np.nan, 0.3])
        with pytest.raises(ValueError, match="contain NaN"):
            create_bernoulli_imputations(probs, n_imputations=5)

    def test_zero_imputations_raises(self):
        """Test that n_imputations < 1 raises ValueError."""
        probs = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="n_imputations must be"):
            create_bernoulli_imputations(probs, n_imputations=0)

    def test_negative_imputations_raises(self):
        """Test that negative n_imputations raises ValueError."""
        probs = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="n_imputations must be"):
            create_bernoulli_imputations(probs, n_imputations=-1)


class TestCalibrateToTotal:
    """Tests for calibrate_to_total function."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for calibration tests."""
        return pd.DataFrame({
            "indicator": [1, 1, 0, 0, 1],
            "weight": [100.0, 150.0, 200.0, 100.0, 50.0]
        })

    def test_successful_calibration(self, sample_df):
        """Test successful weight calibration."""
        # Current total: 100 + 150 + 50 = 300
        result = calibrate_to_total(
            sample_df, "indicator", "weight", target_total=600.0
        )
        # Should have calibrated weight column
        assert "weight_calibrated" in result.columns
        # New total should be ~600
        new_total = (result["indicator"] * result["weight_calibrated"]).sum()
        assert new_total == pytest.approx(600.0, rel=0.01)

    def test_negative_target_raises(self, sample_df):
        """Test that negative target raises ValueError."""
        with pytest.raises(ValueError, match="target_total must be positive"):
            calibrate_to_total(sample_df, "indicator", "weight", target_total=-100.0)

    def test_zero_target_raises(self, sample_df):
        """Test that zero target raises ValueError."""
        with pytest.raises(ValueError, match="target_total must be positive"):
            calibrate_to_total(sample_df, "indicator", "weight", target_total=0.0)

    def test_zero_current_total_raises(self):
        """Test that zero current total raises ValueError."""
        df = pd.DataFrame({
            "indicator": [0, 0, 0],
            "weight": [100.0, 100.0, 100.0]
        })
        with pytest.raises(ValueError, match="current_total is effectively zero"):
            calibrate_to_total(df, "indicator", "weight", target_total=100.0)

    def test_extreme_ratio_raises(self, sample_df):
        """Test that extreme calibration ratio raises ValueError."""
        # Current total is 300, target 60000 would give ratio of 200
        with pytest.raises(ValueError, match="Extreme calibration ratio"):
            calibrate_to_total(sample_df, "indicator", "weight", target_total=60000.0)

    def test_extreme_low_ratio_raises(self, sample_df):
        """Test that very low calibration ratio raises ValueError."""
        # Current total is 300, target 1 would give ratio of 0.003
        with pytest.raises(ValueError, match="Extreme calibration ratio"):
            calibrate_to_total(sample_df, "indicator", "weight", target_total=1.0)


class TestCalibrateByRaking:
    """Tests for calibrate_to_total_by_raking function."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for raking tests."""
        return pd.DataFrame({
            "indicator": [1, 1, 0, 0, 1],
            "weight": [100.0, 150.0, 200.0, 100.0, 50.0]
        })

    def test_successful_raking_convergence(self, sample_df):
        """Test successful raking convergence."""
        result = calibrate_to_total_by_raking(
            sample_df, "indicator", "weight", target_total=450.0
        )
        assert "weight_calibrated" in result.columns
        new_total = (result["indicator"] * result["weight_calibrated"]).sum()
        assert new_total == pytest.approx(450.0, rel=0.01)

    def test_negative_target_raises(self, sample_df):
        """Test that negative target raises ValueError."""
        with pytest.raises(ValueError, match="target_total must be positive"):
            calibrate_to_total_by_raking(
                sample_df, "indicator", "weight", target_total=-100.0
            )

    def test_zero_max_iterations_raises(self, sample_df):
        """Test that zero max_iterations raises ValueError."""
        with pytest.raises(ValueError, match="max_iterations must be"):
            calibrate_to_total_by_raking(
                sample_df, "indicator", "weight", target_total=450.0, max_iterations=0
            )

    def test_negative_tolerance_raises(self, sample_df):
        """Test that negative tolerance raises ValueError."""
        with pytest.raises(ValueError, match="tolerance must be positive"):
            calibrate_to_total_by_raking(
                sample_df, "indicator", "weight", target_total=450.0, tolerance=-0.01
            )

    def test_non_convergence_raises(self):
        """Test that non-convergence raises ValueError."""
        # Create a case that won't converge in 1 iteration with tight tolerance
        df = pd.DataFrame({
            "indicator": [1, 1, 0],
            "weight": [100.0, 100.0, 100.0]
        })
        with pytest.raises(ValueError, match="failed to converge"):
            calibrate_to_total_by_raking(
                df, "indicator", "weight",
                target_total=1000.0,  # Needs ratio of 5
                max_iterations=1,
                tolerance=0.0001
            )
