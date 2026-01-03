"""Tests for survey weight calculation utilities."""

import numpy as np
import pandas as pd
import pytest

from src.utils.weights import (
    ACSReplicateWeightVariance,
    coefficient_of_variation,
    confidence_interval,
    margin_of_error,
    weighted_mean,
    weighted_proportion,
)


class TestWeightedMean:
    """Tests for weighted_mean function."""

    def test_basic_weighted_mean(self):
        """Test basic weighted mean calculation."""
        values = np.array([1, 2, 3])
        weights = np.array([1, 1, 1])

        result = weighted_mean(values, weights)
        assert result == pytest.approx(2.0)

    def test_weighted_mean_with_weights(self):
        """Test weighted mean with unequal weights."""
        values = np.array([1, 2])
        weights = np.array([3, 1])  # Weight 1 more heavily

        result = weighted_mean(values, weights)
        assert result == pytest.approx(1.25)  # (1*3 + 2*1) / 4

    def test_empty_array_returns_nan(self):
        """Test that empty arrays return NaN."""
        values = np.array([])
        weights = np.array([])

        result = weighted_mean(values, weights)
        assert np.isnan(result)

    def test_zero_weights_returns_nan(self):
        """Test that all-zero weights return NaN."""
        values = np.array([1, 2, 3])
        weights = np.array([0, 0, 0])

        result = weighted_mean(values, weights)
        assert np.isnan(result)

    def test_handles_nan_values(self):
        """Test that NaN values are properly excluded."""
        values = np.array([1, np.nan, 3])
        weights = np.array([1, 1, 1])

        result = weighted_mean(values, weights)
        assert result == pytest.approx(2.0)

    def test_handles_nan_weights(self):
        """Test that NaN weights are properly excluded."""
        values = np.array([1, 2, 3])
        weights = np.array([1, np.nan, 1])

        result = weighted_mean(values, weights)
        assert result == pytest.approx(2.0)


class TestWeightedProportion:
    """Tests for weighted_proportion function."""

    def test_basic_proportion(self):
        """Test basic proportion calculation."""
        indicator = np.array([1, 1, 0, 0])
        weights = np.array([1, 1, 1, 1])

        result = weighted_proportion(indicator, weights)
        assert result == pytest.approx(0.5)

    def test_proportion_bounds(self):
        """Test that proportion is always in [0, 1]."""
        for _ in range(100):
            indicator = np.random.choice([0, 1], 50)
            weights = np.random.uniform(0.1, 10, 50)

            result = weighted_proportion(indicator, weights)
            assert 0 <= result <= 1

    def test_all_ones(self):
        """Test proportion when all values are 1."""
        indicator = np.array([1, 1, 1, 1])
        weights = np.array([1, 2, 3, 4])

        result = weighted_proportion(indicator, weights)
        assert result == pytest.approx(1.0)

    def test_all_zeros(self):
        """Test proportion when all values are 0."""
        indicator = np.array([0, 0, 0, 0])
        weights = np.array([1, 2, 3, 4])

        result = weighted_proportion(indicator, weights)
        assert result == pytest.approx(0.0)


class TestCoefficientOfVariation:
    """Tests for coefficient_of_variation function."""

    def test_basic_cv(self):
        """Test basic CV calculation."""
        cv = coefficient_of_variation(estimate=0.5, se=0.1)
        assert cv == pytest.approx(0.2)

    def test_zero_estimate_returns_inf(self):
        """Test that CV is infinity when estimate is zero."""
        cv = coefficient_of_variation(estimate=0.0, se=0.1)
        assert cv == np.inf

    def test_cv_is_positive(self):
        """Test that CV is always positive (uses absolute value)."""
        cv = coefficient_of_variation(estimate=-0.5, se=0.1)
        assert cv == pytest.approx(0.2)
        assert cv >= 0


class TestConfidenceInterval:
    """Tests for confidence_interval function."""

    def test_basic_ci(self):
        """Test basic 95% CI calculation."""
        lower, upper = confidence_interval(estimate=0.5, se=0.1)

        # 95% CI should be roughly +/- 1.96 SE
        assert lower == pytest.approx(0.5 - 1.96 * 0.1, rel=0.01)
        assert upper == pytest.approx(0.5 + 1.96 * 0.1, rel=0.01)

    def test_ci_contains_estimate(self):
        """Test that CI always contains the point estimate."""
        for _ in range(100):
            est = np.random.uniform(-10, 10)
            se = np.random.uniform(0.1, 5)

            lower, upper = confidence_interval(est, se)
            assert lower <= est <= upper

    def test_narrower_ci_with_higher_confidence(self):
        """Test that 90% CI is narrower than 95% CI."""
        lower_90, upper_90 = confidence_interval(0.5, 0.1, confidence=0.90)
        lower_95, upper_95 = confidence_interval(0.5, 0.1, confidence=0.95)

        assert (upper_90 - lower_90) < (upper_95 - lower_95)


class TestBinaryIndicatorValidation:
    """Tests for binary indicator validation in weighted_proportion."""

    def test_valid_binary_indicator(self):
        """Test that valid binary indicators pass validation."""
        indicator = np.array([0, 1, 0, 1, 1])
        weights = np.array([1, 1, 1, 1, 1])
        # Should not raise
        result = weighted_proportion(indicator, weights, validate_binary=True)
        assert 0 <= result <= 1

    def test_non_binary_indicator_raises(self):
        """Test that non-binary values raise ValueError."""
        indicator = np.array([0, 1, 2, 0, 1])
        weights = np.array([1, 1, 1, 1, 1])
        with pytest.raises(ValueError, match="must be binary"):
            weighted_proportion(indicator, weights, validate_binary=True)

    def test_float_indicator_raises(self):
        """Test that float values other than 0/1 raise ValueError."""
        indicator = np.array([0.0, 0.5, 1.0])
        weights = np.array([1, 1, 1])
        with pytest.raises(ValueError, match="must be binary"):
            weighted_proportion(indicator, weights, validate_binary=True)

    def test_validation_disabled(self):
        """Test that validation can be disabled."""
        indicator = np.array([0, 1, 2])  # Invalid for binary
        weights = np.array([1, 1, 1])
        # Should not raise when validation disabled
        result = weighted_proportion(indicator, weights, validate_binary=False)
        assert result == pytest.approx(1.0)  # (0+1+2)/3

    def test_nan_values_ignored_in_validation(self):
        """Test that NaN values are ignored during binary validation."""
        indicator = np.array([0, 1, np.nan, 0, 1])
        weights = np.array([1, 1, 1, 1, 1])
        # Should not raise (NaN values are filtered)
        result = weighted_proportion(indicator, weights, validate_binary=True)
        assert 0 <= result <= 1


class TestMarginOfError:
    """Tests for margin_of_error function."""

    def test_basic_moe(self):
        """Test basic margin of error calculation."""
        moe = margin_of_error(se=0.1, confidence=0.90)
        # 90% CI z-value is ~1.645
        assert moe == pytest.approx(1.645 * 0.1, rel=0.01)

    def test_moe_95_confidence(self):
        """Test MOE with 95% confidence."""
        moe = margin_of_error(se=0.1, confidence=0.95)
        # 95% CI z-value is ~1.96
        assert moe == pytest.approx(1.96 * 0.1, rel=0.01)

    def test_moe_proportional_to_se(self):
        """Test that MOE is proportional to SE."""
        moe1 = margin_of_error(se=0.1)
        moe2 = margin_of_error(se=0.2)
        assert moe2 == pytest.approx(2 * moe1)


class TestACSReplicateWeightVariance:
    """Tests for ACSReplicateWeightVariance class."""

    @pytest.fixture
    def sample_df_with_replicates(self):
        """Create DataFrame with 80 replicate weights."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "indicator": np.random.choice([0, 1], n),
                "value": np.random.normal(100, 20, n),
                "PWGTP": np.random.randint(1, 1000, n),
            }
        )
        # Add 80 replicate weights
        for i in range(1, 81):
            df[f"PWGTP{i}"] = np.random.randint(1, 1000, n)
        return df

    @pytest.fixture
    def sample_df_no_replicates(self):
        """Create DataFrame without replicate weights."""
        np.random.seed(42)
        n = 100
        return pd.DataFrame(
            {
                "indicator": np.random.choice([0, 1], n),
                "value": np.random.normal(100, 20, n),
                "PWGTP": np.random.randint(1, 1000, n),
            }
        )

    def test_initialization(self):
        """Test class initialization."""
        var_estimator = ACSReplicateWeightVariance(weight_prefix="PWGTP")
        assert var_estimator.weight_prefix == "PWGTP"
        assert var_estimator.N_REPLICATES == 80
        assert var_estimator.FACTOR == pytest.approx(4.0 / 80.0)

    def test_check_replicate_weights_present(self, sample_df_with_replicates):
        """Test checking for replicate weights when present."""
        var_estimator = ACSReplicateWeightVariance()
        assert var_estimator.check_replicate_weights(sample_df_with_replicates) is True

    def test_check_replicate_weights_missing(self, sample_df_no_replicates):
        """Test checking for replicate weights when missing."""
        var_estimator = ACSReplicateWeightVariance()
        assert var_estimator.check_replicate_weights(sample_df_no_replicates) is False

    def test_get_replicate_weight_columns(self):
        """Test getting replicate weight column names."""
        var_estimator = ACSReplicateWeightVariance()
        cols = var_estimator.get_replicate_weight_columns(pd.DataFrame())
        assert len(cols) == 80
        assert cols[0] == "PWGTP1"
        assert cols[-1] == "PWGTP80"

    def test_compute_variance_proportion(self, sample_df_with_replicates):
        """Test variance computation for proportion."""
        var_estimator = ACSReplicateWeightVariance()
        estimate, variance, se = var_estimator.compute_variance_proportion(
            sample_df_with_replicates, "indicator"
        )
        assert 0 <= estimate <= 1
        assert variance >= 0
        assert se >= 0
        assert se == pytest.approx(np.sqrt(variance))

    def test_compute_variance_mean(self, sample_df_with_replicates):
        """Test variance computation for mean."""
        var_estimator = ACSReplicateWeightVariance()
        estimate, variance, se = var_estimator.compute_variance_mean(
            sample_df_with_replicates, "value"
        )
        assert variance >= 0
        assert se >= 0
        assert se == pytest.approx(np.sqrt(variance))

    def test_compute_variance_total(self, sample_df_with_replicates):
        """Test variance computation for total."""
        var_estimator = ACSReplicateWeightVariance()
        sample_df_with_replicates["ones"] = 1
        estimate, variance, se = var_estimator.compute_variance_total(
            sample_df_with_replicates, "ones"
        )
        assert estimate > 0  # Sum of weights
        assert variance >= 0
        assert se >= 0

    def test_variance_without_replicate_weights(self, sample_df_no_replicates):
        """Test that missing replicate weights returns NaN variance."""
        var_estimator = ACSReplicateWeightVariance()
        estimate, variance, se = var_estimator.compute_variance_proportion(
            sample_df_no_replicates, "indicator"
        )
        assert 0 <= estimate <= 1
        assert np.isnan(variance)
        assert np.isnan(se)

    def test_sdr_variance_helper(self, sample_df_with_replicates):
        """Test the private _compute_sdr_variance helper."""
        var_estimator = ACSReplicateWeightVariance()
        main_estimate = 0.5
        rep_estimates = np.array([0.48, 0.52, 0.49, 0.51] * 20)  # 80 estimates

        variance, se = var_estimator._compute_sdr_variance(main_estimate, rep_estimates)

        # Variance should be (4/80) * sum of squared differences
        expected_var = (4.0 / 80.0) * np.sum((rep_estimates - main_estimate) ** 2)
        assert variance == pytest.approx(expected_var)
        assert se == pytest.approx(np.sqrt(expected_var))
