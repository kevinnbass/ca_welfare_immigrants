"""Tests for data validation utilities."""

import numpy as np
import pandas as pd
import pytest

from src.utils.validation import (
    DataValidator,
    ValidationResult,
    check_coefficient_of_variation,
    check_missing_values,
    check_unweighted_sample_size,
    compare_to_published_estimates,
    validate_category_distribution,
    validate_population_total,
    validate_rate_range,
)


class TestValidatePopulationTotal:
    """Tests for population total validation."""

    def test_valid_population_within_tolerance(self):
        """Test validation passes when population is within tolerance."""
        df = pd.DataFrame({"weight": [100, 200, 300]})
        result = validate_population_total(df, "weight", expected_total=600, tolerance=0.05)
        assert result.passed
        assert result.actual == 600
        assert result.expected == 600

    def test_population_slightly_below_tolerance(self):
        """Test validation passes when just within tolerance."""
        df = pd.DataFrame({"weight": [100, 200, 280]})  # 580, diff is 3.3%
        result = validate_population_total(df, "weight", expected_total=600, tolerance=0.05)
        assert result.passed

    def test_population_exceeds_tolerance(self):
        """Test validation fails when population exceeds tolerance."""
        df = pd.DataFrame({"weight": [100, 200, 400]})  # 700, diff is 16.7%
        result = validate_population_total(df, "weight", expected_total=600, tolerance=0.05)
        assert not result.passed
        assert "Relative difference" in result.message

    def test_zero_expected_total_raises(self):
        """Test that zero expected total causes div by zero in relative diff."""
        df = pd.DataFrame({"weight": [100]})
        # Note: the function divides by expected_total for relative diff
        # This will cause a division by zero
        result = validate_population_total(df, "weight", expected_total=0)
        # Result will have infinite rel_diff, so passed will be False
        assert not result.passed

    def test_empty_dataframe(self):
        """Test validation with empty dataframe."""
        df = pd.DataFrame({"weight": []})
        result = validate_population_total(df, "weight", expected_total=1000, tolerance=0.05)
        assert not result.passed
        assert result.actual == 0

    def test_custom_name(self):
        """Test that custom name is used."""
        df = pd.DataFrame({"weight": [100]})
        result = validate_population_total(df, "weight", expected_total=100, name="Custom Check")
        assert result.name == "Custom Check"


class TestValidateCategoryDistribution:
    """Tests for category distribution validation."""

    def test_distribution_within_tolerance(self):
        """Test validation passes when proportions match."""
        df = pd.DataFrame({"category": ["A", "A", "A", "B", "B"], "weight": [20, 20, 20, 20, 20]})
        expected = {"A": 0.60, "B": 0.40}
        result = validate_category_distribution(df, "category", "weight", expected, tolerance=0.05)
        assert result.passed is True

    def test_distribution_exceeds_tolerance(self):
        """Test validation fails when proportions differ."""
        df = pd.DataFrame(
            {
                "category": ["A", "A", "B"],
                "weight": [30, 30, 40],  # 60% A, 40% B
            }
        )
        expected = {"A": 0.80, "B": 0.20}  # 20% diff
        result = validate_category_distribution(df, "category", "weight", expected, tolerance=0.10)
        assert not result.passed

    def test_missing_category_treated_as_zero(self):
        """Test that missing categories are treated as 0 proportion."""
        df = pd.DataFrame({"category": ["A", "A"], "weight": [50, 50]})
        expected = {"A": 1.0, "B": 0.0}
        result = validate_category_distribution(df, "category", "weight", expected, tolerance=0.05)
        assert result.passed is True

    def test_message_contains_details(self):
        """Test that message contains category details."""
        df = pd.DataFrame({"category": ["A", "B"], "weight": [50, 50]})
        expected = {"A": 0.5, "B": 0.5}
        result = validate_category_distribution(df, "category", "weight", expected)
        assert "A:" in result.message
        assert "B:" in result.message


class TestCheckMissingValues:
    """Tests for missing value checks."""

    def test_no_missing_values(self):
        """Test passes when no missing values."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        result = check_missing_values(df, ["col1", "col2"], max_missing_pct=0.10)
        assert result.passed is True
        assert "All columns OK" in result.message

    def test_missing_values_within_threshold(self):
        """Test passes when missing values within threshold."""
        df = pd.DataFrame(
            {
                "col1": [1, None, 3, 4, 5, 6, 7, 8, 9, 10]  # 10% missing
            }
        )
        result = check_missing_values(df, ["col1"], max_missing_pct=0.10)
        assert result.passed

    def test_missing_values_exceed_threshold(self):
        """Test fails when missing values exceed threshold."""
        df = pd.DataFrame(
            {
                "col1": [1, None, None, None, 5]  # 60% missing
            }
        )
        result = check_missing_values(df, ["col1"], max_missing_pct=0.10)
        assert not result.passed
        assert "col1" in result.message

    def test_column_not_found(self):
        """Test fails when column doesn't exist."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        result = check_missing_values(df, ["col1", "missing_col"], max_missing_pct=0.10)
        assert not result.passed
        assert "COLUMN NOT FOUND" in result.message

    def test_all_missing(self):
        """Test handling of all-missing column."""
        df = pd.DataFrame({"col1": [None, None, None]})
        result = check_missing_values(df, ["col1"], max_missing_pct=0.10)
        assert not result.passed
        assert result.actual == 1.0


class TestCheckUnweightedSampleSize:
    """Tests for unweighted sample size checks."""

    def test_all_groups_above_threshold(self):
        """Test passes when all groups above threshold."""
        df = pd.DataFrame({"group": ["A"] * 50 + ["B"] * 40})
        results = check_unweighted_sample_size(df, "group", min_n=30)
        assert all(r.passed for r in results.values())

    def test_group_below_threshold(self):
        """Test fails when a group is below threshold."""
        df = pd.DataFrame({"group": ["A"] * 50 + ["B"] * 10})
        results = check_unweighted_sample_size(df, "group", min_n=30)
        assert results["A"].passed
        assert not results["B"].passed
        assert "BELOW THRESHOLD" in results["B"].message

    def test_edge_case_at_threshold(self):
        """Test exact threshold value."""
        df = pd.DataFrame({"group": ["A"] * 30})
        results = check_unweighted_sample_size(df, "group", min_n=30)
        assert results["A"].passed

    def test_nan_groups_skipped(self):
        """Test that NaN group values are skipped."""
        df = pd.DataFrame({"group": ["A", "A", None, None]})
        results = check_unweighted_sample_size(df, "group", min_n=1)
        assert len(results) == 1
        assert "A" in results


class TestCheckCoefficientOfVariation:
    """Tests for coefficient of variation checks."""

    def test_low_cv_passes(self):
        """Test passes when CV is low."""
        result = check_coefficient_of_variation(estimate=0.50, se=0.05, max_cv=0.30)
        assert result.passed is True
        assert result.actual == pytest.approx(0.10)  # 5% / 50%

    def test_high_cv_fails(self):
        """Test fails when CV exceeds threshold."""
        result = check_coefficient_of_variation(estimate=0.10, se=0.05, max_cv=0.30)
        assert result.passed is False
        assert result.actual == pytest.approx(0.50)  # 50% CV

    def test_zero_estimate_returns_inf(self):
        """Test that zero estimate produces infinite CV."""
        result = check_coefficient_of_variation(estimate=0, se=0.05, max_cv=0.30)
        assert result.passed is False
        assert result.actual == np.inf

    def test_cv_at_threshold(self):
        """Test exact threshold value passes."""
        result = check_coefficient_of_variation(estimate=1.0, se=0.30, max_cv=0.30)
        assert result.passed is True

    def test_negative_estimate_uses_absolute_value(self):
        """Test CV calculation handles negative estimates."""
        result = check_coefficient_of_variation(estimate=-0.50, se=0.05, max_cv=0.30)
        assert result.passed is True
        assert result.actual == pytest.approx(0.10)


class TestValidateRateRange:
    """Tests for rate range validation."""

    def test_valid_rate(self):
        """Test passes when rate is in valid range."""
        result = validate_rate_range(0.25)
        assert result.passed is True
        assert result.actual == 0.25

    def test_rate_exactly_zero(self):
        """Test passes when rate is exactly zero."""
        result = validate_rate_range(0.0)
        assert result.passed is True

    def test_rate_exactly_one(self):
        """Test passes when rate is exactly one."""
        result = validate_rate_range(1.0)
        assert result.passed is True

    def test_negative_rate_fails(self):
        """Test fails when rate is negative."""
        result = validate_rate_range(-0.1)
        assert result.passed is False
        assert "OUTSIDE" in result.message

    def test_rate_above_one_fails(self):
        """Test fails when rate exceeds one."""
        result = validate_rate_range(1.1)
        assert result.passed is False

    def test_custom_range(self):
        """Test custom min/max range."""
        result = validate_rate_range(0.75, min_rate=0.5, max_rate=1.0)
        assert result.passed is True

        result = validate_rate_range(0.25, min_rate=0.5, max_rate=1.0)
        assert result.passed is False


class TestDataValidator:
    """Tests for the DataValidator class."""

    def test_initialization(self):
        """Test validator initializes correctly."""
        validator = DataValidator(year=2023)
        assert validator.year == 2023
        assert validator.results == []

    def test_add_result(self):
        """Test adding results to validator."""
        validator = DataValidator(year=2023)
        result = ValidationResult(name="test", passed=True, message="ok")
        validator.add_result(result)
        assert len(validator.results) == 1
        assert validator.results[0].name == "test"

    def test_validate_acs_data(self, sample_acs_df):
        """Test full ACS validation workflow."""
        validator = DataValidator(year=2023)
        results = validator.validate_acs_data(sample_acs_df)
        assert len(results) > 0
        assert all(isinstance(r, ValidationResult) for r in results)

    def test_summary_with_all_passing(self):
        """Test summary output when all checks pass."""
        validator = DataValidator(year=2023)
        validator.add_result(ValidationResult(name="test1", passed=True, message="ok"))
        validator.add_result(ValidationResult(name="test2", passed=True, message="ok"))
        summary = validator.summary()
        assert "Passed: 2" in summary
        assert "Failed: 0" in summary

    def test_summary_with_failures(self):
        """Test summary output when checks fail."""
        validator = DataValidator(year=2023)
        validator.add_result(ValidationResult(name="test1", passed=True, message="ok"))
        validator.add_result(ValidationResult(name="test2", passed=False, message="failed"))
        summary = validator.summary()
        assert "Passed: 1" in summary
        assert "Failed: 1" in summary
        assert "Failed Checks:" in summary
        assert "test2" in summary

    def test_year_affects_expected_population(self):
        """Test that year affects expected population."""
        validator_2023 = DataValidator(year=2023)
        validator_2024 = DataValidator(year=2024)
        # Internal constants differ by year
        assert validator_2023.CA_POPULATION_2023 < validator_2024.CA_POPULATION_2024


class TestCompareToPublishedEstimates:
    """Tests for comparing to published benchmarks."""

    def test_within_tolerance(self):
        """Test passes when computed values match published."""
        computed = {"metric1": 0.25, "metric2": 0.50}
        published = {"metric1": 0.26, "metric2": 0.51}
        results = compare_to_published_estimates(computed, published, tolerance=0.10)
        assert all(r.passed for r in results)

    def test_outside_tolerance(self):
        """Test fails when values differ significantly."""
        computed = {"metric1": 0.25}
        published = {"metric1": 0.50}  # 100% diff
        results = compare_to_published_estimates(computed, published, tolerance=0.10)
        assert len(results) == 1
        assert results[0].passed is False

    def test_missing_published_value_skipped(self):
        """Test that missing published values are skipped."""
        computed = {"metric1": 0.25, "metric2": 0.50}
        published = {"metric1": 0.25}  # metric2 missing
        results = compare_to_published_estimates(computed, published)
        assert len(results) == 1

    def test_zero_published_value_skipped(self):
        """Test that zero published values are skipped."""
        computed = {"metric1": 0.25}
        published = {"metric1": 0}
        results = compare_to_published_estimates(computed, published)
        assert len(results) == 0

    def test_message_contains_difference(self):
        """Test that message includes the difference."""
        computed = {"metric1": 0.30}
        published = {"metric1": 0.25}
        results = compare_to_published_estimates(computed, published)
        assert "Diff:" in results[0].message


class TestValidationResultDataclass:
    """Tests for ValidationResult dataclass."""

    def test_creation_with_all_fields(self):
        """Test creating result with all fields."""
        result = ValidationResult(
            name="test", passed=True, expected=0.5, actual=0.48, tolerance=0.05, message="OK"
        )
        assert result.name == "test"
        assert result.passed is True
        assert result.expected == 0.5
        assert result.actual == 0.48
        assert result.tolerance == 0.05
        assert result.message == "OK"

    def test_creation_with_minimal_fields(self):
        """Test creating result with minimal fields."""
        result = ValidationResult(name="test", passed=False)
        assert result.name == "test"
        assert result.passed is False
        assert result.expected is None
        assert result.actual is None
        assert result.tolerance is None
        assert result.message == ""
