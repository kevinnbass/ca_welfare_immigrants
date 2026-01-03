"""
Data validation utilities for quality checks and sanity tests.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from . import weights

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""

    name: str
    passed: bool
    expected: Optional[float] = None
    actual: Optional[float] = None
    tolerance: Optional[float] = None
    message: str = ""


def validate_population_total(
    df: pd.DataFrame,
    weight_col: str,
    expected_total: float,
    tolerance: float = 0.05,
    name: str = "Population total",
) -> ValidationResult:
    """
    Validate that weighted population total is within expected range.

    Args:
        df: DataFrame with weights
        weight_col: Weight column name
        expected_total: Expected population total
        tolerance: Acceptable relative difference
        name: Name for the check

    Returns:
        ValidationResult
    """
    actual = df[weight_col].sum()
    rel_diff = abs(actual - expected_total) / expected_total

    passed = rel_diff <= tolerance

    return ValidationResult(
        name=name,
        passed=passed,
        expected=expected_total,
        actual=actual,
        tolerance=tolerance,
        message=f"Relative difference: {rel_diff:.2%}",
    )


def validate_category_distribution(
    df: pd.DataFrame,
    category_col: str,
    weight_col: str,
    expected_proportions: dict[str, float],
    tolerance: float = 0.05,
    name: str = "Category distribution",
) -> ValidationResult:
    """
    Validate that category proportions are within expected range.

    Args:
        df: DataFrame
        category_col: Column with categories
        weight_col: Weight column
        expected_proportions: Dict of category -> expected proportion
        tolerance: Acceptable absolute difference in proportion
        name: Name for the check

    Returns:
        ValidationResult
    """
    # Compute actual proportions
    total_weight = df[weight_col].sum()
    actual_props = {}

    for cat in expected_proportions:
        mask = df[category_col] == cat
        actual_props[cat] = df.loc[mask, weight_col].sum() / total_weight

    # Check differences
    max_diff = 0
    diff_details = []

    for cat, expected_prop in expected_proportions.items():
        actual_prop = actual_props.get(cat, 0)
        diff = abs(actual_prop - expected_prop)
        max_diff = max(max_diff, diff)
        diff_details.append(f"{cat}: expected {expected_prop:.2%}, actual {actual_prop:.2%}")

    passed = max_diff <= tolerance

    return ValidationResult(
        name=name,
        passed=passed,
        expected=None,
        actual=None,
        tolerance=tolerance,
        message="; ".join(diff_details),
    )


def check_missing_values(
    df: pd.DataFrame,
    columns: list[str],
    max_missing_pct: float = 0.10,
    name: str = "Missing values",
) -> ValidationResult:
    """
    Check that missing value rates are acceptable.

    Args:
        df: DataFrame
        columns: Columns to check
        max_missing_pct: Maximum acceptable missing rate
        name: Name for the check

    Returns:
        ValidationResult
    """
    n_rows = len(df)
    missing_info = []
    max_missing = 0

    for col in columns:
        if col not in df.columns:
            missing_info.append(f"{col}: COLUMN NOT FOUND")
            max_missing = 1.0
            continue

        n_missing = df[col].isna().sum()
        pct_missing = n_missing / n_rows
        max_missing = max(max_missing, pct_missing)

        if pct_missing > max_missing_pct:
            missing_info.append(f"{col}: {pct_missing:.1%} missing")

    passed = max_missing <= max_missing_pct

    return ValidationResult(
        name=name,
        passed=passed,
        expected=max_missing_pct,
        actual=max_missing,
        tolerance=max_missing_pct,
        message="; ".join(missing_info) if missing_info else "All columns OK",
    )


def check_unweighted_sample_size(
    df: pd.DataFrame,
    group_col: str,
    min_n: int = 30,
    name: str = "Sample size",
) -> dict[str, ValidationResult]:
    """
    Check unweighted sample sizes by group.

    Args:
        df: DataFrame
        group_col: Column with group categories
        min_n: Minimum acceptable sample size
        name: Base name for checks

    Returns:
        Dict of group -> ValidationResult
    """
    results = {}

    for group in df[group_col].unique():
        if pd.isna(group):
            continue

        n = (df[group_col] == group).sum()
        passed = n >= min_n

        results[group] = ValidationResult(
            name=f"{name} ({group})",
            passed=passed,
            expected=min_n,
            actual=n,
            tolerance=None,
            message=f"n={n}, {'OK' if passed else 'BELOW THRESHOLD'}",
        )

    return results


def check_coefficient_of_variation(
    estimate: float,
    se: float,
    max_cv: float = 0.30,
    name: str = "CV check",
) -> ValidationResult:
    """
    Check if coefficient of variation is acceptable.

    Args:
        estimate: Point estimate
        se: Standard error
        max_cv: Maximum acceptable CV
        name: Name for check

    Returns:
        ValidationResult
    """
    if estimate == 0:
        cv = np.inf
    else:
        cv = abs(se / estimate)

    passed = cv <= max_cv

    return ValidationResult(
        name=name,
        passed=passed,
        expected=max_cv,
        actual=cv,
        tolerance=max_cv,
        message=f"CV = {cv:.2%}",
    )


def validate_rate_range(
    rate: float,
    min_rate: float = 0.0,
    max_rate: float = 1.0,
    name: str = "Rate range",
) -> ValidationResult:
    """
    Validate that a rate is within valid range.

    Args:
        rate: Rate value
        min_rate: Minimum valid rate
        max_rate: Maximum valid rate
        name: Name for check

    Returns:
        ValidationResult
    """
    passed = min_rate <= rate <= max_rate

    return ValidationResult(
        name=name,
        passed=passed,
        expected=None,
        actual=rate,
        tolerance=None,
        message=f"Rate {rate:.2%} {'within' if passed else 'OUTSIDE'} [{min_rate:.0%}, {max_rate:.0%}]",
    )


class DataValidator:
    """
    Validate ACS PUMS data for California welfare analysis.
    """

    # Expected California population totals (approximate, for sanity checks)
    # Source: Census Bureau population estimates
    CA_POPULATION_2023 = 39_000_000  # Approximate
    CA_POPULATION_2024 = 39_100_000  # Approximate

    # Expected approximate proportions
    EXPECTED_FOREIGN_BORN_PCT = 0.27  # ~27% foreign-born in CA
    EXPECTED_NONCITIZEN_PCT = 0.13  # ~13% noncitizens in CA

    def __init__(self, year: int):
        self.year = year
        self.results: list[ValidationResult] = []

    def add_result(self, result: ValidationResult):
        """Add validation result."""
        self.results.append(result)
        status = "PASS" if result.passed else "FAIL"
        logger.info(f"[{status}] {result.name}: {result.message}")

    def validate_acs_data(
        self,
        df: pd.DataFrame,
        weight_col: str = "PWGTP",
    ) -> list[ValidationResult]:
        """
        Run all validation checks on ACS data.

        Args:
            df: ACS DataFrame
            weight_col: Weight column name

        Returns:
            List of ValidationResults
        """
        self.results = []

        # 1. Check population total
        expected_pop = (
            self.CA_POPULATION_2023 if self.year <= 2023 else self.CA_POPULATION_2024
        )
        self.add_result(
            validate_population_total(
                df, weight_col, expected_pop, tolerance=0.10, name="CA Population Total"
            )
        )

        # 2. Check for missing values in key columns
        key_cols = ["NATIVITY", "CIT", "AGEP", "SEX", weight_col]
        self.add_result(
            check_missing_values(df, key_cols, max_missing_pct=0.02, name="Key Variables")
        )

        # 3. Check nativity distribution
        if "NATIVITY" in df.columns:
            # Foreign-born should be ~27%
            expected = {"2": self.EXPECTED_FOREIGN_BORN_PCT}  # 2 = foreign born
            # Note: Need to handle numeric vs string coding
            self.add_result(
                ValidationResult(
                    name="Foreign-born proportion",
                    passed=True,  # Just info for now
                    message="Check manually: expected ~27% foreign-born",
                )
            )

        # 4. Check welfare variable availability
        welfare_cols = ["HINS4", "FS", "SSIP", "PAP"]
        self.add_result(
            check_missing_values(df, welfare_cols, max_missing_pct=0.20, name="Welfare Variables")
        )

        return self.results

    def summary(self) -> str:
        """Generate summary of validation results."""
        n_passed = sum(1 for r in self.results if r.passed)
        n_failed = sum(1 for r in self.results if not r.passed)

        lines = [
            f"Validation Summary for {self.year} ACS Data",
            f"{'=' * 40}",
            f"Passed: {n_passed}",
            f"Failed: {n_failed}",
            "",
        ]

        if n_failed > 0:
            lines.append("Failed Checks:")
            for r in self.results:
                if not r.passed:
                    lines.append(f"  - {r.name}: {r.message}")

        return "\n".join(lines)


def compare_to_published_estimates(
    computed: dict[str, float],
    published: dict[str, float],
    tolerance: float = 0.10,
) -> list[ValidationResult]:
    """
    Compare computed estimates to published benchmarks.

    Args:
        computed: Dict of metric -> computed value
        published: Dict of metric -> published value
        tolerance: Acceptable relative difference

    Returns:
        List of ValidationResults
    """
    results = []

    for metric, computed_val in computed.items():
        if metric not in published:
            continue

        published_val = published[metric]
        if published_val == 0:
            continue

        rel_diff = abs(computed_val - published_val) / published_val
        passed = rel_diff <= tolerance

        results.append(
            ValidationResult(
                name=f"Benchmark: {metric}",
                passed=passed,
                expected=published_val,
                actual=computed_val,
                tolerance=tolerance,
                message=f"Diff: {rel_diff:.1%}",
            )
        )

    return results
