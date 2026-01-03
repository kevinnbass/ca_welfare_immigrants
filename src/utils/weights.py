"""
Survey weight calculations and variance estimation utilities.

For ACS PUMS, uses successive difference replication (SDR) method.
"""

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def weighted_mean(
    values: Union[pd.Series, np.ndarray],
    weights: Union[pd.Series, np.ndarray],
) -> float:
    """
    Compute weighted mean.

    Args:
        values: Values to average
        weights: Survey weights

    Returns:
        Weighted mean
    """
    values = np.asarray(values)
    weights = np.asarray(weights)

    # Handle missing values
    mask = ~(np.isnan(values) | np.isnan(weights))
    values = values[mask]
    weights = weights[mask]

    if len(values) == 0 or weights.sum() == 0:
        return np.nan

    return np.average(values, weights=weights)


def weighted_proportion(
    indicator: Union[pd.Series, np.ndarray],
    weights: Union[pd.Series, np.ndarray],
    validate_binary: bool = True,
) -> float:
    """
    Compute weighted proportion (0-1 scale).

    Args:
        indicator: Binary indicator (0/1)
        weights: Survey weights
        validate_binary: If True, validate that indicator is binary (default True)

    Returns:
        Weighted proportion

    Raises:
        ValueError: If validate_binary=True and indicator contains non-binary values
    """
    indicator_arr = np.asarray(indicator)

    if validate_binary:
        # Get unique non-NaN values
        valid_values = indicator_arr[~np.isnan(indicator_arr)]
        if len(valid_values) > 0:
            unique_vals = np.unique(valid_values)
            if not set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                raise ValueError(
                    f"Indicator must be binary (0/1), found values: {unique_vals[:10]}"
                )

    return weighted_mean(indicator_arr, weights)


def weighted_sum(
    values: Union[pd.Series, np.ndarray],
    weights: Union[pd.Series, np.ndarray],
) -> float:
    """
    Compute weighted sum (population total).

    Args:
        values: Values to sum (often 1s for counts)
        weights: Survey weights

    Returns:
        Weighted sum
    """
    values = np.asarray(values)
    weights = np.asarray(weights)

    mask = ~(np.isnan(values) | np.isnan(weights))

    return np.sum(values[mask] * weights[mask])


def weighted_count(
    weights: Union[pd.Series, np.ndarray],
    mask: Optional[Union[pd.Series, np.ndarray]] = None,
) -> float:
    """
    Compute weighted count (sum of weights).

    Args:
        weights: Survey weights
        mask: Optional boolean mask for subsetting

    Returns:
        Sum of weights (population estimate)
    """
    weights = np.asarray(weights)

    if mask is not None:
        mask = np.asarray(mask)
        weights = weights[mask]

    return np.nansum(weights)


class ACSReplicateWeightVariance:
    """
    Compute variance using ACS successive difference replication (SDR).

    The ACS uses 80 replicate weights with the formula:
    Var(theta) = (4/80) * sum_r (theta_r - theta)^2

    Reference:
    https://www.census.gov/programs-surveys/acs/microdata/documentation.html
    """

    N_REPLICATES = 80
    FACTOR = 4.0 / 80.0  # SDR factor for ACS

    def __init__(self, weight_prefix: str = "PWGTP"):
        """
        Initialize variance estimator.

        Args:
            weight_prefix: Prefix for replicate weights (PWGTP or WGTP)
        """
        self.weight_prefix = weight_prefix

    def get_replicate_weight_columns(self, df: pd.DataFrame) -> list[str]:
        """Get list of replicate weight column names."""
        return [f"{self.weight_prefix}{i}" for i in range(1, self.N_REPLICATES + 1)]

    def check_replicate_weights(self, df: pd.DataFrame) -> bool:
        """Check if replicate weights are present in dataframe."""
        rep_cols = self.get_replicate_weight_columns(df)
        missing = [c for c in rep_cols if c not in df.columns]
        if missing:
            logger.warning(f"Missing replicate weights: {missing[:5]}...")
            return False
        return True

    def _compute_sdr_variance(
        self,
        main_estimate: float,
        replicate_estimates: np.ndarray,
    ) -> tuple[float, float]:
        """
        Compute SDR variance and standard error.

        Args:
            main_estimate: Estimate from main weights
            replicate_estimates: Array of estimates from replicate weights

        Returns:
            Tuple of (variance, standard_error)
        """
        variance = self.FACTOR * np.sum((replicate_estimates - main_estimate) ** 2)
        se = np.sqrt(variance)
        return variance, se

    def compute_variance_proportion(
        self,
        df: pd.DataFrame,
        indicator_col: str,
        main_weight_col: Optional[str] = None,
    ) -> tuple[float, float, float]:
        """
        Compute variance for a proportion estimate using replicate weights.

        Args:
            df: DataFrame with indicator and weights
            indicator_col: Column name for binary indicator
            main_weight_col: Column for main weight (default: weight_prefix without number)

        Returns:
            Tuple of (estimate, variance, standard_error)
        """
        if main_weight_col is None:
            main_weight_col = self.weight_prefix

        if not self.check_replicate_weights(df):
            logger.warning("Replicate weights not found, returning NaN variance")
            return weighted_proportion(df[indicator_col], df[main_weight_col]), np.nan, np.nan

        # Main estimate
        theta = weighted_proportion(df[indicator_col], df[main_weight_col])

        # Replicate estimates
        rep_cols = self.get_replicate_weight_columns(df)
        theta_reps = []

        for rep_col in rep_cols:
            theta_r = weighted_proportion(df[indicator_col], df[rep_col])
            theta_reps.append(theta_r)

        theta_reps = np.array(theta_reps)

        # SDR variance using common helper
        variance, se = self._compute_sdr_variance(theta, theta_reps)

        return theta, variance, se

    def compute_variance_mean(
        self,
        df: pd.DataFrame,
        value_col: str,
        main_weight_col: Optional[str] = None,
    ) -> tuple[float, float, float]:
        """
        Compute variance for a mean estimate using replicate weights.

        Args:
            df: DataFrame with values and weights
            value_col: Column name for values
            main_weight_col: Column for main weight

        Returns:
            Tuple of (estimate, variance, standard_error)
        """
        if main_weight_col is None:
            main_weight_col = self.weight_prefix

        if not self.check_replicate_weights(df):
            return weighted_mean(df[value_col], df[main_weight_col]), np.nan, np.nan

        # Main estimate
        theta = weighted_mean(df[value_col], df[main_weight_col])

        # Replicate estimates
        rep_cols = self.get_replicate_weight_columns(df)
        theta_reps = []

        for rep_col in rep_cols:
            theta_r = weighted_mean(df[value_col], df[rep_col])
            theta_reps.append(theta_r)

        theta_reps = np.array(theta_reps)

        # SDR variance using common helper
        variance, se = self._compute_sdr_variance(theta, theta_reps)

        return theta, variance, se

    def compute_variance_total(
        self,
        df: pd.DataFrame,
        value_col: str,
        main_weight_col: Optional[str] = None,
    ) -> tuple[float, float, float]:
        """
        Compute variance for a total (sum) estimate using replicate weights.

        Args:
            df: DataFrame with values and weights
            value_col: Column name for values (often 1s for counts)
            main_weight_col: Column for main weight

        Returns:
            Tuple of (estimate, variance, standard_error)
        """
        if main_weight_col is None:
            main_weight_col = self.weight_prefix

        if not self.check_replicate_weights(df):
            return weighted_sum(df[value_col], df[main_weight_col]), np.nan, np.nan

        # Main estimate
        theta = weighted_sum(df[value_col], df[main_weight_col])

        # Replicate estimates
        rep_cols = self.get_replicate_weight_columns(df)
        theta_reps = []

        for rep_col in rep_cols:
            theta_r = weighted_sum(df[value_col], df[rep_col])
            theta_reps.append(theta_r)

        theta_reps = np.array(theta_reps)

        # SDR variance using common helper
        variance, se = self._compute_sdr_variance(theta, theta_reps)

        return theta, variance, se


def confidence_interval(
    estimate: float,
    se: float,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Compute confidence interval.

    Args:
        estimate: Point estimate
        se: Standard error
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (lower, upper) bounds
    """
    from scipy import stats

    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha / 2)

    lower = estimate - z * se
    upper = estimate + z * se

    return lower, upper


def coefficient_of_variation(estimate: float, se: float) -> float:
    """
    Compute coefficient of variation (CV).

    Args:
        estimate: Point estimate
        se: Standard error

    Returns:
        CV as proportion (not percentage)
    """
    if estimate == 0:
        return np.inf

    return abs(se / estimate)


def margin_of_error(se: float, confidence: float = 0.90) -> float:
    """
    Compute margin of error (Census typically uses 90% CI).

    Args:
        se: Standard error
        confidence: Confidence level (default 0.90 for Census convention)

    Returns:
        Margin of error
    """
    from scipy import stats

    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha / 2)

    return z * se
