"""
Multiple imputation utilities for legal status assignment.

Implements:
1. Bernoulli-based multiple imputation from probabilities
2. Calibration to external totals
3. Rubin's rules for combining multiple imputation results
"""

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MIResult:
    """Results from combining multiple imputations."""

    estimate: float
    within_variance: float  # Average within-imputation variance
    between_variance: float  # Between-imputation variance
    total_variance: float  # Combined variance
    se: float  # Total standard error
    ci_lower: float
    ci_upper: float
    df: float  # Degrees of freedom (Barnard-Rubin adjusted)
    n_imputations: int
    fraction_missing_info: float  # Fraction of information due to missingness


def create_bernoulli_imputations(
    probabilities: Union[pd.Series, np.ndarray],
    n_imputations: int = 10,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Create multiple imputed binary outcomes from probabilities.

    Args:
        probabilities: P(outcome=1) for each observation
        n_imputations: Number of imputations to create
        random_state: Random seed for reproducibility

    Returns:
        Array of shape (n_observations, n_imputations) with 0/1 values

    Raises:
        ValueError: If n_imputations < 1, probabilities contain NaN,
                   or probabilities are outside [0, 1]
    """
    # Validate inputs
    if n_imputations < 1:
        raise ValueError(f"n_imputations must be >= 1, got {n_imputations}")

    probs = np.asarray(probabilities)

    if np.any(np.isnan(probs)):
        raise ValueError("Probabilities contain NaN values")

    if (probs < 0).any() or (probs > 1).any():
        raise ValueError("Probabilities must be in [0, 1]")

    rng = np.random.default_rng(random_state)
    n_obs = len(probs)

    # Draw uniform random numbers for each imputation
    uniforms = rng.random((n_obs, n_imputations))

    # Convert to binary based on probabilities
    imputations = (uniforms < probs[:, np.newaxis]).astype(int)

    return imputations


def calibrate_to_total(
    df: pd.DataFrame,
    indicator_col: str,
    weight_col: str,
    target_total: float,
    method: str = "scaling",
) -> pd.DataFrame:
    """
    Calibrate imputed indicator to match external total.

    Args:
        df: DataFrame with indicator and weights
        indicator_col: Column with 0/1 indicator
        weight_col: Column with survey weights
        target_total: Target weighted total for indicator=1
        method: 'scaling' (adjust weights) or 'threshold' (adjust classification)

    Returns:
        DataFrame with calibrated weights or indicator

    Raises:
        ValueError: If target_total is not positive, current_total is zero,
                   or calibration ratio is extreme
    """
    # Strict validation
    if target_total <= 0:
        raise ValueError(f"target_total must be positive, got {target_total}")

    df = df.copy()

    current_total = (df[indicator_col] * df[weight_col]).sum()

    if abs(current_total) < 1e-10:
        raise ValueError(
            "Cannot calibrate: current_total is effectively zero. "
            "Check that indicator column has positive values."
        )

    # Check for extreme calibration ratio
    ratio = target_total / current_total
    if ratio > 100 or ratio < 0.01:
        raise ValueError(
            f"Extreme calibration ratio {ratio:.2f} detected "
            f"(target={target_total:,.0f}, current={current_total:,.0f}). "
            "This may indicate a data or target mismatch."
        )

    if method == "scaling":
        # Create calibration weight adjustment (ratio already computed above)
        # Only adjust weights for indicator=1 cases
        cal_weight_col = f"{weight_col}_calibrated"
        df[cal_weight_col] = df[weight_col].copy()
        df.loc[df[indicator_col] == 1, cal_weight_col] *= ratio

        logger.info(
            f"Calibrated weights: current={current_total:,.0f}, "
            f"target={target_total:,.0f}, ratio={ratio:.4f}"
        )

    elif method == "threshold":
        # Not implemented in this simple version
        # Would need probability column to re-threshold
        raise NotImplementedError("Threshold calibration requires probability column")

    return df


def calibrate_to_total_by_raking(
    df: pd.DataFrame,
    indicator_col: str,
    weight_col: str,
    target_total: float,
    strata_col: Optional[str] = None,
    max_iterations: int = 100,
    tolerance: float = 0.001,
) -> pd.DataFrame:
    """
    Calibrate using iterative proportional fitting (raking).

    Args:
        df: DataFrame with indicator and weights
        indicator_col: Column with 0/1 indicator
        weight_col: Column with survey weights
        target_total: Target weighted total
        strata_col: Optional stratification column
        max_iterations: Maximum raking iterations
        tolerance: Convergence tolerance

    Returns:
        DataFrame with calibrated weights

    Raises:
        ValueError: If parameters are invalid or raking fails to converge
    """
    # Strict parameter validation
    if target_total <= 0:
        raise ValueError(f"target_total must be positive, got {target_total}")
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
    if tolerance <= 0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")

    df = df.copy()
    cal_weight_col = f"{weight_col}_calibrated"
    df[cal_weight_col] = df[weight_col].copy()

    converged = False
    for iteration in range(max_iterations):
        current_total = (df[indicator_col] * df[cal_weight_col]).sum()

        if abs(current_total) < 1e-10:
            raise ValueError(
                f"Raking failed: current_total became zero at iteration {iteration + 1}"
            )

        ratio = target_total / current_total

        # Check convergence
        if abs(ratio - 1.0) < tolerance:
            logger.info(f"Raking converged after {iteration + 1} iterations")
            converged = True
            break

        # Adjust weights
        df.loc[df[indicator_col] == 1, cal_weight_col] *= ratio

    if not converged:
        raise ValueError(
            f"Raking failed to converge after {max_iterations} iterations. "
            f"Final ratio: {ratio:.4f}, tolerance: {tolerance}"
        )

    return df


def combine_mi_results_rubins_rules(
    estimates: list[float],
    variances: list[float],
    confidence: float = 0.95,
) -> MIResult:
    """
    Combine multiple imputation results using Rubin's rules.

    Args:
        estimates: List of point estimates from each imputation
        variances: List of within-imputation variances
        confidence: Confidence level for interval

    Returns:
        MIResult with combined estimate and uncertainty

    Reference:
        Rubin, D.B. (1987). Multiple Imputation for Nonresponse in Surveys.
    """
    from scipy import stats

    m = len(estimates)
    if m != len(variances):
        raise ValueError("estimates and variances must have same length")

    estimates = np.array(estimates)
    variances = np.array(variances)

    # Combined estimate (mean across imputations)
    q_bar = np.mean(estimates)

    # Within-imputation variance (average)
    u_bar = np.mean(variances)

    # Between-imputation variance
    b = np.var(estimates, ddof=1)  # Sample variance with n-1 denominator

    # Total variance (Rubin's formula)
    total_var = u_bar + (1 + 1 / m) * b

    # Standard error
    se = np.sqrt(total_var)

    # Degrees of freedom (Barnard-Rubin adjustment)
    if u_bar > 0 and not np.isnan(u_bar):
        r = (1 + 1 / m) * b / u_bar  # Relative increase in variance
        # Handle case where r is very small (near-zero between-imputation variance)
        if r > 0:
            df_old = (m - 1) * (1 + 1 / r) ** 2  # Original Rubin df
        else:
            df_old = float("inf")  # No imputation variance contribution

        # Barnard-Rubin adjustment for small samples
        # (simplified version - full version needs sample size)
        df = df_old
    else:
        r = np.inf if (b > 0) else 0
        df = m - 1

    # Fraction of missing information
    if total_var > 0 and not np.isnan(total_var):
        gamma = (1 + 1 / m) * b / total_var
    else:
        gamma = 0.0 if b == 0 else 1.0

    # Confidence interval
    alpha = 1 - confidence
    if df > 0 and np.isfinite(df):
        t_crit = stats.t.ppf(1 - alpha / 2, df)
    else:
        t_crit = stats.norm.ppf(1 - alpha / 2)

    ci_lower = q_bar - t_crit * se
    ci_upper = q_bar + t_crit * se

    return MIResult(
        estimate=q_bar,
        within_variance=u_bar,
        between_variance=b,
        total_variance=total_var,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        df=df,
        n_imputations=m,
        fraction_missing_info=gamma,
    )


def run_mi_analysis(
    df: pd.DataFrame,
    imputation_col_prefix: str,
    n_imputations: int,
    analysis_func: Callable[[pd.DataFrame, str], tuple[float, float]],
    **analysis_kwargs,
) -> MIResult:
    """
    Run analysis across multiple imputations and combine.

    Args:
        df: DataFrame with imputed indicator columns
        imputation_col_prefix: Prefix for imputation columns (e.g., 'status_imp_')
        n_imputations: Number of imputations
        analysis_func: Function that takes (df, indicator_col) and returns (estimate, variance)
        **analysis_kwargs: Additional arguments for analysis_func

    Returns:
        Combined MIResult
    """
    estimates = []
    variances = []

    for i in range(n_imputations):
        imp_col = f"{imputation_col_prefix}{i}"

        if imp_col not in df.columns:
            raise ValueError(f"Imputation column not found: {imp_col}")

        est, var = analysis_func(df, imp_col, **analysis_kwargs)
        estimates.append(est)
        variances.append(var)

    return combine_mi_results_rubins_rules(estimates, variances)


class MultipleImputationEngine:
    """
    Engine for creating and managing multiple imputations of legal status.
    """

    def __init__(
        self,
        n_imputations: int = 10,
        random_state: Optional[int] = None,
    ):
        self.n_imputations = n_imputations
        self.random_state = random_state
        self._imputed_datasets: list[pd.DataFrame] = []

    def impute_status(
        self,
        df: pd.DataFrame,
        prob_col: str,
        noncitizen_mask: pd.Series,
        calibration_target: Optional[float] = None,
        weight_col: str = "PWGTP",
    ) -> pd.DataFrame:
        """
        Create multiple imputed status indicators.

        Args:
            df: DataFrame with probability predictions
            prob_col: Column with P(undocumented)
            noncitizen_mask: Boolean mask for noncitizens
            calibration_target: Target total for undocumented (optional)
            weight_col: Weight column name

        Returns:
            DataFrame with imputation columns added
        """
        df = df.copy()

        # Get probabilities for noncitizens
        probs = df.loc[noncitizen_mask, prob_col].values

        # Create imputations
        imputations = create_bernoulli_imputations(
            probs,
            n_imputations=self.n_imputations,
            random_state=self.random_state,
        )

        # Add imputation columns
        for i in range(self.n_imputations):
            col_name = f"undoc_imp_{i}"
            df[col_name] = 0  # Default for non-noncitizens

            # Assign imputed values for noncitizens
            df.loc[noncitizen_mask, col_name] = imputations[:, i]

        # Calibrate if target provided
        if calibration_target is not None:
            for i in range(self.n_imputations):
                col_name = f"undoc_imp_{i}"
                df = calibrate_to_total(
                    df,
                    indicator_col=col_name,
                    weight_col=weight_col,
                    target_total=calibration_target,
                )

        return df

    def create_status_column(
        self,
        df: pd.DataFrame,
        imputation_index: int,
        nativity_col: str = "NATIVITY",
        citizenship_col: str = "CIT",
    ) -> pd.Series:
        """
        Create final status column for a specific imputation.

        Categories:
        - US_BORN: Native born
        - NATURALIZED: Foreign-born citizen
        - LEGAL_NONCITIZEN: Noncitizen, imputed as legal
        - ILLEGAL: Noncitizen, imputed as illegal

        Args:
            df: DataFrame with imputation and demographic columns
            imputation_index: Which imputation to use
            nativity_col: Nativity column name
            citizenship_col: Citizenship column name

        Returns:
            Series with status categories
        """
        imp_col = f"undoc_imp_{imputation_index}"

        status = pd.Series(index=df.index, dtype="object")

        # US-born
        us_born_mask = df[nativity_col] == 1
        status[us_born_mask] = "US_BORN"

        # Naturalized citizens (foreign-born citizens)
        naturalized_mask = (df[nativity_col] == 2) & (df[citizenship_col] != 5)
        status[naturalized_mask] = "NATURALIZED"

        # Noncitizens: use imputation
        noncitizen_mask = df[citizenship_col] == 5

        if imp_col in df.columns:
            # Legal noncitizens (imputed as legal)
            legal_nc_mask = noncitizen_mask & (df[imp_col] == 0)
            status[legal_nc_mask] = "LEGAL_NONCITIZEN"

            # Illegal (imputed)
            undoc_mask = noncitizen_mask & (df[imp_col] == 1)
            status[undoc_mask] = "ILLEGAL"
        else:
            # If no imputation, classify all noncitizens as unknown
            status[noncitizen_mask] = "NONCITIZEN_UNKNOWN"

        return status
