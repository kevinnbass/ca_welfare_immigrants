"""
Bootstrap model uncertainty infrastructure.

Implements bootstrap-based uncertainty propagation for the legal status
imputation model, combining three sources of variance:
1. Model uncertainty (from finite SIPP training sample)
2. Imputation uncertainty (from not observing true status in ACS)
3. Survey uncertainty (from ACS complex sample design)

Combining rule:
    V_total = V_survey + (1 + 1/M) * V_mi + (1 + 1/B) * V_model

Where:
- B = number of bootstrap replicates
- M = number of MI draws per bootstrap
- V_survey = SDR variance from replicate weights
- V_mi = between-imputation variance (within one bootstrap)
- V_model = between-bootstrap variance
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BootstrapMIResult:
    """Combined bootstrap + MI result with variance decomposition."""

    estimate: float
    total_variance: float
    survey_variance: float  # Within-imputation SDR variance (average)
    mi_variance: float  # Between-imputation variance (average across bootstraps)
    model_variance: float  # Between-bootstrap variance
    se_total: float
    ci_lower: float
    ci_upper: float
    df: float  # Degrees of freedom
    n_bootstraps: int
    n_imputations: int

    # Variance decomposition (fractions)
    fraction_survey: float  # V_survey / V_total
    fraction_mi: float  # V_mi / V_total
    fraction_model: float  # V_model / V_total


def bootstrap_sipp_sample(
    df: pd.DataFrame,
    weight_col: str = "WPFINWGT",
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Draw bootstrap sample from SIPP data.

    Uses weighted resampling with replacement to create a bootstrap
    sample that preserves approximate population characteristics.

    Args:
        df: SIPP DataFrame
        weight_col: Weight column for stratified sampling
        random_state: Random seed for reproducibility

    Returns:
        Bootstrap sample DataFrame (same size as original)
    """
    rng = np.random.default_rng(random_state)
    n = len(df)

    # Weighted sampling with replacement
    if weight_col in df.columns:
        weights = df[weight_col].values
        weights = weights / weights.sum()  # Normalize
        indices = rng.choice(n, size=n, replace=True, p=weights)
    else:
        indices = rng.choice(n, size=n, replace=True)

    return df.iloc[indices].reset_index(drop=True)


def train_bootstrap_models(
    sipp_df: pd.DataFrame,
    n_bootstraps: int,
    train_func: Callable[[pd.DataFrame], object],
    weight_col: str = "WPFINWGT",
    random_state: int = 42,
    n_jobs: int = 1,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> list:
    """
    Train B bootstrap models on resampled SIPP data.

    Args:
        sipp_df: SIPP training DataFrame
        n_bootstraps: Number of bootstrap replicates
        train_func: Function that takes DataFrame and returns fitted model
        weight_col: Weight column for bootstrap sampling
        random_state: Base random seed
        n_jobs: Number of parallel jobs (1 = sequential)
        progress_callback: Optional callback(current, total) for progress

    Returns:
        List of B fitted models
    """
    rng = np.random.SeedSequence(random_state)
    seeds = [int(s.generate_state(1)[0]) for s in rng.spawn(n_bootstraps)]

    models = []

    if n_jobs == 1:
        # Sequential processing
        for b, seed in enumerate(seeds):
            if progress_callback:
                progress_callback(b + 1, n_bootstraps)

            boot_df = bootstrap_sipp_sample(sipp_df, weight_col, random_state=seed)
            model = train_func(boot_df)
            models.append(model)
            logger.debug(f"Trained bootstrap model {b + 1}/{n_bootstraps}")

    else:
        # Parallel processing
        try:
            from joblib import Parallel, delayed

            def train_single(seed):
                boot_df = bootstrap_sipp_sample(sipp_df, weight_col, random_state=seed)
                return train_func(boot_df)

            models = Parallel(n_jobs=n_jobs)(delayed(train_single)(seed) for seed in seeds)
        except ImportError:
            logger.warning("joblib not available, falling back to sequential")
            return train_bootstrap_models(
                sipp_df, n_bootstraps, train_func, weight_col, random_state, n_jobs=1
            )

    logger.info(f"Trained {len(models)} bootstrap models")
    return models


def apply_bootstrap_models_to_acs(
    acs_df: pd.DataFrame,
    models: list,
    predict_func: Callable[[object, pd.DataFrame], np.ndarray],
    noncitizen_mask: Optional[pd.Series] = None,
) -> np.ndarray:
    """
    Apply B models to ACS data to get B sets of probabilities.

    Args:
        acs_df: ACS DataFrame
        models: List of B fitted models
        predict_func: Function(model, df) -> probabilities array
        noncitizen_mask: Optional mask for noncitizens only

    Returns:
        Array of shape (n_obs, B) with P(undocumented) from each model
    """
    if noncitizen_mask is not None:
        work_df = acs_df[noncitizen_mask].copy()
    else:
        work_df = acs_df

    n_obs = len(work_df)
    n_bootstraps = len(models)

    probabilities = np.zeros((n_obs, n_bootstraps))

    for b, model in enumerate(models):
        probs = predict_func(model, work_df)
        probabilities[:, b] = probs
        logger.debug(f"Applied bootstrap model {b + 1}/{n_bootstraps}")

    logger.info(f"Applied {n_bootstraps} models to {n_obs:,} observations")
    return probabilities


def create_bootstrap_mi_imputations(
    probabilities: np.ndarray,
    n_imputations_per_bootstrap: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """
    Create M imputation draws for each of B bootstrap probability sets.

    Args:
        probabilities: Array of shape (n_obs, B) with P(undocumented)
        n_imputations_per_bootstrap: Number of MI draws per bootstrap
        random_state: Random seed

    Returns:
        Array of shape (n_obs, B, M) with binary imputations
    """
    n_obs, n_bootstraps = probabilities.shape
    n_mi = n_imputations_per_bootstrap

    rng = np.random.default_rng(random_state)

    # Draw uniform random numbers for all imputations
    uniforms = rng.random((n_obs, n_bootstraps, n_mi))

    # Convert to binary based on probabilities
    # probabilities[:, :, np.newaxis] broadcasts to (n_obs, B, 1)
    imputations = (uniforms < probabilities[:, :, np.newaxis]).astype(int)

    logger.info(
        f"Created {n_bootstraps * n_mi} imputation sets "
        f"({n_bootstraps} bootstraps x {n_mi} imputations)"
    )

    return imputations


def combine_bootstrap_mi_results(
    bootstrap_estimates: np.ndarray,
    bootstrap_mi_variances: np.ndarray,
    confidence: float = 0.95,
) -> BootstrapMIResult:
    """
    Combine B bootstrap replicates, each with M multiple imputations.

    Combining rule:
    1. For each bootstrap b, combine M imputations via Rubin's rules:
       - Q_b = mean of estimates across m
       - U_b = mean of within-imp variances across m
       - B_m,b = variance of estimates across m (between-MI)
       - T_b = U_b + (1 + 1/M) * B_m,b

    2. Across bootstraps:
       - Q = mean of Q_b across b
       - B_boot = variance of Q_b across b (between-bootstrap / model variance)
       - Mean_T_b = mean of within-bootstrap total variance

    3. Total variance:
       - V_total = Mean_T_b + (1 + 1/B) * B_boot

    Args:
        bootstrap_estimates: Shape (B, M) - estimates for each (b, m)
        bootstrap_mi_variances: Shape (B, M) - SDR variance for each (b, m)
        confidence: Confidence level for intervals

    Returns:
        BootstrapMIResult with combined estimates and variance decomposition

    Reference:
        Adapted from Raghunathan et al. (2003) for nested MI structure
    """
    from scipy import stats

    B, M = bootstrap_estimates.shape

    # Step 1: For each bootstrap, combine imputations using Rubin's rules
    Q_b = np.zeros(B)  # Bootstrap-specific point estimates
    T_b = np.zeros(B)  # Bootstrap-specific total variances
    U_b_avg = np.zeros(B)  # Bootstrap-specific average within-imp variances
    B_m_b = np.zeros(B)  # Bootstrap-specific between-MI variances

    for b in range(B):
        estimates_m = bootstrap_estimates[b, :]
        variances_m = bootstrap_mi_variances[b, :]

        # Mean estimate across imputations
        Q_b[b] = np.nanmean(estimates_m)

        # Within-imputation variance (average)
        U_b_avg[b] = np.nanmean(variances_m)

        # Between-imputation variance
        B_m_b[b] = np.nanvar(estimates_m, ddof=1) if M > 1 else 0

        # Total variance for this bootstrap (Rubin's formula)
        T_b[b] = U_b_avg[b] + (1 + 1 / M) * B_m_b[b]

    # Step 2: Combine across bootstraps
    Q = np.mean(Q_b)  # Overall point estimate

    # Between-bootstrap variance (model variance)
    B_boot = np.var(Q_b, ddof=1) if B > 1 else 0

    # Average within-bootstrap total variance
    mean_T_b = np.mean(T_b)

    # Average survey variance (within-imputation)
    avg_survey_var = np.mean(U_b_avg)

    # Average MI variance
    avg_mi_var = np.mean(B_m_b)

    # Step 3: Total variance
    total_var = mean_T_b + (1 + 1 / B) * B_boot
    se_total = np.sqrt(total_var)

    # Degrees of freedom (simplified)
    if avg_survey_var > 0:
        # Barnard-Rubin style df adjustment
        r_model = (1 + 1 / B) * B_boot / mean_T_b if mean_T_b > 0 else 0
        df = (B - 1) * (1 + 1 / r_model) ** 2 if r_model > 0 else float("inf")
    else:
        df = B - 1

    # Confidence interval
    alpha = 1 - confidence
    if df > 0 and np.isfinite(df):
        t_crit = stats.t.ppf(1 - alpha / 2, df)
    else:
        t_crit = stats.norm.ppf(1 - alpha / 2)

    ci_lower = Q - t_crit * se_total
    ci_upper = Q + t_crit * se_total

    # Variance decomposition
    if total_var > 0:
        frac_survey = avg_survey_var / total_var
        frac_mi = (1 + 1 / M) * avg_mi_var / total_var
        frac_model = (1 + 1 / B) * B_boot / total_var
    else:
        frac_survey = frac_mi = frac_model = 0

    return BootstrapMIResult(
        estimate=Q,
        total_variance=total_var,
        survey_variance=avg_survey_var,
        mi_variance=avg_mi_var,
        model_variance=B_boot,
        se_total=se_total,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        df=df,
        n_bootstraps=B,
        n_imputations=M,
        fraction_survey=frac_survey,
        fraction_mi=frac_mi,
        fraction_model=frac_model,
    )


def save_bootstrap_models(
    models: list,
    output_dir: Path,
    model_name: str = "status_model",
) -> list[Path]:
    """
    Save bootstrap models to disk.

    Args:
        models: List of fitted models
        output_dir: Directory to save models
        model_name: Base name for model files

    Returns:
        List of paths to saved model files
    """
    import joblib

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for b, model in enumerate(models):
        path = output_dir / f"{model_name}_bootstrap_{b:04d}.joblib"
        joblib.dump(model, path)
        paths.append(path)

    logger.info(f"Saved {len(models)} bootstrap models to {output_dir}")
    return paths


def load_bootstrap_models(
    model_dir: Path,
    model_name: str = "status_model",
) -> list:
    """
    Load bootstrap models from disk.

    Args:
        model_dir: Directory containing model files
        model_name: Base name for model files

    Returns:
        List of loaded models
    """
    import joblib

    model_dir = Path(model_dir)
    pattern = f"{model_name}_bootstrap_*.joblib"

    model_files = sorted(model_dir.glob(pattern))

    if not model_files:
        raise FileNotFoundError(f"No bootstrap models found in {model_dir}")

    models = []
    for path in model_files:
        model = joblib.load(path)
        models.append(model)

    logger.info(f"Loaded {len(models)} bootstrap models from {model_dir}")
    return models


class BootstrapModelTrainer:
    """
    High-level interface for bootstrap model training and application.

    Example usage:
        trainer = BootstrapModelTrainer(n_bootstraps=100, n_imputations=5)
        trainer.train(sipp_df, train_function)
        trainer.save_models(output_dir)

        # Later:
        trainer.load_models(model_dir)
        probs = trainer.predict(acs_df, predict_function)
        imputations = trainer.create_imputations(probs)
    """

    def __init__(
        self,
        n_bootstraps: int = 100,
        n_imputations: int = 5,
        random_state: int = 42,
        n_jobs: int = 1,
    ):
        """
        Initialize trainer.

        Args:
            n_bootstraps: Number of bootstrap replicates (B)
            n_imputations: Number of MI draws per bootstrap (M)
            random_state: Base random seed
            n_jobs: Number of parallel jobs for training
        """
        self.n_bootstraps = n_bootstraps
        self.n_imputations = n_imputations
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models: list = []

    def train(
        self,
        sipp_df: pd.DataFrame,
        train_func: Callable[[pd.DataFrame], object],
        weight_col: str = "WPFINWGT",
    ) -> list:
        """
        Train bootstrap models.

        Args:
            sipp_df: SIPP training data
            train_func: Function(df) -> fitted model
            weight_col: Weight column for bootstrap sampling

        Returns:
            List of trained models
        """
        self.models = train_bootstrap_models(
            sipp_df,
            n_bootstraps=self.n_bootstraps,
            train_func=train_func,
            weight_col=weight_col,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        return self.models

    def predict(
        self,
        acs_df: pd.DataFrame,
        predict_func: Callable[[object, pd.DataFrame], np.ndarray],
        noncitizen_mask: Optional[pd.Series] = None,
    ) -> np.ndarray:
        """
        Apply all models to get probability matrix.

        Args:
            acs_df: ACS data
            predict_func: Function(model, df) -> probabilities
            noncitizen_mask: Optional filter mask

        Returns:
            Array of shape (n_obs, B) with probabilities
        """
        if not self.models:
            raise ValueError("No models trained or loaded. Call train() or load_models() first.")

        return apply_bootstrap_models_to_acs(acs_df, self.models, predict_func, noncitizen_mask)

    def create_imputations(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Create MI draws from probability matrix.

        Args:
            probabilities: Shape (n_obs, B)

        Returns:
            Array of shape (n_obs, B, M) with binary imputations
        """
        return create_bootstrap_mi_imputations(
            probabilities,
            n_imputations_per_bootstrap=self.n_imputations,
            random_state=self.random_state,
        )

    def save_models(self, output_dir: Path, model_name: str = "status_model") -> None:
        """Save all bootstrap models to disk."""
        if not self.models:
            raise ValueError("No models to save")
        save_bootstrap_models(self.models, output_dir, model_name)

    def load_models(self, model_dir: Path, model_name: str = "status_model") -> None:
        """Load bootstrap models from disk."""
        self.models = load_bootstrap_models(model_dir, model_name)
        self.n_bootstraps = len(self.models)
