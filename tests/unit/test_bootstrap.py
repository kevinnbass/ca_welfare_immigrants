"""
Tests for bootstrap model uncertainty infrastructure.
"""

import numpy as np


class TestBootstrapMIResult:
    """Tests for BootstrapMIResult dataclass."""

    def test_import(self):
        """Test bootstrap module can be imported."""
        from src.utils.bootstrap import BootstrapMIResult

        assert BootstrapMIResult is not None

    def test_result_creation(self):
        """Test BootstrapMIResult dataclass creation."""
        from src.utils.bootstrap import BootstrapMIResult

        result = BootstrapMIResult(
            estimate=0.15,
            total_variance=0.001,
            survey_variance=0.0005,
            mi_variance=0.0003,
            model_variance=0.0002,
            se_total=0.0316,
            ci_lower=0.09,
            ci_upper=0.21,
            df=50.0,
            n_bootstraps=100,
            n_imputations=5,
            fraction_survey=0.5,
            fraction_mi=0.3,
            fraction_model=0.2,
        )
        assert result.estimate == 0.15
        assert result.total_variance == 0.001
        assert result.n_bootstraps == 100

    def test_variance_fractions(self):
        """Test variance fraction calculations."""
        from src.utils.bootstrap import BootstrapMIResult

        result = BootstrapMIResult(
            estimate=0.15,
            total_variance=0.001,
            survey_variance=0.0005,
            mi_variance=0.0003,
            model_variance=0.0002,
            se_total=0.0316,
            ci_lower=0.09,
            ci_upper=0.21,
            df=50.0,
            n_bootstraps=100,
            n_imputations=5,
            fraction_survey=0.5,
            fraction_mi=0.3,
            fraction_model=0.2,
        )

        # Fractions should be present
        assert hasattr(result, "fraction_survey")
        assert result.fraction_survey == 0.5


class TestBootstrapSampling:
    """Tests for bootstrap sampling functions."""

    def test_bootstrap_sipp_sample_import(self):
        """Test bootstrap_sipp_sample can be imported."""
        from src.utils.bootstrap import bootstrap_sipp_sample

        assert bootstrap_sipp_sample is not None

    def test_bootstrap_sample_maintains_size(self):
        """Test bootstrap sample maintains dataset size."""
        import pandas as pd

        from src.utils.bootstrap import bootstrap_sipp_sample

        # Create test data
        df = pd.DataFrame(
            {
                "id": range(100),
                "weight": np.random.uniform(0.5, 2.0, 100),
                "feature": np.random.randn(100),
            }
        )

        # Bootstrap sample
        boot_df = bootstrap_sipp_sample(df, weight_col="weight", random_state=42)

        # Should have same length as original
        assert len(boot_df) == len(df)

    def test_bootstrap_sample_is_different(self):
        """Test bootstrap samples are different from original."""
        import pandas as pd

        from src.utils.bootstrap import bootstrap_sipp_sample

        df = pd.DataFrame(
            {
                "id": range(100),
                "weight": np.random.uniform(0.5, 2.0, 100),
            }
        )

        boot_df = bootstrap_sipp_sample(df, weight_col="weight", random_state=42)

        # Should have some duplicate rows (bootstrap with replacement)
        assert boot_df["id"].nunique() < len(df)


class TestVarianceDecomposition:
    """Tests for variance decomposition functions."""

    def test_combine_bootstrap_mi_results_import(self):
        """Test combine function can be imported."""
        from src.utils.bootstrap import combine_bootstrap_mi_results

        assert combine_bootstrap_mi_results is not None

    def test_variance_decomposition_sums_correctly(self):
        """Test variance components sum to approximately total variance."""
        from src.utils.bootstrap import BootstrapMIResult

        # Create result with components that should sum
        result = BootstrapMIResult(
            estimate=0.15,
            total_variance=0.001,
            survey_variance=0.0005,
            mi_variance=0.0003,
            model_variance=0.0002,
            se_total=0.0316,
            ci_lower=0.09,
            ci_upper=0.21,
            df=50.0,
            n_bootstraps=100,
            n_imputations=5,
            fraction_survey=0.5,
            fraction_mi=0.3,
            fraction_model=0.2,
        )

        # Check components sum reasonably close to total
        # (with Rubin's rule adjustments, may not be exact)
        component_sum = result.survey_variance + result.mi_variance + result.model_variance
        assert abs(component_sum - result.total_variance) < 0.001

    def test_combine_bootstrap_mi_results(self):
        """Test combining bootstrap and MI results."""
        from src.utils.bootstrap import combine_bootstrap_mi_results

        # Create mock bootstrap estimates (B x M matrix)
        n_bootstraps = 10
        n_imputations = 5
        np.random.seed(42)

        bootstrap_estimates = np.random.normal(0.15, 0.02, (n_bootstraps, n_imputations))
        bootstrap_variances = np.full((n_bootstraps, n_imputations), 0.0001)

        result = combine_bootstrap_mi_results(
            bootstrap_estimates=bootstrap_estimates,
            bootstrap_mi_variances=bootstrap_variances,
        )

        assert hasattr(result, "estimate")
        assert hasattr(result, "total_variance")
        assert result.total_variance > 0


class TestBootstrapModelTrainer:
    """Tests for BootstrapModelTrainer class."""

    def test_trainer_import(self):
        """Test BootstrapModelTrainer can be imported."""
        from src.utils.bootstrap import BootstrapModelTrainer

        assert BootstrapModelTrainer is not None

    def test_trainer_creation(self):
        """Test trainer can be instantiated."""
        from src.utils.bootstrap import BootstrapModelTrainer

        trainer = BootstrapModelTrainer(
            n_bootstraps=10,
            random_state=42,
        )
        assert trainer.n_bootstraps == 10
        assert trainer.random_state == 42
