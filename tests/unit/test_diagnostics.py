"""
Tests for calibration diagnostics functions.
"""

import numpy as np
import pytest


class TestCalibrationCurve:
    """Tests for calibration curve computation."""

    def test_compute_weighted_calibration_curve_import(self):
        """Test calibration curve function can be imported."""
        from src.utils.diagnostics import compute_weighted_calibration_curve
        assert compute_weighted_calibration_curve is not None

    def test_calibration_curve_bins(self):
        """Test calibration curve produces correct number of bins."""
        from src.utils.diagnostics import compute_weighted_calibration_curve

        np.random.seed(42)
        n = 1000
        y_true = np.random.binomial(1, 0.3, n)
        y_prob = np.clip(y_true + np.random.normal(0, 0.2, n), 0, 1)

        frac_pos, mean_pred, bin_counts = compute_weighted_calibration_curve(
            y_true, y_prob, n_bins=10
        )

        # Should have at most n_bins non-empty bins
        assert len(frac_pos) <= 10
        assert len(mean_pred) <= 10
        assert len(bin_counts) <= 10

    def test_calibration_curve_with_weights(self):
        """Test calibration curve with sample weights."""
        from src.utils.diagnostics import compute_weighted_calibration_curve

        np.random.seed(42)
        n = 1000
        y_true = np.random.binomial(1, 0.3, n)
        y_prob = np.clip(y_true + np.random.normal(0, 0.2, n), 0, 1)
        weights = np.random.uniform(0.5, 2.0, n)

        frac_pos, mean_pred, bin_counts = compute_weighted_calibration_curve(
            y_true, y_prob, n_bins=10, sample_weight=weights
        )

        # Should produce valid output
        assert np.all(frac_pos >= 0) and np.all(frac_pos <= 1)
        assert np.all(mean_pred >= 0) and np.all(mean_pred <= 1)
        assert np.all(bin_counts > 0)


class TestCalibrationMetrics:
    """Tests for calibration metric computation."""

    def test_compute_calibration_metrics_import(self):
        """Test metrics function can be imported."""
        from src.utils.diagnostics import compute_calibration_metrics
        assert compute_calibration_metrics is not None

    def test_ece_computation(self):
        """Test ECE (Expected Calibration Error) computation."""
        from src.utils.diagnostics import compute_calibration_metrics

        np.random.seed(42)
        n = 1000

        # Well-calibrated predictions
        y_true = np.random.binomial(1, 0.3, n)
        y_prob_good = np.clip(y_true * 0.6 + (1 - y_true) * 0.3 + np.random.normal(0, 0.1, n), 0, 1)

        # Poorly calibrated predictions
        y_prob_bad = np.clip(y_true * 0.9 + (1 - y_true) * 0.1, 0, 1)

        metrics_good = compute_calibration_metrics(y_true, y_prob_good)
        metrics_bad = compute_calibration_metrics(y_true, y_prob_bad)

        assert "ece" in metrics_good
        assert "mce" in metrics_good
        assert "brier" in metrics_good
        assert metrics_good["ece"] >= 0

    def test_brier_score_computation(self):
        """Test Brier score is computed correctly."""
        from src.utils.diagnostics import compute_calibration_metrics

        # Perfect predictions
        y_true = np.array([0, 0, 1, 1])
        y_prob_perfect = np.array([0.0, 0.0, 1.0, 1.0])
        y_prob_wrong = np.array([1.0, 1.0, 0.0, 0.0])

        metrics_perfect = compute_calibration_metrics(y_true, y_prob_perfect)
        metrics_wrong = compute_calibration_metrics(y_true, y_prob_wrong)

        # Perfect predictions should have Brier score of 0
        assert metrics_perfect["brier"] == pytest.approx(0.0, abs=0.001)

        # Wrong predictions should have Brier score of 1
        assert metrics_wrong["brier"] == pytest.approx(1.0, abs=0.001)


class TestSubgroupMetrics:
    """Tests for subgroup calibration metrics."""

    def test_compute_subgroup_metrics_import(self):
        """Test subgroup metrics function can be imported."""
        from src.utils.diagnostics import compute_subgroup_metrics
        assert compute_subgroup_metrics is not None

    def test_subgroup_metrics_by_region(self):
        """Test computing metrics by subgroup."""
        from src.utils.diagnostics import compute_subgroup_metrics
        import pandas as pd

        np.random.seed(42)
        n = 500

        y_true = np.random.binomial(1, 0.3, n)
        y_prob = np.clip(y_true + np.random.normal(0, 0.2, n), 0, 1)
        subgroups = np.random.choice(["Mexico", "Asia", "Europe"], n)

        result = compute_subgroup_metrics(y_true, y_prob, subgroups)

        assert isinstance(result, pd.DataFrame)
        assert "subgroup" in result.columns
        assert len(result) <= 3  # At most 3 subgroups


class TestCalibrationPlotting:
    """Tests for calibration plotting functions."""

    def test_plot_calibration_curve_import(self):
        """Test plotting function can be imported."""
        from src.utils.diagnostics import plot_calibration_curve
        assert plot_calibration_curve is not None

    def test_generate_calibration_report_import(self):
        """Test report generation function can be imported."""
        from src.utils.diagnostics import generate_calibration_report
        assert generate_calibration_report is not None

    def test_generate_calibration_report(self):
        """Test calibration report generation."""
        from src.utils.diagnostics import generate_calibration_report
        import tempfile
        from pathlib import Path

        np.random.seed(42)
        n = 200
        y_true = np.random.binomial(1, 0.3, n)
        y_prob = np.clip(y_true + np.random.normal(0, 0.2, n), 0, 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_calibration_report(
                y_true=y_true,
                y_prob=y_prob,
                output_dir=Path(tmpdir),
                model_name="test_model",
            )

            assert "model_name" in result
            assert "overall_metrics" in result
            assert result["overall_metrics"]["ece"] >= 0
