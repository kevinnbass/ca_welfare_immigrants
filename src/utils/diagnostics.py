"""
Calibration diagnostics for status imputation models.

This module provides functions for assessing model calibration,
including calibration curves, expected calibration error (ECE),
and subgroup-level diagnostics.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

logger = logging.getLogger(__name__)


def compute_weighted_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    sample_weight: Optional[np.ndarray] = None,
    strategy: str = "uniform",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibration curve with optional sample weights.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for the positive class
        n_bins: Number of bins for calibration curve
        sample_weight: Optional sample weights
        strategy: Binning strategy ('uniform' or 'quantile')

    Returns:
        Tuple of (fraction_of_positives, mean_predicted_value, bin_counts)
    """
    if sample_weight is None:
        sample_weight = np.ones(len(y_true))

    # Determine bin edges
    if strategy == "uniform":
        bin_edges = np.linspace(0, 1, n_bins + 1)
    else:  # quantile
        bin_edges = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0

    # Assign predictions to bins
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])

    fraction_of_positives = np.zeros(n_bins)
    mean_predicted_value = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            w = sample_weight[mask]
            bin_counts[i] = w.sum()
            mean_predicted_value[i] = np.average(y_prob[mask], weights=w)
            fraction_of_positives[i] = np.average(y_true[mask], weights=w)

    # Filter out empty bins
    valid = bin_counts > 0
    return fraction_of_positives[valid], mean_predicted_value[valid], bin_counts[valid]


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    n_bins: int = 10,
) -> dict[str, float]:
    """
    Compute calibration metrics including ECE, MCE, and Brier score.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for the positive class
        sample_weight: Optional sample weights
        n_bins: Number of bins for ECE/MCE calculation

    Returns:
        Dictionary with calibration metrics:
        - ece: Expected Calibration Error
        - mce: Maximum Calibration Error
        - brier: Brier score
        - auc: Area under ROC curve
    """
    if sample_weight is None:
        sample_weight = np.ones(len(y_true))

    # Compute calibration curve
    frac_pos, mean_pred, bin_counts = compute_weighted_calibration_curve(
        y_true, y_prob, n_bins=n_bins, sample_weight=sample_weight
    )

    # ECE: weighted average of |accuracy - confidence| per bin
    calibration_errors = np.abs(frac_pos - mean_pred)
    total_weight = bin_counts.sum()
    ece = np.sum(calibration_errors * bin_counts) / total_weight if total_weight > 0 else 0.0

    # MCE: maximum calibration error
    mce = calibration_errors.max() if len(calibration_errors) > 0 else 0.0

    # Brier score (weighted)
    brier = np.average((y_prob - y_true) ** 2, weights=sample_weight)

    # AUC (unweighted - sklearn doesn't support weighted AUC directly)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan

    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier": float(brier),
        "auc": float(auc),
        "n_bins": n_bins,
        "n_samples": len(y_true),
    }


def compute_subgroup_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    subgroup_labels: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Compute calibration metrics for each subgroup.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for the positive class
        subgroup_labels: Array of subgroup labels for each observation
        sample_weight: Optional sample weights

    Returns:
        DataFrame with metrics for each subgroup
    """
    if sample_weight is None:
        sample_weight = np.ones(len(y_true))

    unique_subgroups = np.unique(subgroup_labels)
    results = []

    for subgroup in unique_subgroups:
        mask = subgroup_labels == subgroup
        if mask.sum() < 10:
            # Skip subgroups with too few observations
            continue

        metrics = compute_calibration_metrics(
            y_true[mask],
            y_prob[mask],
            sample_weight=sample_weight[mask],
        )
        metrics["subgroup"] = subgroup
        metrics["n_samples"] = int(mask.sum())
        metrics["weighted_n"] = float(sample_weight[mask].sum())
        metrics["prevalence"] = float(y_true[mask].mean())
        results.append(metrics)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    # Reorder columns
    cols = ["subgroup", "n_samples", "weighted_n", "prevalence", "auc", "ece", "mce", "brier"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].sort_values("subgroup")


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    sample_weight: Optional[np.ndarray] = None,
    title: str = "Calibration Curve",
    output_path: Optional[Path] = None,
) -> Optional["plt.Figure"]:
    """
    Plot calibration curve with optional sample weights.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for the positive class
        n_bins: Number of bins for calibration curve
        sample_weight: Optional sample weights
        title: Plot title
        output_path: Optional path to save the figure

    Returns:
        matplotlib Figure object, or None if matplotlib unavailable
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping calibration plot")
        return None

    # Compute calibration curve
    frac_pos, mean_pred, bin_counts = compute_weighted_calibration_curve(
        y_true, y_prob, n_bins=n_bins, sample_weight=sample_weight
    )

    # Compute metrics for annotation
    metrics = compute_calibration_metrics(y_true, y_prob, sample_weight, n_bins)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: calibration curve
    ax1.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax1.plot(mean_pred, frac_pos, "s-", label="Model")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Fraction of positives")
    ax1.set_title(title)
    ax1.legend(loc="lower right")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Add metrics annotation
    metrics_text = f"ECE: {metrics['ece']:.3f}\nMCE: {metrics['mce']:.3f}\nBrier: {metrics['brier']:.3f}"
    ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Right plot: histogram of predictions
    ax2.hist(y_prob, bins=n_bins, range=(0, 1), alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of Predictions")

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved calibration plot: {output_path}")

    return fig


def plot_subgroup_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    subgroup_labels: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    title: str = "Calibration by Subgroup",
    output_path: Optional[Path] = None,
) -> Optional["plt.Figure"]:
    """
    Plot calibration curves for each subgroup.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for the positive class
        subgroup_labels: Array of subgroup labels for each observation
        sample_weight: Optional sample weights
        title: Plot title
        output_path: Optional path to save the figure

    Returns:
        matplotlib Figure object, or None if matplotlib unavailable
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping subgroup calibration plot")
        return None

    if sample_weight is None:
        sample_weight = np.ones(len(y_true))

    unique_subgroups = sorted(np.unique(subgroup_labels))

    # Determine subplot layout
    n_subgroups = len(unique_subgroups)
    n_cols = min(3, n_subgroups)
    n_rows = (n_subgroups + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_subgroups == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, subgroup in enumerate(unique_subgroups):
        ax = axes[idx]
        mask = subgroup_labels == subgroup

        if mask.sum() < 10:
            ax.text(0.5, 0.5, f"{subgroup}\n(insufficient data)",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            continue

        frac_pos, mean_pred, _ = compute_weighted_calibration_curve(
            y_true[mask], y_prob[mask], n_bins=10, sample_weight=sample_weight[mask]
        )

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.plot(mean_pred, frac_pos, "s-")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"{subgroup} (n={mask.sum():,})")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    # Hide unused subplots
    for idx in range(n_subgroups, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved subgroup calibration plot: {output_path}")

    return fig


def generate_calibration_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    subgroup_labels: Optional[np.ndarray] = None,
    subgroup_name: str = "subgroup",
    output_dir: Optional[Path] = None,
    model_name: str = "model",
) -> dict:
    """
    Generate a complete calibration diagnostic report.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for the positive class
        sample_weight: Optional sample weights
        subgroup_labels: Optional array of subgroup labels
        subgroup_name: Name of the subgroup variable
        output_dir: Directory to save outputs
        model_name: Name of the model for file naming

    Returns:
        Dictionary with all computed metrics and file paths
    """
    results = {"model_name": model_name}

    # Overall metrics
    results["overall_metrics"] = compute_calibration_metrics(
        y_true, y_prob, sample_weight
    )

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save calibration plot
        cal_plot_path = output_dir / f"{model_name}_calibration_curve.png"
        plot_calibration_curve(
            y_true, y_prob,
            sample_weight=sample_weight,
            title=f"Calibration Curve - {model_name}",
            output_path=cal_plot_path,
        )
        results["calibration_plot"] = str(cal_plot_path)

    # Subgroup analysis
    if subgroup_labels is not None:
        subgroup_df = compute_subgroup_metrics(
            y_true, y_prob, subgroup_labels, sample_weight
        )
        results["subgroup_metrics"] = subgroup_df.to_dict(orient="records")

        if output_dir:
            # Save subgroup metrics CSV
            csv_path = output_dir / f"{model_name}_subgroup_calibration.csv"
            subgroup_df.to_csv(csv_path, index=False)
            results["subgroup_csv"] = str(csv_path)

            # Save subgroup calibration plot
            subgroup_plot_path = output_dir / f"{model_name}_subgroup_calibration.png"
            plot_subgroup_calibration(
                y_true, y_prob, subgroup_labels,
                sample_weight=sample_weight,
                title=f"Calibration by {subgroup_name} - {model_name}",
                output_path=subgroup_plot_path,
            )
            results["subgroup_plot"] = str(subgroup_plot_path)

    logger.info(f"Calibration report generated for {model_name}")
    logger.info(f"  ECE: {results['overall_metrics']['ece']:.4f}")
    logger.info(f"  MCE: {results['overall_metrics']['mce']:.4f}")
    logger.info(f"  Brier: {results['overall_metrics']['brier']:.4f}")

    return results
