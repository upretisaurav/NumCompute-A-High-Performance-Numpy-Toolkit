"""
Plots for streaming results. Every function takes ax, save_path and show,
and returns (fig, ax).
"""

import matplotlib.pyplot as plt
import numpy as np


def _get_ax(ax):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure
    return fig, ax


def _finish(fig, save_path, show):
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()


def plot_metric_over_time(
    metric_values,
    title="Metric over time",
    ylabel="Accuracy",
    retrained=None,
    ax=None,
    save_path=None,
    show=True,
):
    """
    Line plot of a metric per chunk (NaN = gap). Chunks in `retrained` are
    marked with dotted lines. Raises ValueError on a length mismatch.
    """

    values = np.asarray(metric_values, dtype=float)
    fig, ax = _get_ax(ax)

    ax.plot(np.arange(len(values)), values, marker=".", label=ylabel)

    if retrained is not None:
        if len(retrained) != len(values):
            raise ValueError("retrained must have one entry per chunk.")
        for i in np.flatnonzero(retrained):
            ax.axvline(i, color="grey", linestyle=":", linewidth=0.8)
        ax.plot([], [], color="grey", linestyle=":", label="retrain")

    ax.set_title(title)
    ax.set_xlabel("Chunk")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.3)

    _finish(fig, save_path, show)
    return fig, ax


def compare_models(
    metric1,
    metric2,
    labels=("Model 1", "Model 2"),
    title="Model comparison",
    ylabel="Accuracy",
    ax=None,
    save_path=None,
    show=True,
):
    """Plot one metric for two models. Raises ValueError on a length mismatch."""

    metric1 = np.asarray(metric1, dtype=float)
    metric2 = np.asarray(metric2, dtype=float)
    if len(metric1) != len(metric2):
        raise ValueError(
            f"metric1 and metric2 have different lengths: {len(metric1)} vs "
            f"{len(metric2)}."
        )

    fig, ax = _get_ax(ax)
    chunks = np.arange(len(metric1))
    ax.plot(chunks, metric1, label=labels[0])
    ax.plot(chunks, metric2, label=labels[1])

    ax.set_title(title)
    ax.set_xlabel("Chunk")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.3)

    _finish(fig, save_path, show)
    return fig, ax


def plot_predictions_vs_ground_truth(
    y_true,
    y_pred,
    title="Predictions vs ground truth",
    ax=None,
    save_path=None,
    show=True,
):
    """
    True vs predicted label for each sample, wrong ones in red. Raises
    ValueError on a length mismatch.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred have different lengths: {len(y_true)} vs {len(y_pred)}."
        )

    # Map labels to positions 0..k-1 so string labels also work.
    labels = np.union1d(y_true, y_pred)
    true_pos = np.searchsorted(labels, y_true)
    pred_pos = np.searchsorted(labels, y_pred)
    wrong = true_pos != pred_pos
    idx = np.arange(len(y_true))

    fig, ax = _get_ax(ax)
    ax.scatter(
        idx,
        true_pos,
        marker="o",
        s=30,
        facecolors="none",
        edgecolors="black",
        label="true",
    )
    ax.scatter(
        idx[~wrong],
        pred_pos[~wrong],
        marker="x",
        s=15,
        color="tab:green",
        label="predicted (correct)",
    )
    ax.scatter(
        idx[wrong],
        pred_pos[wrong],
        marker="x",
        s=25,
        color="tab:red",
        label="predicted (wrong)",
    )

    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels([str(label) for label in labels])
    ax.set_title(f"{title} (accuracy {np.mean(~wrong):.3f})")
    ax.set_xlabel("Sample in chunk")
    ax.set_ylabel("Class")
    ax.legend(loc="center right")

    _finish(fig, save_path, show)
    return fig, ax
