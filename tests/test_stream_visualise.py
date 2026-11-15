"""Tests for numcompute_stream.visualise."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from numcompute_stream.visualise import (  # noqa: E402
    compare_models,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_plot_metric_over_time_returns_figure_and_saves(tmp_path):
    path = tmp_path / "acc.png"
    fig, ax = plot_metric_over_time(
        [np.nan, 0.8, 0.9], retrained=[False, True, False], save_path=path, show=False
    )
    assert isinstance(fig, plt.Figure)
    assert path.exists() and path.stat().st_size > 0


def test_plot_draws_on_given_axes():
    fig, ax = plt.subplots()
    _, returned_ax = compare_models([0.1, 0.2], [0.3, 0.4], ax=ax, show=False)
    assert returned_ax is ax
    assert len(ax.lines) == 2


def test_predictions_plot_with_string_labels():
    fig, ax = plot_predictions_vs_ground_truth(
        ["a", "b", "a"], ["a", "a", "a"], show=False
    )
    assert "0.667" in ax.get_title()


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        compare_models([0.1, 0.2], [0.3], show=False)
    with pytest.raises(ValueError):
        plot_predictions_vs_ground_truth([0, 1], [0], show=False)
    with pytest.raises(ValueError):
        plot_metric_over_time([0.1, 0.2], retrained=[True], show=False)
