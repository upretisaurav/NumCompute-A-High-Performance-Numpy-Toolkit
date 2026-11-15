"""Tests for numcompute_stream.stats.StreamingStats."""

import numpy as np
import pytest

from numcompute_stream.stats import StreamingStats


def feed(stats, X, chunk_size):
    for start in range(0, len(X), chunk_size):
        stats.update_stats(X[start : start + chunk_size])
    return stats


def test_chunked_stats_match_full_data():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=3, scale=2, size=(1000, 3))
    stats = feed(StreamingStats(), X, 77)
    np.testing.assert_allclose(stats.mean_, X.mean(axis=0))
    np.testing.assert_allclose(stats.var_, X.var(axis=0))
    np.testing.assert_array_equal(stats.min_, X.min(axis=0))
    np.testing.assert_array_equal(stats.max_, X.max(axis=0))
    assert stats.n_chunks_ == 13


def test_nan_values_are_ignored():
    X = np.array([[1.0, np.nan], [np.nan, np.nan], [3.0, 5.0]])
    stats = feed(StreamingStats(), X, 1)
    np.testing.assert_array_equal(stats.count_, [2, 1])
    np.testing.assert_allclose(stats.mean_, [2.0, 5.0])
    np.testing.assert_array_equal(stats.min_, [1.0, 5.0])


def test_zero_variance_chunk():
    stats = StreamingStats().update_stats(np.full((20, 2), 4.0))
    np.testing.assert_array_equal(stats.var_, [0.0, 0.0])
    assert np.all(np.isfinite(stats.std_))


def test_histogram_counts_sum_to_n_and_match_numpy():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 10, size=500)
    edges = np.linspace(0, 10, 11)
    stats = feed(StreamingStats(bin_edges=edges), x, 60)
    hist = stats.histogram()
    assert hist.shape == (1, 10)
    assert hist.sum() == 500
    np.testing.assert_array_equal(hist[0], np.histogram(x, edges)[0])


def test_sliding_window_drops_old_chunks():
    stats = StreamingStats(bin_edges=[0, 1, 2], window=2)
    stats.update_stats(np.full(5, 0.5))  # old chunk, bin 0
    stats.update_stats(np.full(3, 1.5))
    stats.update_stats(np.full(4, 1.5))
    np.testing.assert_array_equal(stats.histogram(recent=True)[0], [0, 7])
    np.testing.assert_array_equal(stats.histogram()[0], [5, 7])


def test_quantile_close_to_exact_within_bin_width():
    rng = np.random.default_rng(2)
    x = rng.normal(size=5000)
    edges = np.linspace(-5, 5, 201)  # bin width 0.05
    stats = feed(StreamingStats(bin_edges=edges), x, 500)
    approx = stats.quantile([0.1, 0.5, 0.9])[:, 0]
    exact = np.quantile(x, [0.1, 0.5, 0.9])
    assert np.all(np.abs(approx - exact) <= 0.05)


def test_quantile_bad_q_and_no_histogram():
    stats = StreamingStats(bin_edges=[0, 1]).update_stats(np.array([0.5]))
    with pytest.raises(ValueError):
        stats.quantile(1.5)
    with pytest.raises(ValueError):
        StreamingStats().update_stats(np.array([1.0])).histogram()


def test_reset_forgets_everything():
    stats = StreamingStats(bin_edges=[0, 1]).update_stats(np.array([0.5, 0.7]))
    stats.reset()
    assert stats.n_chunks_ == 0
    stats.update_stats(np.array([0.2]))
    assert stats.count_[0] == 1
    assert stats.histogram().sum() == 1


def test_feature_count_change_raises():
    stats = StreamingStats().update_stats(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        stats.update_stats(np.zeros((2, 3)))
