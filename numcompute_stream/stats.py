"""
Descriptive statistics. The functions work on whole arrays (NaN ignored);
StreamingStats updates the same kind of statistics one chunk at a time.
"""

from collections import deque

import numpy as np

from numcompute_stream.utils import chunk_moments, merge_moments, safe_divide


def mean(data, axis=0):
    """Mean along axis, ignoring NaN."""
    return np.nanmean(data, axis)


def median(data, axis=0):
    """Median along axis, ignoring NaN."""
    return np.nanmedian(data, axis)


def stan_dev(data, axis=0):
    """Standard deviation along axis, ignoring NaN."""
    return np.nanstd(data, axis)


def min(data, axis=0):
    """Minimum along axis, ignoring NaN."""
    return np.nanmin(data, axis)


def max(data, axis=0):
    """Maximum along axis, ignoring NaN."""
    return np.nanmax(data, axis)


def histogram(data, bins=10):
    """np.histogram of the non-NaN values."""
    return np.histogram(data[~np.isnan(data)], bins)


def quantile(data, q):
    """Quantile(s) q, ignoring NaN."""
    return np.nanquantile(data, q)


class StreamingStats:
    """
    Running per-feature count, mean, variance, min, max and an optional
    fixed-bin histogram. Old chunks are not stored.
    """

    def __init__(self, bin_edges=None, window=None):
        """
        bin_edges: increasing 1D edges used for every feature (None = no
        histogram). window: number of recent chunks for histogram(recent=True).
        """

        if bin_edges is not None:
            bin_edges = np.asarray(bin_edges, dtype=float)
            if bin_edges.ndim != 1 or len(bin_edges) < 2:
                raise ValueError("bin_edges must be a 1D array with at least 2 edges.")
            if np.any(np.diff(bin_edges) <= 0):
                raise ValueError("bin_edges must be strictly increasing.")
        if window is not None and window < 1:
            raise ValueError("window must be at least 1.")

        self.bin_edges = bin_edges
        self.window = window
        self.reset()

    def reset(self):
        """Forget all chunks."""

        self.n_chunks_ = 0
        self.count_ = None
        self.mean_ = None
        self._M2 = None
        self.min_ = None
        self.max_ = None
        self._hist = None
        self._recent_hists = deque(maxlen=self.window)
        return self

    def update_stats(self, X_chunk):
        """
        Add one chunk of shape (n_samples, n_features), or 1D for one feature.

        Raises ValueError if the number of features changes.
        """

        X = np.asarray(X_chunk, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_features = X.shape[1]
        if self.count_ is None:
            self.count_ = np.zeros(n_features)
            self.mean_ = np.zeros(n_features)
            self._M2 = np.zeros(n_features)
            self.min_ = np.full(n_features, np.nan)
            self.max_ = np.full(n_features, np.nan)
        elif n_features != len(self.count_):
            raise ValueError(
                f"Chunk has {n_features} features, expected {len(self.count_)}."
            )

        self.count_, self.mean_, self._M2 = merge_moments(
            self.count_, self.mean_, self._M2, *chunk_moments(X)
        )

        # fmin/fmax ignore NaN (and give NaN only if both sides are NaN).
        self.min_ = np.fmin(self.min_, np.fmin.reduce(X, axis=0))
        self.max_ = np.fmax(self.max_, np.fmax.reduce(X, axis=0))

        if self.bin_edges is not None:
            chunk_hist = np.array(
                [
                    np.histogram(X[:, j][~np.isnan(X[:, j])], self.bin_edges)[0]
                    for j in range(n_features)
                ]
            )
            if self._hist is None:
                self._hist = chunk_hist
            else:
                self._hist = self._hist + chunk_hist
            self._recent_hists.append(chunk_hist)

        self.n_chunks_ += 1
        return self

    @property
    def var_(self):
        """Population variance per feature."""
        return safe_divide(self._M2, self.count_)

    @property
    def std_(self):
        return np.sqrt(self.var_)

    def histogram(self, recent=False):
        """
        Counts of shape (n_features, n_bins); values outside bin_edges are
        ignored. recent=True uses only the last `window` chunks.
        """

        if self.bin_edges is None:
            raise ValueError("No bin_edges were given, so there is no histogram.")
        if self._hist is None:
            raise ValueError("No chunks seen yet.")
        if recent:
            if self.window is None:
                raise ValueError("Set window to use recent=True.")
            return np.sum(self._recent_hists, axis=0)
        return self._hist

    def quantile(self, q, recent=False):
        """
        Approximate quantiles from the histogram (linear inside a bin, so the
        error is at most one bin width). Returns shape (n_features,) for one q,
        else (len(q), n_features).
        """

        q_arr = np.atleast_1d(np.asarray(q, dtype=float))
        if np.any((q_arr < 0) | (q_arr > 1)):
            raise ValueError("q must be between 0 and 1.")

        hist = self.histogram(recent=recent)
        edges = self.bin_edges
        result = np.full((len(q_arr), hist.shape[0]), np.nan)

        for j, counts in enumerate(hist):
            total = counts.sum()
            if total == 0:
                continue
            cum = np.cumsum(counts)
            for i, qi in enumerate(q_arr):
                target = qi * total
                # First bin whose cumulative count reaches the target
                # (for q = 0, the first non-empty bin).
                side = "right" if target == 0 else "left"
                b = np.searchsorted(cum, target, side=side)
                if b >= len(counts):  # (min() is shadowed in this module)
                    b = len(counts) - 1
                before = cum[b] - counts[b]
                frac = safe_divide(target - before, counts[b])
                result[i, j] = edges[b] + frac * (edges[b + 1] - edges[b])

        if np.ndim(q) == 0:
            return result[0]
        return result
