"""
Decision tree classifier built with NumPy, with buffered streaming updates.

Rules:
  - Split: a row goes left if x <= threshold; thresholds are midpoints
    between two different neighbouring values.
  - Ties: the first split found wins (lowest feature, then lowest threshold).
  - NaN: replaced by the feature's median from the training rows (0 if
    the feature is all NaN).
"""

import numpy as np

from numcompute_stream.utils import safe_divide


def gini(counts):
    """
    Gini impurity 1 - sum(p^2) from class counts.

    counts: (n_classes,) or (n_groups, n_classes). Empty groups give 0.
    """

    counts = np.asarray(counts, dtype=float)
    total = counts.sum(axis=-1, keepdims=True)
    p = safe_divide(counts, total)
    result = 1.0 - np.sum(p**2, axis=-1)
    # Empty groups would give 1 - 0 = 1, so set them to 0.
    result = np.where(total[..., 0] > 0, result, 0.0)

    if counts.ndim == 1:
        return float(result)
    return result


def entropy(counts):
    """
    Entropy -sum(p * log2(p)) in bits from class counts (p = 0 is skipped).

    counts: (n_classes,) or (n_groups, n_classes). Empty groups give 0.
    """

    counts = np.asarray(counts, dtype=float)
    total = counts.sum(axis=-1, keepdims=True)
    p = safe_divide(counts, total)

    # Use 1 in place of 0 inside the log: 0 * log2(1) = 0.
    terms = p * np.log2(np.where(p > 0, p, 1.0))
    result = -np.sum(terms, axis=-1) + 0.0  # + 0.0 turns -0.0 into 0.0

    if counts.ndim == 1:
        return float(result)
    return result


CRITERIA = {"gini": gini, "entropy": entropy}

# A split must improve impurity by more than this (avoids splits that
# only "gain" because of rounding error).
MIN_GAIN = 1e-12


def best_split(x, y, n_classes, criterion="gini"):
    """
    Best threshold for one feature, with no loop over thresholds.

    Sort the column once; cumsum of the one-hot labels gives the left class
    counts for every cut, and right = total - left. O(n log n).
    x: (n,), y: class indices (n,). Returns (threshold, gain), or
    (None, 0.0) if no split has gain > 0.
    """

    impurity = CRITERIA[criterion]
    n = len(x)

    order = np.argsort(x, kind="stable")
    x_sorted = x[order]
    one_hot = np.eye(n_classes)[y[order]]

    # left_counts[i] = class counts of rows 0..i (cut after row i)
    left_counts = np.cumsum(one_hot, axis=0)[:-1]
    total_counts = one_hot.sum(axis=0)
    right_counts = total_counts - left_counts

    n_left = np.arange(1, n)
    n_right = n - n_left
    child_impurity = (
        n_left * impurity(left_counts) + n_right * impurity(right_counts)
    ) / n
    gains = impurity(total_counts) - child_impurity

    # Never cut between two equal values.
    gains[x_sorted[:-1] == x_sorted[1:]] = -np.inf

    if len(gains) == 0:
        return None, 0.0

    best = np.argmax(gains)  # argmax keeps the first (lowest threshold) on ties
    if gains[best] <= MIN_GAIN:
        return None, 0.0

    threshold = (x_sorted[best] + x_sorted[best + 1]) / 2
    return float(threshold), float(gains[best])


def best_split_loop(x, y, n_classes, criterion="gini"):
    """
    Same as best_split with a Python loop over thresholds (O(n^2)). For benchmarking.
    """

    impurity = CRITERIA[criterion]
    n = len(x)
    parent = impurity(np.bincount(y, minlength=n_classes))

    values = np.unique(x)
    best_threshold, best_gain = None, 0.0

    for i in range(len(values) - 1):
        threshold = (values[i] + values[i + 1]) / 2
        left = x <= threshold

        left_counts = np.bincount(y[left], minlength=n_classes)
        right_counts = np.bincount(y[~left], minlength=n_classes)
        n_left = left.sum()
        n_right = n - n_left

        gain = (
            parent
            - (n_left * impurity(left_counts) + n_right * impurity(right_counts)) / n
        )

        # Strict ">" keeps the first threshold on ties.
        if gain > best_gain and gain > MIN_GAIN:
            best_threshold, best_gain = float(threshold), float(gain)

    return best_threshold, best_gain
