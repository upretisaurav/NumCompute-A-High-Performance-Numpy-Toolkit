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
