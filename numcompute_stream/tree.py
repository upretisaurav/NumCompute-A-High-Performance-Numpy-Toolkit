"""
Decision tree classifier built with NumPy, with buffered streaming updates.

Rules:
  - Split: a row goes left if x <= threshold; thresholds are midpoints
    between two different neighbouring values.
  - Ties: the first split found wins (lowest feature, then lowest threshold).
  - NaN: replaced by the feature's median from the training rows (0 if
    the feature is all NaN).
"""

import time

import numpy as np

from numcompute_stream.utils import (
    check_array,
    check_is_fitted,
    check_positive_int,
    check_X_y,
    safe_divide,
)


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


class Node:
    """A tree node; a leaf if feature is None. counts = class counts of its rows."""

    def __init__(self, counts):
        self.counts = counts
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None

    def is_leaf(self):
        return self.feature is None


def _count_nodes(node):
    if node.is_leaf():
        return 1
    return 1 + _count_nodes(node.left) + _count_nodes(node.right)


class DecisionTreeClassifier:
    """
    Decision tree classifier with a vectorised split search and buffered
    partial_fit for streams.
    """

    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        max_features=None,
        buffer_size=500,
        criterion="gini",
        random_state=None,
    ):
        """
        max_depth: None for no limit. min_samples_split: nodes with fewer rows
        become leaves. max_features: None, an int or "sqrt" (random subset per
        node). buffer_size: rows buffered before partial_fit retrains.
        criterion: "gini" or "entropy". Raises ValueError for invalid values.
        """

        if max_depth is not None:
            check_positive_int(max_depth, "max_depth")
        if check_positive_int(min_samples_split, "min_samples_split") < 2:
            raise ValueError("min_samples_split must be at least 2.")
        if max_features not in (None, "sqrt"):
            check_positive_int(max_features, "max_features")
        check_positive_int(buffer_size, "buffer_size")
        if criterion not in CRITERIA:
            raise ValueError(
                f"criterion must be 'gini' or 'entropy', got {criterion!r}."
            )

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.buffer_size = buffer_size
        self.criterion = criterion
        self.random_state = random_state

        # Streaming state, used by partial_fit.
        self._buffer_X = []
        self._buffer_y = []
        self.n_retrains_ = 0
        self.last_retrain_time_ = None

    def fit(self, X, y):
        """
        Train from scratch. X: (n_samples, n_features), NaN allowed; y: (n_samples,).

        Raises ValueError for bad shapes, empty input or Inf, TypeError if X is
        not numeric.
        """

        X, y = check_X_y(X, y)

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        self._buffer_X = []
        self._buffer_y = []

        self._train(X, y)
        return self

    def _train(self, X, y):
        """Build a new tree on X, y and time it."""

        start = time.perf_counter()

        # NaN rule: replace NaN with the column median of the training rows.
        self.fill_values_ = np.zeros(X.shape[1])
        has_values = ~np.all(np.isnan(X), axis=0)
        self.fill_values_[has_values] = np.nanmedian(X[:, has_values], axis=0)
        X = self._fill_nan(X)

        self._rng = np.random.default_rng(self.random_state)
        y_index = np.searchsorted(self.classes_, y)

        self.tree_ = self._build(X, y_index, depth=0)
        self.tree_classes_ = self.classes_.copy()
        self.n_nodes_ = _count_nodes(self.tree_)
        self.last_retrain_time_ = time.perf_counter() - start

    def _fill_nan(self, X):
        return np.where(np.isnan(X), self.fill_values_, X)

    def _features_to_try(self, n_features):
        """Feature indices to search at one node."""

        if self.max_features is None:
            return np.arange(n_features)
        if self.max_features == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        else:
            k = min(self.max_features, n_features)
        # Sorted so the tie rule (lowest feature index wins) still holds.
        return np.sort(self._rng.choice(n_features, size=k, replace=False))

    def _build(self, X, y, depth):
        """
        Recursively build the tree (y holds class indices). Makes a leaf at
        max_depth, below min_samples_split, when pure, or when no split helps.
        """

        n_classes = len(self.classes_)
        node = Node(np.bincount(y, minlength=n_classes))

        if self.max_depth is not None and depth >= self.max_depth:
            return node
        if len(y) < self.min_samples_split:
            return node
        if np.count_nonzero(node.counts) <= 1:
            return node

        best_feature, best_threshold, best_gain = None, None, 0.0
        for f in self._features_to_try(X.shape[1]):
            threshold, gain = best_split(X[:, f], y, n_classes, self.criterion)
            # Strict ">" keeps the lowest feature index on ties.
            if threshold is not None and gain > best_gain:
                best_feature, best_threshold, best_gain = f, threshold, gain

        if best_feature is None:
            return node

        go_left = X[:, best_feature] <= best_threshold
        node.feature = int(best_feature)
        node.threshold = best_threshold
        node.left = self._build(X[go_left], y[go_left], depth + 1)
        node.right = self._build(X[~go_left], y[~go_left], depth + 1)
        return node

    def _check_predict_input(self, X):
        """Check the tree is trained and X matches it, then fill NaN."""

        check_is_fitted(
            self,
            "tree_",
            msg="This DecisionTreeClassifier is not fitted yet. Call fit, or "
            "keep calling partial_fit until buffer_size rows have arrived "
            "(or call flush).",
        )

        X = check_array(X, allow_nan=True)
        if np.isinf(X).any():
            raise ValueError("Input contains Inf.")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the tree was fitted "
                f"with {self.n_features_in_}."
            )
        return self._fill_nan(X)

    def predict_proba(self, X):
        """
        Class probabilities from the leaf class proportions, shape
        (n_samples, n_classes) in classes_ order.

        Rows go down the tree in groups using masks (one step per node, not per
        row). Classes seen after the last retrain get 0. Raises NotFittedError
        if no tree has been trained.
        """

        X = self._check_predict_input(X)

        proba = np.zeros((X.shape[0], len(self.tree_classes_)))
        self._fill_leaf_proba(self.tree_, X, np.arange(X.shape[0]), proba)

        # New classes may have arrived since the last retrain.
        if len(self.tree_classes_) != len(self.classes_):
            full = np.zeros((X.shape[0], len(self.classes_)))
            full[:, np.searchsorted(self.classes_, self.tree_classes_)] = proba
            proba = full

        return proba

    def _fill_leaf_proba(self, node, X, rows, out):
        """Route the row indices down from node and fill out."""

        if len(rows) == 0:
            return
        if node.is_leaf():
            out[rows] = node.counts / node.counts.sum()
            return

        go_left = X[rows, node.feature] <= node.threshold
        self._fill_leaf_proba(node.left, X, rows[go_left], out)
        self._fill_leaf_proba(node.right, X, rows[~go_left], out)

    def predict(self, X):
        """Majority class of each row's leaf, shape (n_samples,)."""

        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_loop(self, X):
        """Same as predict, one row at a time. For benchmarking."""

        X = self._check_predict_input(X)
        y_pred = []
        for row in X:
            node = self.tree_
            while not node.is_leaf():
                if row[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            y_pred.append(self.tree_classes_[np.argmax(node.counts)])
        return np.array(y_pred)
