"""
Classification and regression metrics, all classification ones computed
from one confusion matrix. Arguments are (y_true, y_pred).

2.1 fix: F1/precision/recall give 0 instead of NaN on zero division, and
the averaging is explicit ("macro" or "binary").
"""

from collections import deque

import numpy as np

from numcompute_stream.utils import check_positive_int, safe_divide


def _check_labels(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays.")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred have different lengths: {len(y_true)} vs {len(y_pred)}."
        )
    return y_true, y_pred


def confusion_matrix(y_true, y_pred, labels=None):
    """
    Counts of (true, predicted) label pairs, shape (n_labels, n_labels).

    Rows are true labels, columns predicted. Raises ValueError if the
    lengths differ or a label is not in labels.
    """

    y_true, y_pred = _check_labels(y_true, y_pred)

    if labels is None:
        labels = np.union1d(y_true, y_pred)
    labels = np.asarray(labels)
    k = len(labels)

    if not np.all(np.isin(y_true, labels)) or not np.all(np.isin(y_pred, labels)):
        raise ValueError("y_true or y_pred contains a label that is not in labels.")

    true_idx = np.searchsorted(labels, y_true)
    pred_idx = np.searchsorted(labels, y_pred)

    # Each (true, pred) pair gets a unique number true * k + pred, so one
    # bincount fills the whole matrix.
    counts = np.bincount(true_idx * k + pred_idx, minlength=k * k)
    return counts.reshape(k, k)


def _per_class_scores(cm):
    """Per-class precision, recall and F1 from a confusion matrix."""

    tp = np.diag(cm)
    precision = safe_divide(tp, cm.sum(axis=0))  # 0 if never predicted
    recall = safe_divide(tp, cm.sum(axis=1))  # 0 if never in y_true
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return precision, recall, f1


def _average(per_class, labels, average, pos_label):
    """Average per-class scores: "macro" or "binary" (pos_label)."""

    if average == "macro":
        return float(np.mean(per_class)) if len(per_class) > 0 else 0.0
    if average == "binary":
        where = np.flatnonzero(labels == pos_label)
        return float(per_class[where[0]]) if len(where) > 0 else 0.0
    raise ValueError(f"average must be 'macro' or 'binary', got {average!r}.")


def accuracy_from_cm(cm):
    """Accuracy from a confusion matrix (NaN if empty)."""
    total = cm.sum()
    return float(np.trace(cm) / total) if total > 0 else float("nan")


def precision_from_cm(cm, labels, average="macro", pos_label=1):
    return _average(_per_class_scores(cm)[0], labels, average, pos_label)


def recall_from_cm(cm, labels, average="macro", pos_label=1):
    return _average(_per_class_scores(cm)[1], labels, average, pos_label)


def f1_from_cm(cm, labels, average="macro", pos_label=1):
    """Macro F1 is the mean of per-class F1 (not F1 of mean P and R)."""
    return _average(_per_class_scores(cm)[2], labels, average, pos_label)


def accuracy(y_true, y_pred):
    """Fraction of correct predictions. Raises ValueError on empty input."""

    y_true, y_pred = _check_labels(y_true, y_pred)
    if len(y_true) == 0:
        raise ValueError("accuracy of empty input is undefined.")
    return float(np.mean(y_true == y_pred))


def precision(y_true, y_pred, average="macro", pos_label=1):
    """TP / (TP + FP), 0 if never predicted. average: "macro" or "binary"."""

    labels = np.union1d(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels)
    return precision_from_cm(cm, labels, average, pos_label)


def recall(y_true, y_pred, average="macro", pos_label=1):
    """TP / (TP + FN), 0 if never in y_true. average: "macro" or "binary"."""

    labels = np.union1d(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels)
    return recall_from_cm(cm, labels, average, pos_label)


def f1_score(y_true, y_pred, average="macro", pos_label=1):
    """2PR / (P + R), 0 when P + R = 0. average: "macro" or "binary"."""

    labels = np.union1d(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels)
    return f1_from_cm(cm, labels, average, pos_label)


def mse(y_true, y_pred):
    """Mean squared error."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


# ---------------------------------------------------------------------------
# Streaming metrics
# ---------------------------------------------------------------------------


class ConfusionMatrix:
    """
    Confusion matrix accumulated over chunks. Grows when new labels appear;
    labels_ is always sorted.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.labels_ = None
        self.matrix_ = np.zeros((0, 0), dtype=int)
        return self

    def _grow_labels(self, labels):
        """Add rows/columns for any new labels, keeping old counts."""

        if self.labels_ is None:
            new_labels = np.unique(labels)
        else:
            new_labels = np.union1d(self.labels_, labels)

        if self.labels_ is not None and len(new_labels) == len(self.labels_):
            return

        grown = np.zeros((len(new_labels), len(new_labels)), dtype=int)
        if self.labels_ is not None:
            # Copy the old counts to their new positions.
            idx = np.searchsorted(new_labels, self.labels_)
            grown[np.ix_(idx, idx)] = self.matrix_
        self.labels_ = new_labels
        self.matrix_ = grown

    def update(self, y_true, y_pred):
        """Add the counts of one chunk."""

        y_true, y_pred = _check_labels(y_true, y_pred)
        self._grow_labels(np.union1d(y_true, y_pred))
        self.matrix_ += confusion_matrix(y_true, y_pred, self.labels_)
        return self

    def add(self, other):
        """Add the counts of another ConfusionMatrix."""

        if other.labels_ is None:
            return self
        self._grow_labels(other.labels_)
        idx = np.searchsorted(self.labels_, other.labels_)
        self.matrix_[np.ix_(idx, idx)] += other.matrix_
        return self


class _StreamingMetric:
    """Base class: keeps a ConfusionMatrix; subclasses implement compute(cm)."""

    def __init__(self):
        self.cm = ConfusionMatrix()

    def update(self, y_true, y_pred):
        """Add one chunk of true and predicted labels."""
        self.cm.update(y_true, y_pred)
        return self

    def result(self):
        """Score over all chunks since the last reset."""
        return self.compute(self.cm)

    def reset(self):
        self.cm.reset()
        return self


class Accuracy(_StreamingMetric):
    """Streaming accuracy (NaN before any update)."""

    def compute(self, cm):
        return accuracy_from_cm(cm.matrix_)


class Precision(_StreamingMetric):
    """Streaming precision, average "macro" or "binary" (with pos_label)."""

    def __init__(self, average="macro", pos_label=1):
        super().__init__()
        self.average = average
        self.pos_label = pos_label

    def compute(self, cm):
        if cm.labels_ is None:
            return 0.0
        return precision_from_cm(cm.matrix_, cm.labels_, self.average, self.pos_label)


class Recall(Precision):
    """Streaming recall, same options as Precision."""

    def compute(self, cm):
        if cm.labels_ is None:
            return 0.0
        return recall_from_cm(cm.matrix_, cm.labels_, self.average, self.pos_label)


class F1(Precision):
    """Streaming F1 (0 instead of NaN on zero division), same options as Precision."""

    def compute(self, cm):
        if cm.labels_ is None:
            return 0.0
        return f1_from_cm(cm.matrix_, cm.labels_, self.average, self.pos_label)


class RollingMetric:
    """A metric over only the last `window` chunks."""

    def __init__(self, metric, window):
        """metric: e.g. Accuracy(); window: number of recent chunks."""

        self.metric = metric
        self.window = check_positive_int(window, "window")
        self.reset()

    def update(self, y_true, y_pred):
        self._chunks.append(ConfusionMatrix().update(y_true, y_pred))
        return self

    def result(self):
        total = ConfusionMatrix()
        for cm in self._chunks:
            total.add(cm)
        return self.metric.compute(total)

    def reset(self):
        self._chunks = deque(maxlen=self.window)
        return self


class StreamingAUC:
    """
    Binary ROC AUC accumulated over chunks.

    Keeps histograms of the positive-class score for positive and negative
    rows (n_bins bins in [0, 1]) instead of every score. Scores in the same
    bin count as ties, so the result is approximate within one bin.
    """

    def __init__(self, n_bins=1000, pos_label=1):
        self.n_bins = check_positive_int(n_bins, "n_bins")
        self.pos_label = pos_label
        self.reset()

    def update(self, y_true, scores):
        """scores: positive-class scores in [0, 1], e.g. predict_proba(X)[:, 1]."""

        y_true, scores = _check_labels(y_true, scores)
        scores = np.clip(scores.astype(float), 0.0, 1.0)

        bins = np.minimum((scores * self.n_bins).astype(int), self.n_bins - 1)
        is_pos = y_true == self.pos_label
        self.pos_hist_ += np.bincount(bins[is_pos], minlength=self.n_bins)
        self.neg_hist_ += np.bincount(bins[~is_pos], minlength=self.n_bins)
        return self

    def result(self):
        """AUC in [0, 1], or NaN if only one class has been seen."""

        n_pos = self.pos_hist_.sum()
        n_neg = self.neg_hist_.sum()
        if n_pos == 0 or n_neg == 0:
            return float("nan")

        neg_below = np.cumsum(self.neg_hist_) - self.neg_hist_
        wins = np.sum(self.pos_hist_ * (neg_below + 0.5 * self.neg_hist_))
        return float(wins / (n_pos * n_neg))

    def reset(self):
        self.pos_hist_ = np.zeros(self.n_bins, dtype=int)
        self.neg_hist_ = np.zeros(self.n_bins, dtype=int)
        return self
