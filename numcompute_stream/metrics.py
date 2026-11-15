"""
Classification and regression metrics, all classification ones computed
from one confusion matrix. Arguments are (y_true, y_pred).

2.1 fix: F1/precision/recall give 0 instead of NaN on zero division, and
the averaging is explicit ("macro" or "binary").
"""

import numpy as np

from numcompute_stream.utils import safe_divide


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
