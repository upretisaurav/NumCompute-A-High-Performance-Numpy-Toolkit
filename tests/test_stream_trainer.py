"""Tests for numcompute_stream.stream.StreamTrainer."""

import numpy as np
import pytest

from numcompute_stream.io import iter_chunks
from numcompute_stream.metrics import F1, StreamingAUC
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import SimpleImputer, StandardScaler
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier


def make_data(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] - X[:, 1] > 0).astype(int)
    return X, y


def test_history_has_one_row_per_chunk_and_first_chunks_not_scored():
    X, y = make_data()
    trainer = StreamTrainer(DecisionTreeClassifier(max_depth=4, buffer_size=200))
    history = trainer.run(iter_chunks(X, y, chunk_size=100))

    assert all(len(values) == 10 for values in history.values())
    # The first tree is trained after chunk 1 (200 rows), so chunks 0-1 are N/A.
    assert history["scored"][:3] == [False, False, True]
    assert np.isnan(history["cumulative_accuracy"][0])
    assert history["retrained"] == [False, True] * 5
    assert "N/A" in trainer.log_table()


def test_cumulative_accuracy_matches_manual_calculation():
    X, y = make_data()
    model = DecisionTreeClassifier(max_depth=3, buffer_size=100)
    trainer = StreamTrainer(model)

    correct, total = 0, 0
    for X_chunk, y_chunk in iter_chunks(X, y, chunk_size=100):
        if model.n_retrains_ > 0:
            correct += np.sum(model.predict(X_chunk) == y_chunk)
            total += len(y_chunk)
        trainer.step(X_chunk, y_chunk)
        if total > 0:
            assert trainer.history["cumulative_accuracy"][-1] == pytest.approx(
                correct / total
            )


def test_works_with_pipeline_and_extra_metrics():
    X, y = make_data()
    X[::17, 2] = np.nan
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            ("tree", DecisionTreeClassifier(max_depth=4, buffer_size=250)),
        ]
    )
    trainer = StreamTrainer(
        pipe, metrics={"f1": F1(average="binary"), "auc": StreamingAUC()}
    )
    history = trainer.run(iter_chunks(X, y, chunk_size=125))
    assert history["cumulative_accuracy"][-1] > 0.8
    assert 0.5 < history["auc"][-1] <= 1.0
    assert history["n_nodes"][-1] > 1
    assert trainer.last_y_pred.shape == (125,)


def test_model_without_partial_fit_raises():
    with pytest.raises(AttributeError):
        StreamTrainer(object())
