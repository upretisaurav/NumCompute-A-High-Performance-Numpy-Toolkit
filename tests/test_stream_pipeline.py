"""Tests for numcompute_stream.pipeline (2.1 fix: y forwarding, partial_fit)."""

import numpy as np
import pytest

from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import MinMaxScaler, SimpleImputer, StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier


def make_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3)) * [1.0, 10.0, 100.0]
    y = (X[:, 0] + X[:, 1] / 10 > 0).astype(int)
    X[rng.random(X.shape) < 0.05] = np.nan
    return X, y


def make_pipe(buffer_size=100):
    return Pipeline(
        [
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            ("tree", DecisionTreeClassifier(max_depth=4, buffer_size=buffer_size)),
        ]
    )


def test_fit_forwards_y_to_the_model():
    # 2.1 called last_step.fit(X) with no y, so this raised a TypeError.
    X, y = make_data()
    pipe = make_pipe().fit(X, y)
    np.testing.assert_array_equal(pipe.named_steps["tree"].classes_, [0, 1])
    assert np.mean(pipe.predict(X) == y) > 0.9


def test_partial_fit_forwards_y_and_updates_transformers_per_chunk():
    X, y = make_data()
    pipe = make_pipe(buffer_size=100)
    for i, start in enumerate(range(0, 300, 50), start=1):
        pipe.partial_fit(X[start : start + 50], y[start : start + 50])
        # imputer removes NaN, so the scaler sees every row
        assert np.all(pipe.named_steps["scaler"].n_samples_seen_ == 50 * i)
    np.testing.assert_array_equal(pipe.final_estimator.classes_, [0, 1])
    assert pipe.final_estimator.n_retrains_ == 3


def test_pipeline_predictions_equal_doing_steps_by_hand():
    X, y = make_data()
    pipe = make_pipe().fit(X, y)

    imputer = SimpleImputer().fit(X)
    X1 = imputer.transform(X)
    scaler = StandardScaler().fit(X1)
    X2 = scaler.transform(X1)
    tree = DecisionTreeClassifier(max_depth=4).fit(X2, y)

    np.testing.assert_array_equal(pipe.predict(X), tree.predict(X2))
    np.testing.assert_array_equal(pipe.predict_proba(X), tree.predict_proba(X2))


def test_partial_fit_step_without_partial_fit_raises():
    pipe = Pipeline([("minmax", MinMaxScaler()), ("tree", DecisionTreeClassifier())])
    with pytest.raises(AttributeError, match="partial_fit"):
        pipe.partial_fit(np.zeros((4, 2)), [0, 1, 0, 1])


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        make_pipe().predict(np.zeros((2, 3)))
