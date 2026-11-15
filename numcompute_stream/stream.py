"""
Test-then-train loop: each chunk is scored with the current model before
it is used for training, so every score is on unseen data.
"""

import numpy as np

from numcompute_stream.exceptions import NotFittedError
from numcompute_stream.metrics import Accuracy, RollingMetric, StreamingAUC


class StreamTrainer:
    """
    Runs a model over a stream of chunks and logs per-chunk results in
    history (a dict of lists). Chunks before the first tree are logged as
    not scored, with NaN metrics.
    """

    def __init__(self, model, metrics=None, rolling_window=5):
        """
        model: a model or Pipeline with partial_fit and predict.
        metrics: optional dict of name -> streaming metric (StreamingAUC gets
        predict_proba scores). rolling_window: chunks for rolling_accuracy.
        """

        if not callable(getattr(model, "partial_fit", None)):
            raise AttributeError("model must have a partial_fit() method.")

        self.model = model
        self.metrics = metrics if metrics is not None else {}
        self.cumulative_accuracy = Accuracy()
        self.rolling_accuracy = RollingMetric(Accuracy(), window=rolling_window)

        self.history = {
            key: []
            for key in [
                "chunk",
                "n_samples",
                "scored",
                "chunk_accuracy",
                "cumulative_accuracy",
                "rolling_accuracy",
                *self.metrics,
                "retrained",
                "retrain_time",
                "buffer_bytes",
                "n_nodes",
            ]
        }
        self.last_y_true = None
        self.last_y_pred = None

    @property
    def tree(self):
        """The last step of a pipeline, or the model itself."""
        return getattr(self.model, "final_estimator", self.model)

    def score_chunk(self, X, y):
        """Predict the chunk and update metrics; None if the model can't predict yet."""

        try:
            y_pred = self.model.predict(X)
        except (NotFittedError, RuntimeError):
            return None

        self.cumulative_accuracy.update(y, y_pred)
        self.rolling_accuracy.update(y, y_pred)

        for metric in self.metrics.values():
            if isinstance(metric, StreamingAUC):
                proba = self.model.predict_proba(X)
                pos_col = np.flatnonzero(self.tree.classes_ == metric.pos_label)
                if len(pos_col) == 0:
                    # The positive class has not appeared yet, so its score is 0.
                    metric.update(y, np.zeros(len(y)))
                else:
                    metric.update(y, proba[:, pos_col[0]])
            else:
                metric.update(y, y_pred)

        self.last_y_true = np.asarray(y)
        self.last_y_pred = y_pred
        return y_pred

    def fit_chunk(self, X, y):
        """partial_fit on the chunk; returns True if it caused a retrain."""

        retrains_before = getattr(self.tree, "n_retrains_", 0)
        self.model.partial_fit(X, y)
        return getattr(self.tree, "n_retrains_", 0) > retrains_before

    def step(self, X, y):
        """Score, then train on one chunk, and add a row to history."""

        y_pred = self.score_chunk(X, y)
        retrained = self.fit_chunk(X, y)

        h = self.history
        h["chunk"].append(len(h["chunk"]))
        h["n_samples"].append(len(y))
        h["scored"].append(y_pred is not None)

        if y_pred is None:
            h["chunk_accuracy"].append(np.nan)
            h["cumulative_accuracy"].append(np.nan)
            h["rolling_accuracy"].append(np.nan)
            for name in self.metrics:
                h[name].append(np.nan)
        else:
            h["chunk_accuracy"].append(float(np.mean(y_pred == np.asarray(y))))
            h["cumulative_accuracy"].append(self.cumulative_accuracy.result())
            h["rolling_accuracy"].append(self.rolling_accuracy.result())
            for name, metric in self.metrics.items():
                h[name].append(metric.result())

        h["retrained"].append(retrained)
        h["retrain_time"].append(self.tree.last_retrain_time_ if retrained else 0.0)
        h["buffer_bytes"].append(getattr(self.tree, "buffer_nbytes", 0))
        h["n_nodes"].append(getattr(self.tree, "n_nodes_", 0))
        return self

    def run(self, chunks):
        """Run step() on every (X, y) chunk and return history."""

        for X, y in chunks:
            self.step(X, y)
        return self.history

    def log_table(self, every=1):
        """history as a text table, every `every`-th chunk."""

        h = self.history
        lines = [
            f"{'chunk':>5} {'acc':>6} {'cum_acc':>8} {'roll_acc':>9} "
            f"{'retrain':>8} {'time_ms':>8} {'buf_KB':>7} {'nodes':>6}"
        ]
        for i in range(0, len(h["chunk"]), every):
            if h["scored"][i]:
                accs = (
                    f"{h['chunk_accuracy'][i]:>6.3f} "
                    f"{h['cumulative_accuracy'][i]:>8.3f} "
                    f"{h['rolling_accuracy'][i]:>9.3f}"
                )
            else:
                accs = f"{'N/A':>6} {'N/A':>8} {'N/A':>9}"
            lines.append(
                f"{h['chunk'][i]:>5} {accs} "
                f"{'yes' if h['retrained'][i] else 'no':>8} "
                f"{h['retrain_time'][i] * 1000:>8.2f} "
                f"{h['buffer_bytes'][i] / 1024:>7.1f} {h['n_nodes'][i]:>6}"
            )
        return "\n".join(lines)
