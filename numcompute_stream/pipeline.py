"""
Pipeline for chaining transformers and a final model.

2.1 fix: fit and partial_fit pass y on to every step, so a supervised
model at the end can be trained (2.1 called last_step.fit(X) without y).
"""


class Pipeline:
    """
    Runs (name, step) pairs in order. All steps but the last need fit and
    transform; the last needs fit and transform or predict.
    """

    def __init__(self, steps):
        """
        Raises TypeError/ValueError if steps are not unique (name, step) pairs,
        AttributeError if a step is missing a required method.
        """

        self.steps = steps
        self.fitted = False
        self._validate_steps()

    def _validate_steps(self):
        """Check the structure of steps and the methods of each step."""

        names = []
        for item in self.steps:
            if not isinstance(item, (tuple, list)):
                raise TypeError(
                    f"Expected a tuple or a list pair of (name, step) "
                    f"but got: {type(item).__name__}."
                )

            if len(item) != 2:
                raise ValueError(
                    f"Expected a pair of 2 elements, but got "
                    f"a sequence of length: {len(item)}."
                )

            name, step = item

            if not isinstance(name, str):
                raise TypeError(
                    f"Expected name to be str, but got: {type(name).__name__}."
                )

            names.append(name)

        if len(self.steps) == 0:
            raise ValueError("Pipeline steps cannot be empty")

        if not len(names) == len(set(names)):
            raise ValueError(
                "Step names are expected to be unique, but duplicates were found."
            )

        for name, step in self.steps[:-1]:
            if not callable(getattr(step, "fit", None)):
                raise AttributeError(f"Step {name!r} missing callable fit() method.")

            if not callable(getattr(step, "transform", None)):
                raise AttributeError(
                    f"Step {name!r} missing callable transform() method."
                )

        name, last_step = self.steps[-1]

        if not callable(getattr(last_step, "fit", None)):
            raise AttributeError(f"Last step {name!r} missing callable fit() method.")

        if not callable(getattr(last_step, "transform", None)) and not callable(
            getattr(last_step, "predict", None)
        ):
            raise AttributeError(
                f"Last step {name!r} missing callable transform() or predict() method."
            )

    @property
    def named_steps(self):
        """Steps by name, e.g. pipe.named_steps["scaler"]."""
        return dict(self.steps)

    @property
    def final_estimator(self):
        """The last step."""
        return self.steps[-1][1]

    def fit(self, X, y=None):
        """Fit and transform each step in turn, then fit the last step with y."""

        curr_X = X

        for _, step in self.steps[:-1]:
            step.fit(curr_X, y)
            curr_X = step.transform(curr_X)

        self.final_estimator.fit(curr_X, y)

        self.fitted = True

        return self

    def partial_fit(self, X, y=None):
        """
        Update every step with one chunk: partial_fit then transform for each
        transformer, then partial_fit(X, y) on the last step.

        Raises AttributeError if a step has no partial_fit.
        """

        for name, step in self.steps:
            if not callable(getattr(step, "partial_fit", None)):
                raise AttributeError(
                    f"Step {name!r} has no partial_fit() method, so the "
                    "pipeline cannot be trained on a stream."
                )

        curr_X = X

        for _, step in self.steps[:-1]:
            step.partial_fit(curr_X, y)
            curr_X = step.transform(curr_X)

        self.final_estimator.partial_fit(curr_X, y)

        self.fitted = True

        return self

    def transform(self, X):
        """
        Transform X through every step. Raises RuntimeError before fit, and
        AttributeError if the last step can't transform.
        """

        if not self.fitted:
            raise RuntimeError("Pipeline has not yet been fitted.")

        _, last_step = self.steps[-1]

        if not callable(getattr(last_step, "transform", None)):
            raise AttributeError("Last step missing callable transform() method.")

        curr_X = X

        for _, step in self.steps:
            curr_X = step.transform(curr_X)

        return curr_X

    def fit_transform(self, X, y=None):
        """Same as fit(X, y).transform(X)."""
        _, last_step = self.steps[-1]

        if not callable(getattr(last_step, "transform", None)):
            raise AttributeError("Last step missing callable transform() method.")

        return self.fit(X, y).transform(X)

    def predict(self, X):
        """Transform X, then call the last step's predict."""

        if not callable(getattr(self.final_estimator, "predict", None)):
            raise AttributeError("Last step missing callable predict() method.")

        return self.final_estimator.predict(self._transform_before_last(X))

    def _transform_before_last(self, X):
        """Run X through every step except the last."""

        if not self.fitted:
            raise RuntimeError("Pipeline has not yet been fitted.")

        curr_X = X
        for _, step in self.steps[:-1]:
            curr_X = step.transform(curr_X)
        return curr_X

    def predict_proba(self, X):
        """Transform X, then call the last step's predict_proba."""

        if not callable(getattr(self.final_estimator, "predict_proba", None)):
            raise AttributeError("Last step missing callable predict_proba() method.")

        return self.final_estimator.predict_proba(self._transform_before_last(X))
