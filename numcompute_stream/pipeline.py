"""Pipeline for chaining transformers and a final estimator."""

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
                    f"Expected name to be str, but got: "
                    f"{type(name).__name__}."
            )

            names.append(name)
        
        if len(self.steps) == 0:
            raise ValueError("Pipeline steps cannot be empty")
        

        if not len(names) == len(set(names)):
            raise ValueError(
                "Step names are expected to be unique, but "
                f"duplicates were found."
            )

        for name, step in self.steps[:-1]:
            if not callable(getattr(step, "fit", None)):
                raise AttributeError(
                    f"Step {name!r} missing callable fit() method."
                )

            if not callable(getattr(step, "transform", None)):
                raise AttributeError(
                    f"Step {name!r} missing callable transform() method."
                )

        name, last_step = self.steps[-1]

        if not callable(getattr(last_step, "fit", None)):
            raise AttributeError(
                f"Last step {name!r} missing callable fit() method."
            )

        if (
            not callable(getattr(last_step, "transform", None)) 
            and not callable(getattr(last_step, "predict", None))
        ):
            raise AttributeError(
                f"Last step {name!r} missing callable transform()"
                " or predict() method."
            )


    def fit(self, X):
        """Fit and transform each step in turn, then fit the last step."""

        curr_X = X

        for _, step in self.steps[:-1]:
            step.fit(curr_X)
            curr_X = step.transform(curr_X)

        _, last_step = self.steps[-1]
        
        last_step.fit(curr_X)

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
            raise AttributeError(
                "Last step missing callable transform() method."
            )

        curr_X = X

        for _, step in self.steps:
            curr_X = step.transform(curr_X)

        return curr_X

    def fit_transform(self, X):
        """Same as fit(X).transform(X)."""
        _, last_step = self.steps[-1]     

        if not callable(getattr(last_step, "transform", None)):
            raise AttributeError(
                "Last step missing callable transform() method."
            )

        return self.fit(X).transform(X)

    def predict(self, X):
        """
        Transform X, then call the last step's predict. Raises RuntimeError
        before fit, AttributeError if the last step has no predict.
        """
        
        if not self.fitted:
            raise RuntimeError("Pipeline has not yet been fitted.")

        _, last_step = self.steps[-1]     

        if not callable(getattr(last_step, "predict", None)):
            raise AttributeError(
                f"Last step missing callable predict() method."
            )

        curr_X = X

        for _, step in self.steps[:-1]:
            curr_X = step.transform(curr_X)

        return last_step.predict(curr_X)