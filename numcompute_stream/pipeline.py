"""
pipeline.py

Streaming-compatible pipeline that chains preprocessors and a final
estimator together. The key addition over the base NumCompute pipeline
is partial_fit support so the whole chain can be updated chunk by chunk.
"""

import numpy as np


class Pipeline:
    """
    Chains a sequence of transformers followed by a final estimator.

    Each transformer must implement partial_fit and transform. Streaming updates
    freeze preprocessing after the first chunk so buffered examples and new
    observations remain in the same feature space.
    The final step must implement partial_fit and predict.
    All intermediate steps are fitted and transformed in order,
    with the output of each step feeding into the next.

    Parameters
    ----------
    steps : list of (name, object) tuples
        Transformers in order, with the last step being the estimator.
        Names must be unique strings.

    Examples
    --------
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('model', RandomForestClassifier())
    ])
    pipe.partial_fit(X_chunk, y_chunk)
    preds = pipe.predict(X_test)
    """

    def __init__(self, steps):
        if len(steps) == 0:
            raise ValueError("steps cannot be empty")
        names = [n for n, _ in steps]
        if len(names) != len(set(names)):
            raise ValueError("step names must be unique")
        self.steps = steps
        self.fitted_ = False

    @property
    def _transformers(self):
        """All steps except the last one."""
        return self.steps[:-1]

    @property
    def _estimator(self):
        """The final step, expected to be a classifier or model."""
        return self.steps[-1][1]

    def partial_fit(self, X, y, classes=None):
        """
        Update the estimator with a new chunk of data.

        Fit each transformer on the first chunk, then keep its fitted state
        fixed across later chunks. An explicit full fit refreshes preprocessing.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        classes : array-like or None
            Passed to the final estimator if it accepts it.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        final = self._estimator
        if not hasattr(final, "partial_fit"):
            raise TypeError("final estimator does not have partial_fit")
        for name, step in self._transformers:
            if not self.fitted_:
                if not hasattr(step, "partial_fit"):
                    raise TypeError(f"step '{name}' does not have partial_fit")
                step.partial_fit(X)
            X = step.transform(X)
        if classes is not None:
            final.partial_fit(X, y, classes=classes)
        else:
            final.partial_fit(X, y)
        self.fitted_ = True
        return self

    def fit(self, X, y, classes=None):
        """
        Full fit on a single dataset.

        Fits each transformer on the full X then passes transformed
        output to the next step. Useful when you have all data upfront.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        for name, step in self._transformers:
            if hasattr(step, "fit"):
                step.fit(X)
            else:
                step.partial_fit(X)
            X = step.transform(X)
        final = self._estimator
        if classes is not None:
            final.fit(X, y, classes=classes)
        else:
            final.fit(X, y)
        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Pass X through all transformers only (not the final estimator).

        Useful if you want to inspect the preprocessed features.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
        """
        if not self.fitted_:
            raise RuntimeError("call partial_fit before transform")
        X = np.asarray(X, dtype=float)
        for name, step in self._transformers:
            X = step.transform(X)
        return X

    def predict(self, X):
        """
        Transform X through all steps then predict with the estimator.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)

        Raises
        ------
        RuntimeError
            If called before partial_fit.
        """
        if not self.fitted_:
            raise RuntimeError("call partial_fit before predict")
        X = np.asarray(X, dtype=float)
        for name, step in self._transformers:
            X = step.transform(X)
        return self._estimator.predict(X)

    def predict_proba(self, X):
        """
        Transform X then get class probabilities from the estimator.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
        """
        if not self.fitted_:
            raise RuntimeError("call partial_fit before predict_proba")
        X = np.asarray(X, dtype=float)
        for name, step in self._transformers:
            X = step.transform(X)
        return self._estimator.predict_proba(X)

    def score(self, X, y):
        """
        Accuracy of the pipeline on X and y.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)

        Returns
        -------
        float
        """
        y = np.asarray(y)
        preds = self.predict(X)
        return float(np.mean(preds == y))

    def get_step(self, name):
        """
        Retrieve a step object by name.

        Parameters
        ----------
        name : str

        Returns
        -------
        object

        Raises
        ------
        KeyError
            If name is not found in steps.
        """
        for n, step in self.steps:
            if n == name:
                return step
        raise KeyError(f"no step named '{name}'")
