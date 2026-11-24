import numpy as np


class Compose:
    """
    Chain multiple transformers together, applying them in sequence.

    Each step's output becomes the next step's input, similar to
    a Unix pipe. All steps must implement fit() and transform().

    Parameters
    ----------
    steps : list
        List of transformer objects, applied left to right.

    Complexity
    ----------
    Time  : O(sum of each step's complexity)
    Space : O(n * d) for intermediate arrays
    """

    def __init__(self, steps):
        if len(steps) == 0:
            raise ValueError("steps cannot be empty")
        for i, step in enumerate(steps):
            if not hasattr(step, "fit") or not hasattr(step, "transform"):
                raise TypeError(f"Step {i} must have fit and transform methods")
        self.steps = steps

    def fit(self, X):
        """
        Fit each step in order, passing transformed output to the next.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        data = X
        # fit each transformer on the output of the previous one
        for step in self.steps:
            step.fit(data)
            data = step.transform(data)
        return self

    def transform(self, X):
        """
        Apply each fitted transformer in order.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Transformed output after all steps.
        """
        data = X
        for step in self.steps:
            data = step.transform(data)
        return data

    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class FeatureUnion:
    """
    Run multiple transformers in parallel and concatenate their outputs.

    Each transformer sees the same input X. Their outputs are joined
    column-wise, so the result has more features than any single transformer.

    Parameters
    ----------
    transformer_list : list
        List of transformer objects to run in parallel.

    Complexity
    ----------
    Time  : O(sum of each transformer's complexity)
    Space : O(n * total_output_features)
    """

    def __init__(self, transformer_list):
        if len(transformer_list) == 0:
            raise ValueError("transformer_list cannot be empty")
        for i, t in enumerate(transformer_list):
            if not hasattr(t, "fit") or not hasattr(t, "transform"):
                raise TypeError(f"Transformer {i} must have fit and transform methods")
        self.transformer_list = transformer_list

    def fit(self, X):
        """
        Fit each transformer independently on the same X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        # each transformer fits on the original input independently
        for t in self.transformer_list:
            t.fit(X)
        return self

    def transform(self, X):
        """
        Transform X with each transformer and concatenate results column-wise.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, sum_of_output_features)
            Horizontally stacked outputs from all transformers.
        """
        parts = [t.transform(X) for t in self.transformer_list]
        return np.concatenate(parts, axis=1)

    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)