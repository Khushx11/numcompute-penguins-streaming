import numpy as np


class StandardScaler:
    """
    Standardise features by removing the mean and scaling to unit variance.

    Each feature (column) is transformed independently using statistics
    computed from the training data passed to fit().

    Parameters
    ----------
    None

    Attributes
    ----------
    mean_ : np.ndarray, shape (n_features,)
        Per-feature mean computed during fit.
    std_ : np.ndarray, shape (n_features,)
        Per-feature standard deviation computed during fit.
        Zero-std features are set to 1.0 to avoid division by zero.
    fitted_ : bool
        True after fit() has been called.

    Complexity
    ----------
    Time  : O(n * d) for fit and transform
    Space : O(d) for stored statistics
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.fitted_ = False

    def fit(self, X):
        """
        Compute mean and std from training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If X is not 2-D.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        # avoid division by zero for constant features
        self.std_[self.std_ == 0] = 1.0
        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Apply standardisation using the fitted mean and std.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_features)

        Raises
        ------
        RuntimeError
            If called before fit().
        """
        if not self.fitted_:
            raise RuntimeError("Call fit before transform")
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """
        Reverse standardisation back to the original scale.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_features)

        Raises
        ------
        RuntimeError
            If called before fit().
        """
        if not self.fitted_:
            raise RuntimeError("Call fit before inverse_transform")
        return np.asarray(X, dtype=float) * self.std_ + self.mean_


class MinMaxScaler:
    """
    Scale features to a fixed range, default [0, 1].

    Parameters
    ----------
    feature_range : tuple (min, max), optional
        Target range for the transformed data. Default is (0, 1).

    Attributes
    ----------
    min_ : np.ndarray, shape (n_features,)
        Per-feature minimum seen during fit.
    max_ : np.ndarray, shape (n_features,)
        Per-feature maximum seen during fit.
    fitted_ : bool
        True after fit() has been called.

    Complexity
    ----------
    Time  : O(n * d) for fit and transform
    Space : O(d) for stored statistics
    """

    def __init__(self, feature_range=(0, 1)):
        lo, hi = feature_range
        if lo >= hi:
            raise ValueError(f"feature_range min ({lo}) must be less than max ({hi})")
        self.lo = lo
        self.hi = hi
        self.min_ = None
        self.max_ = None
        self.fitted_ = False

    def fit(self, X):
        """
        Compute per-feature min and max from training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If X is not 2-D.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Scale X into the target feature range.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_features)

        Raises
        ------
        RuntimeError
            If called before fit().
        """
        if not self.fitted_:
            raise RuntimeError("Call fit before transform")
        X = np.asarray(X, dtype=float)
        scale = self.max_ - self.min_
        # constant features would cause division by zero - treat range as 1
        scale[scale == 0] = 1.0
        return (X - self.min_) / scale * (self.hi - self.lo) + self.lo

    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """
        Reverse scaling back to the original range.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_features)

        Raises
        ------
        RuntimeError
            If called before fit().
        """
        if not self.fitted_:
            raise RuntimeError("Call fit before inverse_transform")
        X = np.asarray(X, dtype=float)
        scale = self.max_ - self.min_
        scale[scale == 0] = 1.0
        return (X - self.lo) / (self.hi - self.lo) * scale + self.min_


class Imputer:
    """
    Fill missing (NaN) values using a chosen strategy.

    Parameters
    ----------
    strategy : {'mean', 'median', 'constant'}, optional
        Imputation strategy. Default is 'mean'.

    Attributes
    ----------
    fill_values_ : np.ndarray, shape (n_features,)
        The value used to replace NaNs in each column.
    fitted_ : bool
        True after fit() has been called.

    Complexity
    ----------
    Time  : O(n * d) for fit and transform
    Space : O(d) for stored fill values
    """

    def __init__(self, strategy="mean"):
        if strategy not in ("mean", "median", "constant"):
            raise ValueError(f"strategy must be 'mean', 'median', or 'constant'")
        self.strategy = strategy
        self.fill_values_ = None
        self.fitted_ = False

    def fit(self, X, constant=0.0):
        """
        Compute fill values from training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        constant : float, optional
            Fill value used when strategy='constant'. Default is 0.0.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If X is not 2-D.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if self.strategy == "mean":
            self.fill_values_ = np.nanmean(X, axis=0)
        elif self.strategy == "median":
            self.fill_values_ = np.nanmedian(X, axis=0)
        elif self.strategy == "constant":
            self.fill_values_ = np.full(X.shape[1], constant)
        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Replace NaNs with the fitted fill values.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_features)
            Copy of X with NaNs replaced.

        Raises
        ------
        RuntimeError
            If called before fit().
        """
        if not self.fitted_:
            raise RuntimeError("Call fit before transform")
        X = np.asarray(X, dtype=float).copy()
        # vectorised fill - find all NaN positions at once then scatter fill values
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(self.fill_values_, np.where(nan_mask)[1])
        return X

    def fit_transform(self, X, constant=0.0):
        """Fit and transform in one step."""
        return self.fit(X, constant=constant).transform(X)


class OneHotEncoder:
    """
    Convert integer class labels into one-hot encoded rows.

    Parameters
    ----------
    None

    Attributes
    ----------
    categories_ : np.ndarray
        Unique sorted categories learned during fit.
    n_categories_ : int
        Number of unique categories.
    fitted_ : bool
        True after fit() has been called.

    Complexity
    ----------
    Time  : O(n * c) where c is the number of categories
    Space : O(n * c) for the output matrix
    """

    def __init__(self):
        self.categories_ = None
        self.n_categories_ = None
        self.fitted_ = False

    def fit(self, y):
        """
        Learn the unique categories from label array y.

        Parameters
        ----------
        y : array-like, shape (n_samples,)
            Integer class labels.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If y is not 1-D.
        """
        y = np.asarray(y, dtype=int)
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}")
        self.categories_ = np.unique(y)
        self.n_categories_ = len(self.categories_)
        self.fitted_ = True
        return self

    def transform(self, y):
        """
        Convert labels to a one-hot matrix.

        Parameters
        ----------
        y : array-like, shape (n_samples,)

        Returns
        -------
        np.ndarray, shape (n_samples, n_categories)
            Binary matrix with exactly one 1 per row.

        Raises
        ------
        RuntimeError
            If called before fit().
        """
        if not self.fitted_:
            raise RuntimeError("Call fit before transform")
        y = np.asarray(y, dtype=int)
        out = np.zeros((len(y), self.n_categories_), dtype=float)
        # searchsorted gives the column index for each label in one shot
        indices = np.searchsorted(self.categories_, y)
        out[np.arange(len(y)), indices] = 1.0
        return out

    def fit_transform(self, y):
        """Fit and transform in one step."""
        return self.fit(y).transform(y)

    def inverse_transform(self, X):
        """
        Convert a one-hot matrix back to a label array.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_categories)

        Returns
        -------
        np.ndarray, shape (n_samples,)

        Raises
        ------
        RuntimeError
            If called before fit().
        """
        if not self.fitted_:
            raise RuntimeError("Call fit before inverse_transform")
        X = np.asarray(X, dtype=float)
        indices = np.argmax(X, axis=1)
        return self.categories_[indices]