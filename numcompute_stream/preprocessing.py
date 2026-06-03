"""
preprocessing.py

Streaming-compatible versions of the scalers and encoders from the
base NumCompute package. The key difference is partial_fit() support
so these can be updated incrementally as new chunks of data arrive
instead of needing the full dataset up front.
"""

import numpy as np
from numcompute_stream.stats import RunningStats


class StandardScaler:
    """
    Standardises features to zero mean and unit variance, but supports
    partial_fit() so it can update its statistics chunk by chunk.

    Uses Welford's algorithm internally via RunningStats so the mean
    and variance stay accurate without storing past data.

    Attributes
    ----------
    trackers : list of RunningStats or None
        One per feature column, created on first partial_fit call.
    n_features : int or None
    fitted_ : bool
        True once at least one chunk has been fitted.
    """

    def __init__(self):
        self.trackers = None
        self.n_features = None
        self.fitted_ = False

    def partial_fit(self, X):
        """
        Update mean and variance estimates with a new chunk of data.

        Safe to call multiple times. Each call refines the running stats
        rather than replacing them.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If X is not 2-D or feature count changes between chunks.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if self.trackers is None:
            self.n_features = X.shape[1]
            self.trackers = [RunningStats() for _ in range(self.n_features)]
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"expected {self.n_features} features, got {X.shape[1]}"
            )
        for i, tracker in enumerate(self.trackers):
            tracker.update_stats(X[:, i])
        self.fitted_ = True
        return self

    def fit(self, X):
        """Full fit on a single batch. Just calls partial_fit."""
        return self.partial_fit(X)

    def transform(self, X):
        """
        Standardise X using the current mean and std estimates.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_features)

        Raises
        ------
        RuntimeError
            If called before any partial_fit.
        """
        if not self.fitted_:
            raise RuntimeError("call partial_fit before transform")
        X = np.asarray(X, dtype=float)
        mean = np.array([t.mean_ for t in self.trackers])
        std = np.array([t.std() for t in self.trackers])
        # zero-std features stay as-is rather than producing NaN
        std[std == 0] = 1.0
        return (X - mean) / std

    def fit_transform(self, X):
        """Fit on X then transform it in one go."""
        return self.partial_fit(X).transform(X)

    def inverse_transform(self, X):
        """
        Reverse the standardisation back to original scale.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
        """
        if not self.fitted_:
            raise RuntimeError("call partial_fit before inverse_transform")
        mean = np.array([t.mean_ for t in self.trackers])
        std = np.array([t.std() for t in self.trackers])
        std[std == 0] = 1.0
        return np.asarray(X, dtype=float) * std + mean


class MinMaxScaler:
    """
    Scales each feature to a fixed range (default 0 to 1).

    Supports partial_fit() by tracking the running min and max across
    chunks. Min and max are just updated if a new chunk extends the range.

    Parameters
    ----------
    feature_range : tuple (lo, hi)
        Target output range. Default is (0, 1).
    """

    def __init__(self, feature_range=(0, 1)):
        lo, hi = feature_range
        if lo >= hi:
            raise ValueError(f"feature_range min ({lo}) must be less than max ({hi})")
        self.lo = lo
        self.hi = hi
        self.min_ = None
        self.max_ = None
        self.n_features = None
        self.fitted_ = False

    def partial_fit(self, X):
        """
        Update running min and max with a new data chunk.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        chunk_min = np.nanmin(X, axis=0)
        chunk_max = np.nanmax(X, axis=0)
        if self.min_ is None:
            self.n_features = X.shape[1]
            self.min_ = chunk_min
            self.max_ = chunk_max
        else:
            if X.shape[1] != self.n_features:
                raise ValueError(
                    f"expected {self.n_features} features, got {X.shape[1]}"
                )
            # expand range if this chunk goes outside what we have seen
            self.min_ = np.minimum(self.min_, chunk_min)
            self.max_ = np.maximum(self.max_, chunk_max)
        self.fitted_ = True
        return self

    def fit(self, X):
        """Full fit on a single batch."""
        return self.partial_fit(X)

    def transform(self, X):
        """
        Scale X into the target feature range.

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
        scale = self.max_ - self.min_
        # avoid division by zero for constant features
        scale[scale == 0] = 1.0
        return (X - self.min_) / scale * (self.hi - self.lo) + self.lo

    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.partial_fit(X).transform(X)

    def inverse_transform(self, X):
        """Reverse the scaling back to original range."""
        if not self.fitted_:
            raise RuntimeError("call partial_fit before inverse_transform")
        X = np.asarray(X, dtype=float)
        scale = self.max_ - self.min_
        scale[scale == 0] = 1.0
        return (X - self.lo) / (self.hi - self.lo) * scale + self.min_


class Imputer:
    """
    Fills NaN values using a running mean estimate that updates with
    each new chunk of data.

    Only 'mean' strategy is supported in streaming mode since median
    and constant do not have natural incremental updates. Constant fill
    is also supported since it never changes.

    Parameters
    ----------
    strategy : str
        'mean' or 'constant'. Default is 'mean'.
    """

    def __init__(self, strategy="mean"):
        if strategy not in ("mean", "constant"):
            raise ValueError("streaming Imputer only supports 'mean' or 'constant'")
        self.strategy = strategy
        self.trackers = None
        self.n_features = None
        self.constant_val = 0.0
        self.fitted_ = False

    def partial_fit(self, X, constant=0.0):
        """
        Update fill value estimates from a new chunk.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        constant : float
            Only used when strategy='constant'.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if self.strategy == "mean":
            if self.trackers is None:
                self.n_features = X.shape[1]
                self.trackers = [RunningStats() for _ in range(self.n_features)]
            for i, tracker in enumerate(self.trackers):
                col = X[:, i]
                tracker.update_stats(col[~np.isnan(col)])
        elif self.strategy == "constant":
            self.constant_val = constant
            self.n_features = X.shape[1]
        self.fitted_ = True
        return self

    def fit(self, X, constant=0.0):
        """Full fit on a single batch."""
        return self.partial_fit(X, constant=constant)

    def transform(self, X):
        """
        Replace NaNs with the current fill values.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray with NaNs replaced
        """
        if not self.fitted_:
            raise RuntimeError("call partial_fit before transform")
        X = np.asarray(X, dtype=float).copy()
        nan_mask = np.isnan(X)
        if not nan_mask.any():
            return X
        if self.strategy == "mean":
            fill = np.array([t.mean_ for t in self.trackers])
        else:
            fill = np.full(self.n_features, self.constant_val)
        # scatter fill values into NaN positions all at once
        X[nan_mask] = np.take(fill, np.where(nan_mask)[1])
        return X

    def fit_transform(self, X, constant=0.0):
        """Fit and transform in one step."""
        return self.partial_fit(X, constant=constant).transform(X)


class OneHotEncoder:
    """
    Encodes integer class labels as one-hot rows.

    Supports partial_fit() so new categories seen in later chunks
    get added to the known set. The output width grows if new
    categories appear.

    Attributes
    ----------
    categories_ : np.ndarray
        Sorted array of all unique categories seen so far.
    fitted_ : bool
    """

    def __init__(self):
        self.categories_ = None
        self.fitted_ = False

    def partial_fit(self, y):
        """
        Expand the known category set with labels from a new chunk.

        Parameters
        ----------
        y : array-like, shape (n_samples,)

        Returns
        -------
        self
        """
        y = np.asarray(y, dtype=int)
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}")
        new_cats = np.unique(y)
        if self.categories_ is None:
            self.categories_ = new_cats
        else:
            # merge and deduplicate categories across chunks
            self.categories_ = np.unique(np.concatenate([self.categories_, new_cats]))
        self.fitted_ = True
        return self

    def fit(self, y):
        """Full fit on a single batch."""
        return self.partial_fit(y)

    def transform(self, y):
        """
        Convert labels to a one-hot matrix.

        Parameters
        ----------
        y : array-like, shape (n_samples,)

        Returns
        -------
        np.ndarray, shape (n_samples, n_categories)
        """
        if not self.fitted_:
            raise RuntimeError("call partial_fit before transform")
        y = np.asarray(y, dtype=int)
        n_cats = len(self.categories_)
        out = np.zeros((len(y), n_cats), dtype=float)
        indices = np.searchsorted(self.categories_, y)
        # clamp to avoid index errors if an unseen label slips through
        indices = np.clip(indices, 0, n_cats - 1)
        out[np.arange(len(y)), indices] = 1.0
        return out

    def fit_transform(self, y):
        """Fit and transform in one step."""
        return self.partial_fit(y).transform(y)

    def inverse_transform(self, X):
        """
        Convert a one-hot matrix back to integer labels.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_categories)

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        if not self.fitted_:
            raise RuntimeError("call partial_fit before inverse_transform")
        return self.categories_[np.argmax(np.asarray(X, dtype=float), axis=1)]


class EMAScaler:
    """
    Standardises features using exponential moving average statistics
    instead of a full Welford running mean.

    Better than StandardScaler for non-stationary streams where recent
    data should count more than data from many chunks ago. The tradeoff
    is that the statistics forget old data, so the scale can shift over
    time.

    Parameters
    ----------
    alpha : float, default 0.1
        EMA smoothing factor. Higher means faster adaptation to new data.
    """

    def __init__(self, alpha=0.1):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be between 0 and 1 exclusive of 0")
        self.alpha = alpha
        self.mean_ = None
        self.var_ = None
        self.n_features = None
        self.fitted_ = False

    def partial_fit(self, X):
        """
        Update EMA mean and variance estimates with a new chunk.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")

        chunk_mean = np.nanmean(X, axis=0)
        chunk_var = np.nanvar(X, axis=0)
        # single-row chunks have no meaningful variance
        chunk_var = np.where(X.shape[0] > 1, chunk_var, np.zeros_like(chunk_var))

        if self.mean_ is None:
            self.n_features = X.shape[1]
            self.mean_ = chunk_mean
            self.var_ = chunk_var
        else:
            if X.shape[1] != self.n_features:
                raise ValueError(
                    f"expected {self.n_features} features, got {X.shape[1]}"
                )
            diff = chunk_mean - self.mean_
            self.mean_ = self.mean_ + self.alpha * diff
            self.var_ = (1 - self.alpha) * (self.var_ + self.alpha * diff ** 2) + self.alpha * chunk_var

        self.fitted_ = True
        return self

    def fit(self, X):
        """Full fit on a single batch."""
        return self.partial_fit(X)

    def transform(self, X):
        """
        Standardise X using the current EMA mean and std.

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
        std = np.sqrt(np.maximum(self.var_, 0.0))
        std[std == 0] = 1.0
        return (X - self.mean_) / std

    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.partial_fit(X).transform(X)
