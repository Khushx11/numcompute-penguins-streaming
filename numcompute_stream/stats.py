"""
stats.py

Incremental statistics for streaming data. The idea is that you never
have all the data at once, so everything here works chunk by chunk.
Running mean/variance uses Welford's algorithm so we never need to
store past values.
"""

import numpy as np


class RunningStats:
    """
    Running mean and variance for a single feature, updated one chunk
    at a time using Welford's online algorithm.

    Just call update_stats() each time new data comes in. No history
    is stored, just the three running values below.

    Attributes
    ----------
    n : int
        How many samples we have seen total.
    mean_ : float
        Current running mean.
    M2 : float
        Sum of squared deviations from the mean so far. Used to get variance.
    """

    def __init__(self):
        self.n = 0
        self.mean_ = 0.0
        self.M2 = 0.0

    def update_stats(self, X_chunk):
        """
        Feed in a new chunk of values and update the running stats.

        NaNs are dropped before processing so they don't corrupt the mean.

        Parameters
        ----------
        X_chunk : array-like, shape (n,)

        Returns
        -------
        self
        """
        X_chunk = np.asarray(X_chunk, dtype=float)
        X_chunk = X_chunk[~np.isnan(X_chunk)]
        if len(X_chunk) == 0:
            return self
        # standard welford update per value
        for x in X_chunk:
            self.n += 1
            delta = x - self.mean_
            self.mean_ += delta / self.n
            delta2 = x - self.mean_
            self.M2 += delta * delta2
        return self

    def variance(self):
        """
        Population variance of everything seen so far.

        Returns 0.0 if we have less than 2 samples since variance
        is undefined for a single point.
        """
        if self.n < 2:
            return 0.0
        return self.M2 / self.n

    def std(self):
        """Standard deviation, just sqrt of variance."""
        return float(np.sqrt(self.variance()))

    def result(self):
        """
        Snapshot of current stats as a dict.

        Returns
        -------
        dict with n, mean, variance, std
        """
        return {
            "n": self.n,
            "mean": self.mean_,
            "variance": self.variance(),
            "std": self.std(),
        }

    def reset(self):
        """Wipe everything back to zero so we can start fresh."""
        self.n = 0
        self.mean_ = 0.0
        self.M2 = 0.0
        return self


class MultiFeatureStats:
    """
    Same as RunningStats but for 2D data with multiple feature columns.

    One RunningStats tracker per column. Feed chunks of shape
    (n_samples, n_features) and it handles the rest.

    Attributes
    ----------
    n_features : int or None
        Figured out from the first chunk.
    trackers : list of RunningStats
        One per feature column.
    """

    def __init__(self):
        self.n_features = None
        self.trackers = []

    def update_stats(self, X_chunk):
        """
        Update all feature trackers with a new chunk.

        Parameters
        ----------
        X_chunk : array-like, shape (n_samples, n_features)

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If the chunk has a different number of features than the first one.
        """
        X_chunk = np.asarray(X_chunk, dtype=float)
        if X_chunk.ndim == 1:
            X_chunk = X_chunk.reshape(-1, 1)
        # set up trackers on first call
        if self.n_features is None:
            self.n_features = X_chunk.shape[1]
            self.trackers = [RunningStats() for _ in range(self.n_features)]
        if X_chunk.shape[1] != self.n_features:
            raise ValueError(
                f"expected {self.n_features} features, got {X_chunk.shape[1]}"
            )
        for i, tracker in enumerate(self.trackers):
            tracker.update_stats(X_chunk[:, i])
        return self

    def means(self):
        """Per-feature means as a numpy array."""
        return np.array([t.mean_ for t in self.trackers])

    def variances(self):
        """Per-feature variances as a numpy array."""
        return np.array([t.variance() for t in self.trackers])

    def stds(self):
        """Per-feature standard deviations as a numpy array."""
        return np.array([t.std() for t in self.trackers])

    def result(self):
        """Summary dict for all features."""
        return {
            "n_features": self.n_features,
            "means": self.means(),
            "variances": self.variances(),
            "stds": self.stds(),
        }

    def reset(self):
        """Reset all feature trackers."""
        for t in self.trackers:
            t.reset()
        return self


class StreamingHistogram:
    """
    A histogram that you can keep updating as new data arrives.

    Bin edges are fixed either from the first chunk or from a manually
    specified range. After that, new chunks just add counts to the
    existing bins.

    Parameters
    ----------
    bins : int
        Number of bins to use.
    range_ : tuple (lo, hi), optional
        If given, fixes the bin edges up front. Otherwise inferred
        from the first chunk of data.
    """

    def __init__(self, bins=10, range_=None):
        if bins < 1:
            raise ValueError("bins must be at least 1")
        self.bins = bins
        self.range_ = range_
        self.counts = None
        self.edges = None

    def update_stats(self, X_chunk):
        """
        Add new data into the histogram counts.

        On the first call, bin edges are set either from range_ or from
        the min/max of the chunk. Subsequent calls just accumulate counts.

        Parameters
        ----------
        X_chunk : array-like, shape (n,)

        Returns
        -------
        self
        """
        X_chunk = np.asarray(X_chunk, dtype=float)
        X_chunk = X_chunk[~np.isnan(X_chunk)]
        if len(X_chunk) == 0:
            return self

        # first chunk sets the edges
        if self.edges is None:
            if self.range_ is not None:
                lo, hi = float(self.range_[0]), float(self.range_[1])
            else:
                lo, hi = float(X_chunk.min()), float(X_chunk.max())
            # handle the case where all values are identical
            if lo == hi:
                lo -= 0.5
                hi += 0.5
            self.edges = np.linspace(lo, hi, self.bins + 1)
            self.counts = np.zeros(self.bins, dtype=int)

        # only count values that fall within the fixed range
        in_range = (X_chunk >= self.edges[0]) & (X_chunk <= self.edges[-1])
        idx = np.clip(
            np.searchsorted(self.edges[1:-1], X_chunk[in_range], side="right"),
            0,
            self.bins - 1,
        )
        np.add.at(self.counts, idx, 1)
        return self

    def result(self):
        """
        Current histogram state.

        Returns
        -------
        dict with 'counts' and 'edges', or None if no data has arrived yet.
        """
        if self.counts is None:
            return None
        return {"counts": self.counts.copy(), "edges": self.edges.copy()}

    def reset(self):
        """Clear counts and edges so the histogram can be reused."""
        self.counts = None
        self.edges = None
        return self


def streaming_quantiles(sorted_buffer, q):
    """
    Compute quantiles from a sorted buffer of all values seen so far.

    This keeps an exact result by storing everything sorted. Fine for
    datasets that fit in memory. For truly huge streams you would want
    a sketch algorithm, but that is out of scope here.

    Parameters
    ----------
    sorted_buffer : array-like, shape (n,)
        All values seen so far, already sorted ascending.
    q : float or array-like
        Quantile(s) to compute, each between 0 and 1.

    Returns
    -------
    float or np.ndarray

    Raises
    ------
    ValueError
        If the buffer is empty or q is outside [0, 1].
    """
    data = np.asarray(sorted_buffer, dtype=float)
    q = np.asarray(q, dtype=float)
    if len(data) == 0:
        raise ValueError("sorted_buffer must not be empty")
    if np.any(q < 0) or np.any(q > 1):
        raise ValueError("all quantile values must be between 0 and 1")

    n = len(data)
    results = []
    for qi in np.atleast_1d(q):
        idx = qi * (n - 1)
        lo = int(np.floor(idx))
        hi = int(np.ceil(idx))
        frac = idx - lo
        # linear interpolation between the two neighbouring values
        results.append(data[lo] * (1 - frac) + data[hi] * frac)

    return float(results[0]) if q.ndim == 0 else np.array(results)
