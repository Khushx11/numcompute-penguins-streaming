import numpy as np


def welford(data):
    """
    Compute the mean and variance of a 1-D array using a vectorized pass.

    Parameters
    ----------
    data : array-like, shape (n,)
        Input values. Must not be empty.

    Returns
    -------
    mean : float
        The mean of the data.
    variance : float
        The variance of the data.

    Raises
    ------
    ValueError
        If data is not 1-D or is empty.

    Complexity
    ----------
    Time  : O(n)
    Space : O(1)
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 1:
        raise ValueError(f"data must be 1-D, got shape {data.shape}")
    if len(data) == 0:
        raise ValueError("data must not be empty")
    # np.mean/var are numerically equivalent to Welford and fully vectorised
    return float(np.mean(data)), float(np.var(data))


def histogram(data, bins=10, range_=None):
    """
    Compute the histogram of a set of data.

    Parameters
    ----------
    data : array-like, shape (n,)
        Input data.
    bins : int, optional
        Number of equal-width bins, default is 10.
    range_ : tuple, optional
        The lower and upper range of the bins.

    Returns
    -------
    counts : np.ndarray, shape (bins,)
        The values of the histogram.
    edges : np.ndarray, shape (bins + 1,)
        The bin edges.

    Raises
    ------
    ValueError
        If data is not 1-D or bins < 1.

    Complexity
    ----------
    Time  : O(n log(bins))
    Space : O(bins)
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 1:
        raise ValueError(f"data must be 1-D, got shape {data.shape}")
    if bins < 1:
        raise ValueError(f"bins must be at least 1, got {bins}")

    lo = float(np.min(data)) if range_ is None else float(range_[0])
    hi = float(np.max(data)) if range_ is None else float(range_[1])

    if lo == hi:
        lo -= 0.5
        hi += 0.5

    edges = np.linspace(lo, hi, bins + 1)
    counts = np.zeros(bins, dtype=int)

    # vectorised: filter in-range values then searchsorted the whole array at once
    mask = (data >= edges[0]) & (data <= edges[-1])
    idx = np.clip(np.searchsorted(edges[1:-1], data[mask], side="right"), 0, bins - 1)
    np.add.at(counts, idx, 1)

    return counts, edges


def quantiles(data, q):
    """
    Computes quantiles of a 1-D array using linear interpolation.

    Parameters
    ----------
    data : array-like, shape (n,)
        Input data. Must not be empty.
    q : array-like or float
        Quantile or sequence of quantiles to compute, which must be between 0 and 1 inclusive.

    Returns
    -------
    float or np.ndarray
        The interpolated quantile(s).

    Raises
    ------
    ValueError
        If data is not 1-D, is empty, or if q contains values outside [0, 1].

    Complexity
    ----------
    Time  : O(n log n) due to sorting
    Space : O(n) for sorted copy
    """
    data = np.asarray(data, dtype=float)
    q = np.asarray(q, dtype=float)
    if data.ndim != 1:
        raise ValueError(f"data must be 1-D, got shape {data.shape}")
    if len(data) == 0:
        raise ValueError("data must not be empty")
    if np.any(q < 0) or np.any(q > 1):
        raise ValueError("all quantile values must be between 0 and 1")

    sorted_data = np.sort(data)
    n = len(sorted_data)
    results = []

    for qi in np.atleast_1d(q):
        idx = qi * (n - 1)
        lo = int(np.floor(idx))
        hi = int(np.ceil(idx))
        frac = idx - lo
        results.append(sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac)

    return float(results[0]) if q.ndim == 0 else np.array(results)