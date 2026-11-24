import numpy as np


def rank_data(arr, method="average"):
    """
    Assign ranks to values in a 1-D array, with tie handling.

    NaN values are preserved in their original positions and given NaN rank.

    Parameters
    ----------
    arr : array-like, shape (n,)
        Input values to rank.
    method : {'average', 'min', 'max', 'ordinal'}, optional
        How to handle tied values. Default is 'average'.
        - 'average' : tied elements share the mean of their ranks
        - 'min'     : tied elements all get the lowest rank in the group
        - 'max'     : tied elements all get the highest rank in the group
        - 'ordinal' : tied elements get distinct ranks by order of appearance

    Returns
    -------
    np.ndarray, shape (n,)
        Rank of each element starting from 1. NaN inputs get NaN rank.

    Raises
    ------
    ValueError
        If arr is not 1-D or method is unrecognised.

    Complexity
    ----------
    Time  : O(n log n) due to argsort
    Space : O(n)
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"arr must be 1-D, got shape {arr.shape}")
    if method not in ("average", "min", "max", "ordinal"):
        raise ValueError(f"method must be 'average', 'min', 'max', or 'ordinal'")

    # NaN comparisons always return False so argsort silently treats NaN as
    # the largest value - instead we rank only the valid entries and propagate
    # NaN back into the original positions
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        ranks = np.full(len(arr), np.nan)
        valid_ranks = rank_data(arr[~nan_mask], method=method)
        ranks[~nan_mask] = valid_ranks
        return ranks

    n = len(arr)
    order = np.argsort(arr, kind="stable")
    ranks = np.empty(n, dtype=float)

    # ordinal just assigns 1..n in sorted order - no tie logic needed
    if method == "ordinal":
        ranks[order] = np.arange(1, n + 1, dtype=float)
        return ranks

    # walk through sorted positions, grouping tied values together
    i = 0
    while i < n:
        j = i + 1
        while j < n and arr[order[j]] == arr[order[i]]:
            j += 1

        group = np.arange(i + 1, j + 1, dtype=float)

        if method == "average":
            val = group.mean()
        elif method == "min":
            val = group.min()
        elif method == "max":
            val = group.max()

        ranks[order[i:j]] = val
        i = j

    return ranks


def percentile_rank(arr, score):
    """
    Return the percentage of values in arr that are strictly below score.

    Parameters
    ----------
    arr : array-like, shape (n,)
        Reference distribution. Must be non-empty.
    score : float
        The value to compare against.

    Returns
    -------
    float
        Percentage of arr values strictly less than score. Range: [0.0, 100.0].

    Raises
    ------
    ValueError
        If arr is not 1-D or is empty.

    Complexity
    ----------
    Time  : O(n)
    Space : O(1)
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"arr must be 1-D, got shape {arr.shape}")
    if len(arr) == 0:
        raise ValueError("arr must not be empty")

    # count how many values fall strictly below the score
    return float(np.sum(arr < score) / len(arr) * 100)
