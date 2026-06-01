import numpy as np


# ── Numerical Stability & Probability ────────────────────────

def log_sum_exp(x, axis=None, keepdims=False):
    """
    Computes log(sum(exp(x))) in a numerically stable way.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    axis : int or None, optional
        Axis over which to sum, default is None.
    keepdims : bool, optional
        Whether to keep the reduced dimensions, default is False.

    Returns
    -------
    np.ndarray
        The computed log-sum-exp.

    Complexity
    ----------
    Time  : O(n)
    Space : O(n)
    """
    x = np.asarray(x, dtype=float)
    max_val = np.max(x, axis=axis, keepdims=True)
    safe_max = np.where(np.isfinite(max_val), max_val, 0.0)  # handle inf edge case
    sum_exp = np.sum(np.exp(x - safe_max), axis=axis, keepdims=True)
    out = np.log(sum_exp) + safe_max
    return np.squeeze(out, axis=axis) if not keepdims else out


def softmax(logits, axis=-1):
    """
    Computes the softmax function in a numerically stable way.

    Parameters
    ----------
    logits : np.ndarray
        Input array.
    axis : int, optional
        Axis over which to compute softmax, default is -1.

    Returns
    -------
    np.ndarray
        Softmax probabilities.

    Complexity
    ----------
    Time  : O(n)
    Space : O(n)
    """
    logits = np.asarray(logits, dtype=float)
    shifted = logits - np.max(logits, axis=axis, keepdims=True)  # max-shift prevents overflow
    exponents = np.exp(shifted)
    return exponents / np.sum(exponents, axis=axis, keepdims=True)


def log_softmax(logits, axis=-1):
    """
    Computes log(softmax(x)) in a numerically stable way.

    Parameters
    ----------
    logits : np.ndarray
        Input array.
    axis : int, optional
        Axis over which to compute, default is -1.

    Returns
    -------
    np.ndarray
        Log-probabilities.

    Complexity
    ----------
    Time  : O(n)
    Space : O(n)
    """
    logits = np.asarray(logits, dtype=float)
    return logits - log_sum_exp(logits, axis=axis, keepdims=True)


# ── Activation Functions ──────────────────────────────────────

def sigmoid(x):
    """
    Computes the sigmoid activation function.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Values squashed into (0, 1).

    Complexity
    ----------
    Time  : O(n)
    Space : O(n)
    """
    x = np.asarray(x, dtype=float)
    # np.where evaluates both branches so we suppress the expected overflow noise
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x))
        )


def relu(x):
    """
    Computes the ReLU activation function.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Negatives clamped to 0.

    Complexity
    ----------
    Time  : O(n)
    Space : O(n)
    """
    return np.maximum(0.0, np.asarray(x, dtype=float))


def leaky_relu(x, alpha=0.01):
    """
    Computes the Leaky ReLU activation function.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    alpha : float, optional
        Slope for the negative side, default is 0.01.

    Returns
    -------
    np.ndarray
        Activation values.

    Complexity
    ----------
    Time  : O(n)
    Space : O(n)
    """
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, x, alpha * x)


# ── Distance Metrics ──────────────────────────────────────────

def get_euclidean_dist(A, B=None):
    # Input:  A (ndarray) shape (m, d), B (ndarray) shape (n, d) or None
    # Output: ndarray shape (m, n) - pairwise L2 distances
    # Raises: ValueError if column counts don't match
    A = np.atleast_2d(A).astype(float)
    B = A if B is None else np.atleast_2d(B).astype(float)
    if A.shape[1] != B.shape[1]:
        raise ValueError("Feature dimension mismatch!")
    # uses ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab to avoid an explicit loop
    sq_norm_A = np.einsum("ij,ij->i", A, A)[:, None]
    sq_norm_B = np.einsum("ij,ij->i", B, B)[None, :]
    dist_sq = np.maximum(0.0, sq_norm_A + sq_norm_B - 2.0 * A @ B.T)  # clip tiny negatives before sqrt
    return np.sqrt(dist_sq)


def get_cosine_sim(A, B=None):
    # Input:  A (ndarray) shape (m, d), B (ndarray) shape (n, d) or None
    # Output: ndarray shape (m, n) - similarity scores in [-1, 1]
    A = np.atleast_2d(A).astype(float)
    B = A if B is None else np.atleast_2d(B).astype(float)
    # normalise rows to unit length so dot product = cos(angle)
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return np.clip(A_norm @ B_norm.T, -1.0, 1.0)  # clip handles float going slightly past ±1


def get_manhattan_dist(A, B=None):
    # Input:  A (ndarray) shape (m, d), B (ndarray) shape (n, d) or None
    # Output: ndarray shape (m, n) - pairwise L1 distances
    A = np.atleast_2d(A).astype(float)
    B = A if B is None else np.atleast_2d(B).astype(float)
    # broadcast to (m, n, d) then sum - no loops needed
    return np.sum(np.abs(A[:, None, :] - B[None, :, :]), axis=-1)


# ── Data Utilities ────────────────────────────────────────────

def iterate_batches(X, y=None, batch_size=32, shuffle=True, seed=None):
    # Input:  X (ndarray) shape (n,...), y (ndarray|None), batch_size (int), seed (int|None)
    # Output: yields X_batch or (X_batch, y_batch) ndarrays
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)  # shuffle index not data to avoid copying
    for start in range(0, n, batch_size):
        sel = idx[start: start + batch_size]
        yield X[sel] if y is None else (X[sel], y[sel])


def to_one_hot(labels, num_classes=None):
    # Input:  labels (ndarray) shape (n,) integer class indices, num_classes (int|None)
    # Output: ndarray shape (n, num_classes) dtype float64
    # Raises: ValueError if any label is negative
    labels = np.asarray(labels, dtype=int)
    if labels.min() < 0:
        raise ValueError("Labels must be non-negative.")
    num_classes = num_classes or (int(labels.max()) + 1)
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1.0  # index-scatter, faster than looping
    return one_hot


def clip_by_norm(x, max_norm, axis=None):
    # Input:  x (ndarray), max_norm (float), axis (int|None)
    # Output: ndarray same shape - x scaled down if norm > max_norm, else unchanged
    x = np.asarray(x, dtype=float)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    scale = np.minimum(1.0, max_norm / (norm + 1e-12))  # 1e-12 avoids div by zero
    return x * scale