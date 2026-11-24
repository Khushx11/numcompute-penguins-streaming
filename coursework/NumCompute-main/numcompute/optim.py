import numpy as np


def finite_diff_gradient(func, x, eps=1e-5):
    """
    Estimates the gradient of func at x using central finite differences.

    Parameters
    ----------
    func : callable
        Function to compute gradient for. Should take a 1-D array and return a scalar.
    x : array-like, shape (n,)
        Point at which to evaluate the gradient.
    eps : float, optional
        Step size for finite differences, default is 1e-5.

    Returns
    -------
    np.ndarray, shape (n,)
        Estimated gradient vector.

    Raises
    ------
    ValueError
        If x is not 1-D.

    Complexity
    ----------
    Time  : O(n * T(func))
    Space : O(n)
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x.shape}")

    grad = np.zeros_like(x)

    for i in range(len(x)):
        x_fwd = x.copy()
        x_bwd = x.copy()
        x_fwd[i] += eps
        x_bwd[i] -= eps
        grad[i] = (func(x_fwd) - func(x_bwd)) / (2 * eps)

    return grad


def finite_diff_jacobian(func, x, eps=1e-5):
    """
    Estimates the Jacobian of a vector-valued function at x using central finite differences.

    Parameters
    ----------
    func : callable
        Function to compute Jacobian for. Should take a 1-D array of size n and return a 1-D array of size m.
    x : array-like, shape (n,)
        Point at which to evaluate the Jacobian.
    eps : float, optional
        Step size for finite differences, default is 1e-5.

    Returns
    -------
    np.ndarray, shape (m, n)
        Estimated Jacobian matrix where J[i,j] = d func_i / d x_j.

    Raises
    ------
    ValueError
        If x is not 1-D.

    Complexity
    ----------
    Time  : O(n * T(func))
    Space : O(m * n)
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x.shape}")

    f0 = np.asarray(func(x), dtype=float)
    m, n = len(f0), len(x)
    J = np.zeros((m, n))

    # central differences column by column - one pair of evals per input dimension
    for i in range(n):
        x_fwd, x_bwd = x.copy(), x.copy()
        x_fwd[i] += eps
        x_bwd[i] -= eps
        J[:, i] = (np.asarray(func(x_fwd), dtype=float) - np.asarray(func(x_bwd), dtype=float)) / (2 * eps)

    return J