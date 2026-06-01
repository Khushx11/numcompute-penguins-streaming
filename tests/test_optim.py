import numpy as np
import pytest
from numcompute.optim import finite_diff_gradient, finite_diff_jacobian


def test_gradient_quadratic():
    def f(x):
        return np.sum(x ** 2)

    x = np.array([1.0, 2.0, 3.0])
    grad = finite_diff_gradient(f, x)
    assert np.allclose(grad, np.array([2.0, 4.0, 6.0]), atol=1e-4)


def test_gradient_linear():
    def f(x):
        return 3 * x[0] + 5 * x[1]

    x = np.array([2.0, 7.0])
    grad = finite_diff_gradient(f, x)
    assert np.allclose(grad, np.array([3.0, 5.0]), atol=1e-4)


def test_gradient_non_1d():
    def f(x):
        return np.sum(x)

    with pytest.raises(ValueError):
        finite_diff_gradient(f, np.array([[1.0, 2.0]]))


def test_jacobian_linear():
    def f(x):
        return np.array([2 * x[0] + x[1], 3 * x[0] - x[1]])
    
    x = np.array([1.0, 2.0])
    J = finite_diff_jacobian(f, x)
    assert np.allclose(J, np.array([[2.0, 1.0], [3.0, -1.0]]), atol=1e-4)


def test_jacobian_non_1d():
    def f(x):
        return x
    
    with pytest.raises(ValueError):
        finite_diff_jacobian(f, np.array([[1.0]]))