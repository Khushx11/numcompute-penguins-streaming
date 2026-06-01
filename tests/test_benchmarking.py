import numpy as np
import pytest
from numcompute.benchmarking import benchmark_fn


def test_benchmark_fn_returns_floats():
    mean_time, std_time = benchmark_fn(np.sum, args=(np.ones(10),), repeats=5)

    assert isinstance(mean_time, float)
    assert isinstance(std_time, float)


def test_benchmark_fn_times_are_non_negative():
    mean_time, std_time = benchmark_fn(np.sum, args=(np.ones(10),), repeats=5)

    assert mean_time >= 0
    assert std_time >= 0


def test_benchmark_fn_invalid_repeats():
    with pytest.raises(ValueError):
        benchmark_fn(np.sum, args=(np.ones(10),), repeats=0)


def test_benchmark_fn_non_callable():
    with pytest.raises(TypeError):
        benchmark_fn("not_a_function", args=(), repeats=5)