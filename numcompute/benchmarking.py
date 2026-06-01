import numpy as np
import time


def benchmark_fn(func, args=(), repeats=100):
    """Runs func repeatedly and returns mean and std of execution time in seconds."""
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")
    if not callable(func):
        raise TypeError("func must be callable")

    times = []

    for _ in range(repeats):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append(end - start)

    times = np.array(times)
    return float(times.mean()), float(times.std())