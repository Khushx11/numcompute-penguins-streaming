"""
run_benchmarks.py

Compares loop-based vs vectorised implementations under streaming
conditions. Also benchmarks single tree vs random forest as the
number of chunks grows. Run this from the repo root with:

    python benchmark/run_benchmarks.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import RandomForestClassifier, BaggingClassifier
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.io import make_stream_chunks


# how wide to print the results table
TABLE_WIDTH = 62


def separator():
    print("-" * TABLE_WIDTH)


def header(title):
    print()
    print("=" * TABLE_WIDTH)
    print(f"  {title}")
    print("=" * TABLE_WIDTH)


def time_fn(fn, repeats=3):
    """Run fn() a few times and return the best elapsed time in ms."""
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return min(times)


# -----------------------------------------------------------------------
# benchmark 1: loop vs vectorised imputation
# -----------------------------------------------------------------------

def impute_loop(X, fill_values):
    """Fill NaNs column by column with a Python loop."""
    X = X.copy()
    for i in range(X.shape[1]):
        col = X[:, i]
        col[np.isnan(col)] = fill_values[i]
        X[:, i] = col
    return X


def impute_vectorised(X, fill_values):
    """Fill NaNs using vectorised indexing."""
    X = X.copy()
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(fill_values, np.where(nan_mask)[1])
    return X


def bench_imputation():
    header("Benchmark 1: NaN imputation  loop vs vectorised")
    rng = np.random.default_rng(0)

    for n in [500, 2000, 10000]:
        X = rng.normal(size=(n, 20))
        # inject 10% NaNs
        mask = rng.random(X.shape) < 0.1
        X[mask] = np.nan
        fill = rng.normal(size=20)

        t_loop = time_fn(lambda: impute_loop(X, fill))
        t_vec  = time_fn(lambda: impute_vectorised(X, fill))
        speedup = t_loop / t_vec if t_vec > 0 else float("inf")

        print(f"  n={n:<6}  loop={t_loop:6.2f}ms  vec={t_vec:6.2f}ms  speedup={speedup:.1f}x")


# -----------------------------------------------------------------------
# benchmark 2: loop vs vectorised confusion matrix accumulation
# -----------------------------------------------------------------------

def confusion_loop(y_true, y_pred, n_classes):
    """Build confusion matrix with a Python loop."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def confusion_vectorised(y_true, y_pred, n_classes):
    """Build confusion matrix using np.add.at."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def bench_confusion():
    header("Benchmark 2: confusion matrix  loop vs vectorised")
    rng = np.random.default_rng(1)
    n_classes = 5

    for n in [500, 5000, 50000]:
        y_true = rng.integers(0, n_classes, size=n)
        y_pred = rng.integers(0, n_classes, size=n)

        t_loop = time_fn(lambda: confusion_loop(y_true, y_pred, n_classes))
        t_vec  = time_fn(lambda: confusion_vectorised(y_true, y_pred, n_classes))
        speedup = t_loop / t_vec if t_vec > 0 else float("inf")

        print(f"  n={n:<6}  loop={t_loop:6.2f}ms  vec={t_vec:6.2f}ms  speedup={speedup:.1f}x")


# -----------------------------------------------------------------------
# benchmark 3: single tree vs random forest under streaming
# -----------------------------------------------------------------------

def bench_streaming_models():
    header("Benchmark 3: single tree vs RF vs bagging  streaming")
    rng = np.random.default_rng(2)

    n_total = 300
    n_features = 8
    X = rng.normal(size=(n_total, n_features))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    chunks = make_stream_chunks(X, y, chunk_size=30)

    models = [
        ("DecisionTree   ", DecisionTreeClassifier(max_depth=4, random_state=0)),
        ("RandomForest10 ", RandomForestClassifier(n_estimators=10, max_depth=4, random_state=0)),
        ("Bagging10      ", BaggingClassifier(n_estimators=10, max_depth=4, random_state=0)),
    ]

    print(f"  {'model':<18}  {'total_ms':>10}  {'ms/chunk':>10}  {'final_acc':>10}")
    separator()

    for name, model in models:
        sc = StandardScaler()
        # pre-fit scaler on all data for fair comparison
        sc.partial_fit(X)

        t_start = time.perf_counter()
        for X_chunk, y_chunk in chunks:
            Xt = sc.transform(X_chunk)
            model.partial_fit(Xt, y_chunk, classes=[0, 1])
        total_ms = (time.perf_counter() - t_start) * 1000

        Xt_all = sc.transform(X)
        preds = model.predict(Xt_all)
        acc = float(np.mean(preds == y))

        print(
            f"  {name:<18}  {total_ms:>10.1f}  "
            f"{total_ms/len(chunks):>10.2f}  {acc:>10.4f}"
        )


# -----------------------------------------------------------------------
# benchmark 4: scaler partial_fit cost as buffer grows
# -----------------------------------------------------------------------

def bench_scaler_growth():
    header("Benchmark 4: StandardScaler cost as data grows")
    rng = np.random.default_rng(3)

    print(f"  {'n_chunks':<10}  {'partial_fit_ms':>14}")
    separator()

    sc = StandardScaler()
    for i in range(1, 11):
        X_chunk = rng.normal(size=(100, 10))
        t = time_fn(lambda: sc.partial_fit(X_chunk), repeats=5)
        print(f"  {i:<10}  {t:>14.3f}")


# -----------------------------------------------------------------------
# main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("NumCompute-Stream benchmark suite")
    print("numpy version:", np.__version__)

    bench_imputation()
    bench_confusion()
    bench_streaming_models()
    bench_scaler_growth()

    print()
    print("done")
