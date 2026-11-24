import time
import numpy as np
from numcompute.benchmarking import benchmark_fn
from numcompute.sort_search import quicksort, top_k
from numcompute.preprocessing import StandardScaler
from numcompute.utils import get_euclidean_dist

def loop_sort(arr):
    arr_list = list(arr)
    arr_list.sort()
    return np.array(arr_list)

def loop_top_k(arr, k):
    arr_list = list(arr)
    arr_list.sort(reverse=True)
    return np.array(arr_list[:k])

def loop_scaler(arr):
    n_rows = len(arr)
    n_cols = len(arr[0])
    
    means = [0.0] * n_cols
    for row in arr:
        for j in range(n_cols):
            means[j] += row[j]
    means = [m / n_rows for m in means]
    
    vars = [0.0] * n_cols
    for row in arr:
        for j in range(n_cols):
            vars[j] += (row[j] - means[j]) ** 2
    stds = [(v / n_rows) ** 0.5 for v in vars]
    stds = [s if s > 0 else 1.0 for s in stds]
    
    out = []
    for row in arr:
        out_row = [(row[j] - means[j]) / stds[j] for j in range(n_cols)]
        out.append(out_row)
    return np.array(out)

def loop_pairwise(arr):
    n = len(arr)
    d = len(arr[0])
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_sq = 0.0
            for k in range(d):
                dist_sq += (arr[i, k] - arr[j, k]) ** 2
            out[i, j] = dist_sq ** 0.5
    return out

def run_all():
    print("Generating data...")
    np.random.seed(42)
    arr_100k = np.random.randn(100000)
    arr_100k_2d = np.random.randn(100000, 10)
    arr_1k_2d = np.random.randn(1000, 10)
    
    scaler = StandardScaler()
    
    print("-" * 65)
    print(f"| {'Operation':<21} | {'Vectorised (ms)':<15} | {'Loop (ms)':<10} | {'Speedup':<7} |")
    print("|" + "-"*23 + "|" + "-"*17 + "|" + "-"*12 + "|" + "-"*9 + "|")
    
    # 1. Sort n=100k
    vm, _ = benchmark_fn(quicksort, (arr_100k,), repeats=3)
    lm, _ = benchmark_fn(loop_sort, (arr_100k,), repeats=3)
    print(f"| {'Sort n=100k':<21} | {vm*1000:<15.1f} | {lm*1000:<10.1f} | {lm/vm:<6.2f}x |")
    
    # 2. Top-k n=100k
    vm, _ = benchmark_fn(top_k, (arr_100k, 10), repeats=3)
    lm, _ = benchmark_fn(loop_top_k, (arr_100k, 10), repeats=3)
    print(f"| {'Top-k n=100k':<21} | {vm*1000:<15.1f} | {lm*1000:<10.1f} | {lm/vm:<6.2f}x |")
    
    # 3. StandardScaler n=100k
    arr_100k_list = arr_100k_2d.tolist()
    vm, _ = benchmark_fn(scaler.fit_transform, (arr_100k_2d,), repeats=3)
    lm, _ = benchmark_fn(loop_scaler, (arr_100k_list,), repeats=3)
    print(f"| {'StandardScaler n=100k':<21} | {vm*1000:<15.1f} | {lm*1000:<10.1f} | {lm/vm:<6.2f}x |")
    
    # 4. Pairwise L2 n=1k
    vm, _ = benchmark_fn(get_euclidean_dist, (arr_1k_2d,), repeats=3)
    lm, _ = benchmark_fn(loop_pairwise, (arr_1k_2d,), repeats=1)
    print(f"| {'Pairwise L2 n=1k':<21} | {vm*1000:<15.1f} | {lm*1000:<10.1f} | {lm/vm:<6.2f}x |")
    print("-" * 65)

if __name__ == '__main__':
    run_all()
