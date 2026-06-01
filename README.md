# NumCompute

A modular, production-grade scientific computing toolkit built with plain Python and NumPy. No external ML/DL libraries.

## Installation

```bash
pip install -e ".[dev]"
```

## Modules

| Module | Description |
|---|---|
| `io` | Streaming CSV reader with dtype inference and chunking |
| `preprocessing` | StandardScaler, MinMaxScaler, Imputer, OneHotEncoder |
| `sort_search` | Vectorised sorts, top-k, quickselect, binary search |
| `rank` | Ranking with tie strategies, percentile ranks |
| `stats` | Welford streaming stats, histograms, quantiles |
| `metrics` | Accuracy, Precision, Recall, F1, MSE, ROC/AUC, confusion matrix |
| `optim` | Finite-difference gradients, Jacobians, line search |
| `pipeline` | Transformer/Estimator API, Compose, FeatureUnion |
| `utils` | Distances, activations, logsumexp, batching helpers |
| `benchmarking` | Micro-benchmark harness, vectorised vs loop comparisons |

## Quick Start

```python
import numpy as np
from numcompute.preprocessing import StandardScaler
from numcompute.pipeline import Compose

scaler = StandardScaler()
X = np.random.randn(100, 4)
X_scaled = scaler.fit_transform(X)
```

## Running Tests

```bash
pytest tests/ -v --cov=numcompute
```

## Benchmarks

```bash
python benchmark/run_benchmarks.py
```

## Performance Table

-----------------------------------------------------------------
| Operation             | Vectorised (ms) | Loop (ms)  | Speedup |
|-----------------------|-----------------|------------|---------|
| Sort n=100k           | 5573.2          | 125.4      | 0.02  x |
| Top-k n=100k          | 4615.6          | 82.8       | 0.02  x |
| StandardScaler n=100k | 29.2            | 1092.7     | 37.42 x |
| Pairwise L2 n=1k      | 18.2            | 9208.3     | 505.71x |
-----------------------------------------------------------------

*(Populated after running `benchmark/run_benchmarks.py`)*

## Requirements

- Python >= 3.10
- NumPy >= 1.26

## Team

| Name | GitHub | Modules |
|---|---|---|
| Jaineel Patel | [@jaineel11](https://github.com/jaineel11) | `utils.py`, `io.py` |
| Harshal Lad | [@Harshal192](https://github.com/Harshal192) | `preprocessing.py`, `pipeline.py` |
| Khush Maru | [@Khushx11](https://github.com/Khushx11) | `sort_search.py`, `rank.py`|
| Vaibhav Patel | [@vaibhavpatel26](https://github.com/vaibhavpatel26) | `stats.py`, `metrics.py`, `optim.py`, `benchmarking.py` |