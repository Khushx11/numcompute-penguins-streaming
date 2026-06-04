# NumCompute-Stream

Streaming, tree-based machine learning built from scratch using NumPy only.

This extends the base NumCompute package with real incremental learning. Every
component accepts data chunk by chunk through partial_fit(), so models keep
updating as new data arrives without ever needing the full dataset loaded at once.
No scikit-learn, no pandas, no shortcuts.

---

## What is in the box

| module | what it does |
|---|---|
| `stats.py` | Welford running mean/variance, streaming histogram, quantiles |
| `preprocessing.py` | StandardScaler, MinMaxScaler, Imputer, OneHotEncoder with partial_fit |
| `metrics.py` | Accuracy, loss, precision, recall, F1, confusion matrix, rolling window |
| `io.py` | CSV loader, chunked reader, in-memory stream splitter |
| `tree.py` | Decision tree classifier with Gini/entropy and partial_fit |
| `ensemble.py` | Random Forest, Bagging, and a unified EnsembleClassifier wrapper |
| `pipeline.py` | Pipeline with partial_fit chaining across all steps |
| `stream.py` | StreamTrainer for the chunk loop, metric logging, and timing |
| `visualise.py` | Metric plots, model comparison, confusion matrix heatmap |

---

## Installation

```bash
git clone https://github.com/Khushx11/NumCompute-Stream.git
cd NumCompute-Stream
pip install -e .
```

Python 3.10+, NumPy, and matplotlib. That is all.

---

## Quick start

```python
import numpy as np
from numcompute_stream.io import make_stream_chunks
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.ensemble import RandomForestClassifier
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.stream import StreamTrainer
import numcompute_stream.visualise as vis

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 4))
y = (X[:, 0] + X[:, 1] > 0).astype(int)

chunks = make_stream_chunks(X, y, chunk_size=20)

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=10, max_depth=4, random_state=0)),
])

trainer = StreamTrainer(pipe, classes=[0, 1])
trainer.run(chunks)
trainer.summary()

vis.plot_metric_over_time(
    trainer.cumulative_accuracy_history(),
    title='Cumulative accuracy over chunks',
    ylabel='accuracy',
)
```

---

## The streaming API

Every component follows the same interface so pieces are interchangeable
without rewriting anything around them:

```
partial_fit(X, y)   update with one chunk, call as many times as needed
fit(X, y)           full fit on a single dataset, clears previous state
transform(X)        apply fitted transform to new data
predict(X)          predict class labels
predict_proba(X)    class probability estimates
```

---

## Running the demo

```bash
cd demo
jupyter notebook stream_demo.ipynb
```

The notebook loads penguins.csv, splits it into 10-sample chunks, and trains
a decision tree, random forest, and bagging ensemble side by side. Plots
include cumulative accuracy over time, loss, a model comparison, predictions
vs ground truth on the last chunk, and a full confusion matrix.

---

## Running the tests

```bash
pytest tests/ -v
```

208 tests across 13 files. Normal cases and edge cases both: NaN inputs,
zero-variance features, single-sample chunks, rolling window accuracy,
incremental category expansion in the encoder, shape mismatches, and
calling predict before fit.

---

## Running the benchmarks

```bash
python benchmark/run_benchmarks.py
```

Four comparisons: loop vs vectorised NaN imputation, loop vs vectorised
confusion matrix building, single tree vs Random Forest vs Bagging over
streaming chunks, and StandardScaler partial_fit cost as chunk count grows.

---

## Design notes

The tree uses a rebuild-from-buffer strategy rather than a proper online
algorithm like Hoeffding trees. Each partial_fit call adds the new chunk
to an internal buffer and rebuilds the whole tree from scratch. Hoeffding
trees handle truly unbounded streams more efficiently but the statistical
bounds they rely on are hard to implement correctly and even harder to
write good tests for. The rebuild approach trades memory efficiency for
simplicity and correctness, and the benchmark shows exactly how the cost
grows so there are no surprises.

Running mean and variance use Welford's algorithm throughout stats.py and
preprocessing.py. It runs in O(1) per sample, stores nothing from previous
chunks, and stays numerically stable where a naive sum-of-squares would
drift on large inputs.

Random Forest and Bagging share the same EnsembleClassifier interface so
switching between them is a single constructor argument. The only real
difference is feature subsampling at each split: Random Forest does it,
Bagging does not.

---

## Repo layout

```
NumCompute-Stream/
├── numcompute_stream/   core package (9 modules)
├── numcompute/          base NumCompute package
├── tests/               177 unit tests
├── demo/                stream_demo.ipynb + penguins.csv
├── benchmark/           run_benchmarks.py
├── report/              report.pdf
└── README.md
```

---

## Author

Khush Bhavin Maru (a[redacted])
University of Adelaide
Programming for Artificial Intelligence — COMP 5004