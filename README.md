# NumCompute-Stream

A streaming, tree-based machine learning framework built on top of NumCompute.
Supports incremental learning via partial_fit throughout, so models and
preprocessors can be updated chunk by chunk as new data arrives.

Built with only NumPy and matplotlib. No scikit-learn, no pandas.

---

## Package structure
numcompute_stream/
├── stats.py          incremental mean, variance, histogram
├── preprocessing.py  StandardScaler, MinMaxScaler, Imputer, OneHotEncoder with partial_fit
├── metrics.py        accuracy, loss, precision, recall, F1, confusion matrix, rolling window
├── io.py             CSV loader, chunked reader, array splitter
├── tree.py           DecisionTreeClassifier with partial_fit
├── ensemble.py       RandomForestClassifier, BaggingClassifier, EnsembleClassifier
├── pipeline.py       Pipeline with partial_fit chaining
├── stream.py         StreamTrainer for chunk-by-chunk training and logging
└── visualise.py      metric plots, model comparison, confusion matrix heatmap

---

## Installation

Clone the repo and install in editable mode:

```bash
git clone https://github.com/Khushx11/NumCompute-Stream.git
cd NumCompute-Stream
pip install -e .
```

Requires Python 3.10+ and NumPy and matplotlib.

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

# some data
rng = np.random.default_rng(0)
X = rng.normal(size=(200, 4))
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# split into streaming chunks
chunks = make_stream_chunks(X, y, chunk_size=20)

# build a pipeline
pipe = Pipeline([
    ('scale', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=10, max_depth=4, random_state=0)),
])

# train incrementally
trainer = StreamTrainer(pipe, classes=[0, 1])
trainer.run(chunks)
trainer.summary()

# visualise accuracy over time
vis.plot_metric_over_time(
    trainer.cumulative_accuracy_history(),
    title='Cumulative accuracy over chunks',
    ylabel='accuracy'
)
```

---

## Streaming API

Every component follows the same pattern:

| method | what it does |
|---|---|
| `partial_fit(X, y)` | update with one chunk, safe to call repeatedly |
| `fit(X, y)` | full fit on a single dataset, resets state |
| `transform(X)` | apply fitted transform |
| `predict(X)` | predict class labels |
| `predict_proba(X)` | class probability estimates |

---

## Running the demo

```bash
cd demo
jupyter notebook stream_demo.ipynb
```

The notebook loads penguins.csv, splits it into chunks, trains a decision
tree, random forest, and bagging ensemble incrementally, then plots accuracy
over time and a confusion matrix.

---

## Running the tests

From the repo root:

```bash
pip install pytest
pytest tests/ -v
```

There are 40+ unit tests covering standard functionality and edge cases
including NaN inputs, zero-variance features, single-sample chunks,
rolling window metrics, and streaming category expansion.

---

## Running the benchmarks

```bash
python benchmark/run_benchmarks.py
```

Runs four benchmarks:
1. NaN imputation: loop vs vectorised
2. Confusion matrix: loop vs vectorised
3. Single tree vs random forest vs bagging under streaming
4. StandardScaler partial_fit cost as data accumulates

---

## Design notes

**Streaming approach:** All stateful components buffer data or maintain
running statistics so partial_fit can be called any number of times.
The decision tree rebuilds from the full accumulated buffer on each
partial_fit call. This is simpler and more correct than true online
tree growth algorithms while still demonstrating incremental learning.

**Numerical stability:** NaNs are dropped before Welford updates.
Zero-variance features default to std=1 to avoid division by zero.
Histogram bin edges handle the case where all values are identical.

**Ensemble methods:** RandomForest adds feature subsampling per split
on top of bootstrap sampling. BaggingClassifier uses bootstrap sampling
only. EnsembleClassifier wraps both with a shared interface so you can
swap methods without changing the rest of the pipeline.

---

## Author

Khush Bhavin Maru
Student ID: a[redacted]
University of Adelaide
Course: Programming for AI