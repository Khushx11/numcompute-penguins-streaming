# Streaming evaluation protocol

## Why evaluation order matters

The original `StreamTrainer.run` uses **train-then-test on the same chunk**. It updates the estimator with a chunk and then evaluates that chunk. This is useful for checking that the pipeline works, but does **not** estimate generalisation to future observations.

The alternative `StreamTrainer.run_prequential` follows a **test-then-train** sequence:

1. Fit the first chunk solely to initialise the model. Its evaluation metrics are `NaN` and `evaluated=False`.
2. Predict the next chunk **before** updating the model with its labels.
3. Record that chunk's accuracy and mean squared error, then call `partial_fit`.
4. Continue in the supplied order, retaining cumulative metrics for only the chunks evaluated before training.

This separation is critical for time-ordered or continually arriving observations.

## Minimal example

```python
from numcompute_stream.io import make_stream_chunks
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.stream import StreamTrainer

chunks = make_stream_chunks(X, y, chunk_size=32)
trainer = StreamTrainer(DecisionTreeClassifier(max_depth=4), classes=[0, 1])
log = trainer.run_prequential(chunks)

observed = [entry for entry in log if entry["evaluated"]]
print(f"Evaluated chunks: {len(observed)}")
print(f"Accuracy after last evaluated chunk: {observed[-1]['cumulative_accuracy']:.3f}"
      if observed else "Only the warm-up chunk was seen")
```

**Important:** The first chunk is excluded from prequential performance totals. The logged `n_samples` count includes processed samples, while `cumulative_total` counts only evaluated samples. Model updates are performed on each chunk after pre-update evaluation.

## Interpretation

Accuracy is the proportion of labels predicted correctly before that chunk's training update:

\[
\mathrm{Accuracy}_t = \frac{\sum_{i=2}^{t} \mathrm{Correct}_i}{\sum_{i=2}^{t} n_i}
\]

where chunk 1 is the initial warm-up chunk.

The `loss` field uses mean squared error between integer class predictions and integer labels, which is **not cross-entropy or a calibrated probabilistic loss**. For multiclass labels, its numeric value depends on how classes are encoded.

## Preprocessing in a streaming pipeline

The pipeline fits preprocessing on the first chunk and holds its parameters fixed for subsequent batches. Buffered models must compare old and new observations in the same feature space. A full call to `fit` refreshes both preprocessing and the estimator using the supplied data. The standalone incremental transformers can still be updated independently.

## Limitations

- The demonstration tree and ensemble implementations accumulate training data and **rebuild** their trees on successive calls to `partial_fit`. They are not bounded-memory online learners.
- Keep chunks in the intended chronological order. Shuffling destroys the meaning of temporal evaluation.
- Repeatedly tuning hyperparameters using the same evaluation stream still creates selection bias.
- Concept drift, sample leakage across entities, class imbalance and nonstationary labels require additional investigation.
- This method is not a replacement for a separately held-out external dataset or a true production pilot.

Use the `run` method for train-then-score sanity checks; use `run_prequential` when the purpose is to examine performance on future chunks.
