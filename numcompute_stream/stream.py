"""
stream.py

StreamTrainer ties together the pipeline, metrics, and logging into
one object. You feed it chunks one at a time and it handles fitting,
scoring, and keeping a log of how the model is performing over time.
"""

import numpy as np
import time


class StreamTrainer:
    """
    Manages incremental training over a stream of data chunks.

    Wraps a pipeline or model and handles the fit/score/log loop
    so the demo notebook and benchmarks do not need to repeat that
    boilerplate. Tracks per-chunk accuracy, cumulative accuracy,
    loss, and timing.

    Parameters
    ----------
    model : object
        Any object with partial_fit(X, y) and predict(X) methods.
        Typically a Pipeline or one of the ensemble classifiers.
    classes : array-like or None
        All expected class labels. Pass this upfront if early chunks
        might not contain all classes.

    Attributes
    ----------
    chunk_log : list of dict
        One entry per chunk with keys: chunk, n_samples, accuracy,
        cumulative_accuracy, loss, elapsed_s.
    cumulative_correct : int
    cumulative_total : int
    n_chunks : int
    """

    def __init__(self, model, classes=None):
        self.model = model
        self.classes = np.asarray(classes, dtype=int) if classes is not None else None
        self.chunk_log = []
        self.cumulative_correct = 0
        self.cumulative_total = 0
        self.n_chunks = 0

    def fit_chunk(self, X, y):
        """
        Fit the model on one chunk and log performance.

        Trains on the full chunk first, then scores on the same chunk.
        Logs accuracy, cumulative accuracy, loss, timing, and memory
        footprint of the model buffer where available.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)

        Returns
        -------
        dict
            The log entry for this chunk.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        t_start = time.perf_counter()

        if self.classes is not None:
            self.model.partial_fit(X, y, classes=self.classes)
        else:
            self.model.partial_fit(X, y)

        elapsed = time.perf_counter() - t_start

        score_entry = self.score_chunk(X, y, _log=False)

        self.cumulative_correct += score_entry["correct"]
        self.cumulative_total += len(y)
        self.n_chunks += 1

        # estimate memory footprint of whatever buffer the model holds
        mem_bytes = self._estimate_memory()

        entry = {
            "chunk": self.n_chunks,
            "n_samples": len(y),
            "accuracy": score_entry["accuracy"],
            "cumulative_accuracy": (
                self.cumulative_correct / self.cumulative_total
                if self.cumulative_total > 0 else 0.0
            ),
            "loss": score_entry["loss"],
            "elapsed_s": round(elapsed, 4),
            "mem_bytes": mem_bytes,
        }
        self.chunk_log.append(entry)
        return entry

    def score_chunk(self, X, y, _log=True):
        """
        Score the current model on a chunk without fitting.

        Use this to evaluate on a held-out validation chunk after
        each training chunk if you want a proper train/val split.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        _log : bool
            Internal flag. Set to False to skip appending to chunk_log.

        Returns
        -------
        dict with keys: accuracy, loss, correct, n_samples
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        preds = self.model.predict(X)
        correct = int(np.sum(preds == y))
        accuracy = correct / len(y) if len(y) > 0 else 0.0
        loss = float(np.mean((preds - y) ** 2))

        return {
            "accuracy": accuracy,
            "loss": loss,
            "correct": correct,
            "n_samples": len(y),
        }

    def run(self, chunks):
        """
        Convenience method to train on a full list of chunks in one call.

        Parameters
        ----------
        chunks : list of (X_chunk, y_chunk) tuples

        Returns
        -------
        list of dict
            The full chunk_log after processing all chunks.
        """
        for X_chunk, y_chunk in chunks:
            self.fit_chunk(X_chunk, y_chunk)
        return self.chunk_log

    def run_prequential(self, chunks):
        """
        Evaluate each chunk BEFORE fitting it, after one warm-up chunk.

        The first chunk is used solely to initialise the estimator; no
        accuracy is claimed for it. Subsequent chunks are predicted before
        partial_fit to prevent evaluation on the same data used to train.

        This is a test-then-train protocol, not a substitute for a separate
        independent evaluation dataset. See docs/evaluation-protocol.md.
        """
        for X_chunk, y_chunk in chunks:
            X = np.asarray(X_chunk, dtype=float)
            y = np.asarray(y_chunk, dtype=int)
            if X.ndim != 2 or y.ndim != 1 or len(y) == 0 or len(X) != len(y):
                raise ValueError("each chunk must have matching nonempty 2D X and 1D y")

            t_start = time.perf_counter()
            is_warmup = self.n_chunks == 0
            if not is_warmup:
                score = self.score_chunk(X, y, _log=False)

            if self.classes is None:
                self.model.partial_fit(X, y)
            else:
                self.model.partial_fit(X, y, classes=self.classes)

            if not is_warmup:
                self.cumulative_correct += score["correct"]
                self.cumulative_total += len(y)
            self.n_chunks += 1
            entry = {
                "chunk": self.n_chunks,
                "n_samples": len(y),
                "accuracy": float("nan") if is_warmup else score["accuracy"],
                "cumulative_accuracy": (
                    float("nan") if is_warmup
                    else self.cumulative_correct / self.cumulative_total
                ),
                "loss": float("nan") if is_warmup else score["loss"],
                "elapsed_s": round(time.perf_counter() - t_start, 4),
                "mem_bytes": self._estimate_memory(),
                "evaluated": not is_warmup,
            }
            self.chunk_log.append(entry)

        return self.chunk_log

    def accuracy_history(self):
        """
        Per-chunk accuracy values extracted from the log.

        Returns
        -------
        list of float
        """
        return [e["accuracy"] for e in self.chunk_log]

    def cumulative_accuracy_history(self):
        """
        Cumulative accuracy after each chunk.

        Returns
        -------
        list of float
        """
        return [e["cumulative_accuracy"] for e in self.chunk_log]

    def loss_history(self):
        """
        Per-chunk loss values from the log.

        Returns
        -------
        list of float
        """
        return [e["loss"] for e in self.chunk_log]

    def summary(self):
        """
        Print a short summary table of the chunk log to stdout.
        """
        if not self.chunk_log:
            print("no chunks processed yet")
            return
        print(f"{'chunk':>6}  {'samples':>8}  {'acc':>8}  {'cum_acc':>8}  {'loss':>8}  {'time_s':>8}  {'mem_kb':>8}")
        print("-" * 70)
        for e in self.chunk_log:
            print(
                f"{e['chunk']:>6}  "
                f"{e['n_samples']:>8}  "
                f"{e['accuracy']:>8.4f}  "
                f"{e['cumulative_accuracy']:>8.4f}  "
                f"{e['loss']:>8.4f}  "
                f"{e['elapsed_s']:>8.4f}  "
                f"{e.get('mem_bytes', 0) // 1024:>8}"
            )
        print("-" * 70)
        print(f"final cumulative accuracy: {self.chunk_log[-1]['cumulative_accuracy']:.4f}")

    def reset(self):
        """
        Clear the log and reset all counters.

        Does not reset the model itself, just the trainer state.
        """
        self.chunk_log = []
        self.cumulative_correct = 0
        self.cumulative_total = 0
        self.n_chunks = 0
        return self

    def _estimate_memory(self):
        """
        Estimate memory used by the model's internal data buffer in bytes.

        Checks for _X_buf on the model or its pipeline estimator.
        Returns 0 if no buffer is found, which just means the model does
        not store raw data (e.g. a pure stats tracker).

        Returns
        -------
        int
            Approximate bytes used by the buffer array.
        """
        # check direct model first
        model = self.model
        # if it is a pipeline, look at the final estimator
        if hasattr(model, '_estimator'):
            model = model._estimator
        elif hasattr(model, 'steps'):
            model = model.steps[-1][1]

        # check for a numpy buffer on the model or its inner classifier
        for attr in ('_X_buf', '_clf'):
            obj = getattr(model, attr, None)
            if obj is None:
                continue
            if isinstance(obj, np.ndarray):
                return obj.nbytes
            # EnsembleClassifier wraps an inner clf
            inner_buf = getattr(obj, '_X_buf', None)
            if isinstance(inner_buf, np.ndarray):
                return inner_buf.nbytes

        return 0

    def memory_history(self):
        """
        Buffer memory usage in bytes after each chunk.

        Returns
        -------
        list of int
        """
        return [e["mem_bytes"] for e in self.chunk_log]
