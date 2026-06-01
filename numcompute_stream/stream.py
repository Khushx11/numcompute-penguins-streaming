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
        This gives an optimistic but consistent measure of how well the
        model has learned so far after seeing this data.

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

        # score on this chunk right after fitting it
        score_entry = self.score_chunk(X, y, _log=False)

        self.cumulative_correct += score_entry["correct"]
        self.cumulative_total += len(y)
        self.n_chunks += 1

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

        Useful for a quick sanity check in the notebook or terminal.
        """
        if not self.chunk_log:
            print("no chunks processed yet")
            return
        print(f"{'chunk':>6}  {'samples':>8}  {'acc':>8}  {'cum_acc':>8}  {'loss':>8}  {'time_s':>8}")
        print("-" * 58)
        for e in self.chunk_log:
            print(
                f"{e['chunk']:>6}  "
                f"{e['n_samples']:>8}  "
                f"{e['accuracy']:>8.4f}  "
                f"{e['cumulative_accuracy']:>8.4f}  "
                f"{e['loss']:>8.4f}  "
                f"{e['elapsed_s']:>8.4f}"
            )
        print("-" * 58)
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
