"""
metrics.py

Streaming classification metrics. Each class keeps a running tally
of true/false positives etc. so you can call update() after each chunk
and get a valid result at any point without storing past predictions.
"""

import numpy as np


class AccuracyTracker:
    """
    Tracks cumulative accuracy across streaming chunks.

    Keeps a simple count of correct predictions and total predictions
    seen so far. Call update() after each chunk, result() to get the
    current accuracy.

    Attributes
    ----------
    correct : int
    total : int
    history : list of float
        Accuracy after each chunk, useful for plotting over time.
    """

    def __init__(self):
        self.correct = 0
        self.total = 0
        self.history = []

    def update(self, y_true, y_pred):
        """
        Add predictions from a new chunk into the running count.

        Parameters
        ----------
        y_true : array-like, shape (n,)
        y_pred : array-like, shape (n,)

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If shapes do not match or inputs are empty.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
            )
        if len(y_true) == 0:
            raise ValueError("chunk must not be empty")
        self.correct += int(np.sum(y_true == y_pred))
        self.total += len(y_true)
        self.history.append(self.result())
        return self

    def result(self):
        """
        Current cumulative accuracy.

        Returns
        -------
        float, or 0.0 if no data yet.
        """
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    def reset(self):
        """Clear all counts and history."""
        self.correct = 0
        self.total = 0
        self.history = []
        return self


class LossTracker:
    """
    Tracks cumulative mean squared error across chunks.

    Attributes
    ----------
    total_loss : float
    total : int
    history : list of float
    """

    def __init__(self):
        self.total_loss = 0.0
        self.total = 0
        self.history = []

    def update(self, y_true, y_pred):
        """
        Add MSE from a new chunk into the running total.

        Parameters
        ----------
        y_true : array-like, shape (n,)
        y_pred : array-like, shape (n,)

        Returns
        -------
        self
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
            )
        if len(y_true) == 0:
            raise ValueError("chunk must not be empty")
        self.total_loss += float(np.sum((y_true - y_pred) ** 2))
        self.total += len(y_true)
        self.history.append(self.result())
        return self

    def result(self):
        """Current cumulative MSE."""
        if self.total == 0:
            return 0.0
        return self.total_loss / self.total

    def reset(self):
        """Clear all state."""
        self.total_loss = 0.0
        self.total = 0
        self.history = []
        return self


class ClassificationReport:
    """
    Tracks precision, recall, and F1 per class across streaming chunks.

    Accumulates a confusion-matrix-style count of TP, FP, FN per class
    so the metrics stay correct as more chunks arrive.

    Attributes
    ----------
    classes : np.ndarray or None
        Unique class labels, set on first update call.
    tp, fp, fn : np.ndarray or None
        Per-class true positives, false positives, false negatives.
    history : list of dict
        Snapshot of precision/recall/f1 after each chunk.
    """

    def __init__(self):
        self.classes = None
        self.tp = None
        self.fp = None
        self.fn = None
        self.history = []

    def update(self, y_true, y_pred):
        """
        Accumulate TP/FP/FN counts from a new chunk.

        Parameters
        ----------
        y_true : array-like, shape (n,)
        y_pred : array-like, shape (n,)

        Returns
        -------
        self
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
            )
        # initialise class list from first chunk
        if self.classes is None:
            self.classes = np.unique(y_true)
            n = len(self.classes)
            self.tp = np.zeros(n, dtype=int)
            self.fp = np.zeros(n, dtype=int)
            self.fn = np.zeros(n, dtype=int)
        for i, c in enumerate(self.classes):
            self.tp[i] += int(np.sum((y_pred == c) & (y_true == c)))
            self.fp[i] += int(np.sum((y_pred == c) & (y_true != c)))
            self.fn[i] += int(np.sum((y_pred != c) & (y_true == c)))
        self.history.append(self.result())
        return self

    def result(self):
        """
        Current precision, recall, and F1 per class plus macro averages.

        Returns
        -------
        dict with keys 'precision', 'recall', 'f1', 'macro_f1'
        """
        if self.classes is None:
            return {}
        prec = np.where(
            self.tp + self.fp > 0, self.tp / (self.tp + self.fp), 0.0
        )
        rec = np.where(
            self.tp + self.fn > 0, self.tp / (self.tp + self.fn), 0.0
        )
        denom = prec + rec
        f1 = np.where(denom > 0, 2 * prec * rec / denom, 0.0)
        return {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "macro_f1": float(np.mean(f1)),
        }

    def reset(self):
        """Wipe all accumulated counts."""
        self.classes = None
        self.tp = None
        self.fp = None
        self.fn = None
        self.history = []
        return self


class ConfusionMatrixTracker:
    """
    Accumulates a confusion matrix across streaming chunks.

    The matrix grows if new classes appear in later chunks, though
    ideally all classes are seen early on.

    Attributes
    ----------
    classes : np.ndarray or None
    matrix : np.ndarray or None
        Shape (n_classes, n_classes). Rows are true, columns are predicted.
    """

    def __init__(self):
        self.classes = None
        self.matrix = None

    def update(self, y_true, y_pred):
        """
        Add a new chunk into the confusion matrix.

        Parameters
        ----------
        y_true : array-like, shape (n,)
        y_pred : array-like, shape (n,)

        Returns
        -------
        self
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        all_labels = np.unique(np.concatenate([y_true, y_pred]))

        if self.classes is None:
            self.classes = all_labels
            n = len(self.classes)
            self.matrix = np.zeros((n, n), dtype=int)
        else:
            # expand matrix if new classes appear mid-stream
            new_classes = np.setdiff1d(all_labels, self.classes)
            if len(new_classes) > 0:
                self.classes = np.concatenate([self.classes, new_classes])
                self.classes = np.sort(self.classes)
                n = len(self.classes)
                new_matrix = np.zeros((n, n), dtype=int)
                # copy old counts into the new expanded matrix
                old_idx = np.searchsorted(self.classes, self.classes[
                    :len(self.matrix)
                ])
                for r in range(len(self.matrix)):
                    for c_col in range(len(self.matrix)):
                        new_matrix[old_idx[r], old_idx[c_col]] += self.matrix[r, c_col]
                self.matrix = new_matrix

        row_idx = np.searchsorted(self.classes, y_true)
        col_idx = np.searchsorted(self.classes, y_pred)
        np.add.at(self.matrix, (row_idx, col_idx), 1)
        return self

    def result(self):
        """
        Return the current confusion matrix and class labels.

        Returns
        -------
        dict with 'matrix' and 'classes', or empty dict if no data yet.
        """
        if self.matrix is None:
            return {}
        return {"matrix": self.matrix.copy(), "classes": self.classes.copy()}

    def reset(self):
        """Reset the matrix."""
        self.classes = None
        self.matrix = None
        return self


class RollingAccuracy:
    """
    Accuracy computed over a sliding window of the last N chunks
    rather than cumulatively.

    Useful for detecting when model performance is drifting over time.

    Parameters
    ----------
    window : int
        Number of recent chunks to keep in the window.
    """

    def __init__(self, window=5):
        if window < 1:
            raise ValueError("window must be at least 1")
        self.window = window
        # store (correct, total) per chunk in a circular buffer
        self._buffer = []
        self.history = []

    def update(self, y_true, y_pred):
        """
        Add a new chunk and drop the oldest if window is full.

        Parameters
        ----------
        y_true : array-like, shape (n,)
        y_pred : array-like, shape (n,)

        Returns
        -------
        self
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if len(y_true) == 0:
            raise ValueError("chunk must not be empty")
        correct = int(np.sum(y_true == y_pred))
        total = len(y_true)
        self._buffer.append((correct, total))
        # drop oldest chunk once we exceed the window size
        if len(self._buffer) > self.window:
            self._buffer.pop(0)
        self.history.append(self.result())
        return self

    def result(self):
        """
        Accuracy over the current window of chunks.

        Returns
        -------
        float
        """
        if not self._buffer:
            return 0.0
        total_correct = sum(c for c, _ in self._buffer)
        total = sum(t for _, t in self._buffer)
        return total_correct / total if total > 0 else 0.0

    def reset(self):
        """Clear the buffer and history."""
        self._buffer = []
        self.history = []
        return self


class StreamingAUC:
    """
    Approximate AUC for binary classification tracked across streaming chunks.

    Uses the Mann-Whitney U statistic approximation. Accumulates positive
    and negative score lists across chunks and recomputes AUC on demand.
    Not memory-free but gives exact AUC for all data seen so far.

    Attributes
    ----------
    pos_scores : list
        Predicted scores for positive class samples seen so far.
    neg_scores : list
        Predicted scores for negative class samples seen so far.
    history : list of float
        AUC after each chunk.
    """

    def __init__(self):
        self.pos_scores = []
        self.neg_scores = []
        self.history = []

    def update(self, y_true, y_scores, pos_label=1):
        """
        Add predictions from a new chunk.

        Parameters
        ----------
        y_true : array-like, shape (n,)
            True binary labels.
        y_scores : array-like, shape (n,)
            Predicted scores or probabilities for the positive class.
        pos_label : int, default 1
            Which label counts as positive.

        Returns
        -------
        self
        """
        y_true = np.asarray(y_true)
        y_scores = np.asarray(y_scores, dtype=float)
        if y_true.shape != y_scores.shape:
            raise ValueError(
                f"shape mismatch: y_true {y_true.shape} vs y_scores {y_scores.shape}"
            )
        pos_mask = y_true == pos_label
        self.pos_scores.extend(y_scores[pos_mask].tolist())
        self.neg_scores.extend(y_scores[~pos_mask].tolist())
        self.history.append(self.result())
        return self

    def result(self):
        """
        Current AUC over all data seen so far.

        Uses the Mann-Whitney U formulation: AUC = U / (n_pos * n_neg)
        where U counts the number of (pos, neg) pairs where the positive
        score is higher.

        Returns
        -------
        float, or 0.5 if not enough data yet.
        """
        if len(self.pos_scores) == 0 or len(self.neg_scores) == 0:
            return 0.5

        pos = np.asarray(self.pos_scores)
        neg = np.asarray(self.neg_scores)
        n_pos = len(pos)
        n_neg = len(neg)

        # vectorised U statistic: for each pos score count neg scores below it
        # pos[:, None] > neg[None, :] gives an (n_pos, n_neg) boolean matrix
        U = float(np.sum(pos[:, None] > neg[None, :]))
        U += 0.5 * float(np.sum(pos[:, None] == neg[None, :]))
        return U / (n_pos * n_neg)

    def reset(self):
        """Clear all accumulated scores."""
        self.pos_scores = []
        self.neg_scores = []
        self.history = []
        return self
