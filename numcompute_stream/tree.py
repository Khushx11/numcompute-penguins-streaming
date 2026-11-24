"""
tree.py

Decision tree classifier with partial_fit support for streaming data.
Grows the tree incrementally by re-splitting nodes when new data arrives.
Supports Gini impurity and entropy as split criteria.
"""

import numpy as np


def _gini(y):
    """
    Gini impurity for a label array.

    Returns 0.0 for empty arrays or pure nodes.
    """
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y.astype(int))
    probs = counts / len(y)
    return 1.0 - float(np.sum(probs ** 2))


def _entropy(y):
    """
    Shannon entropy for a label array.

    Clips probabilities away from zero before log to avoid nan.
    """
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y.astype(int))
    probs = counts / len(y)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _best_split(X, y, criterion_fn, max_features=None, rng=None):
    """
    Find the best (feature, threshold) split for a node.

    Tries all unique midpoint thresholds for each candidate feature
    and picks the one that gives the lowest weighted impurity after
    the split.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    criterion_fn : callable
        _gini or _entropy.
    max_features : int or None
        How many features to consider. None means all of them.
    rng : np.random.Generator or None
        For reproducible feature subsampling.

    Returns
    -------
    best_feat : int or None
        Feature index of best split, None if no valid split found.
    best_thresh : float or None
        Threshold value for the split.
    best_gain : float
        Information gain of the best split.
    """
    n_samples, n_features = X.shape
    if n_samples < 2:
        return None, None, 0.0

    # pick which features to evaluate
    feat_indices = np.arange(n_features)
    if max_features is not None and max_features < n_features:
        if rng is None:
            rng = np.random.default_rng()
        feat_indices = rng.choice(n_features, size=max_features, replace=False)

    base_impurity = criterion_fn(y)
    best_feat = None
    best_thresh = None
    best_gain = -np.inf

    for feat in feat_indices:
        col = X[:, feat]
        unique_vals = np.unique(col)
        if len(unique_vals) < 2:
            # zero variance feature, nothing to split on
            continue
        # midpoints between consecutive unique values as candidate thresholds
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

        for thresh in thresholds:
            left_mask = col <= thresh
            right_mask = ~left_mask
            n_left = left_mask.sum()
            n_right = right_mask.sum()
            if n_left == 0 or n_right == 0:
                continue
            weighted_impurity = (
                n_left / n_samples * criterion_fn(y[left_mask])
                + n_right / n_samples * criterion_fn(y[right_mask])
            )
            gain = base_impurity - weighted_impurity
            if gain > best_gain:
                best_gain = gain
                best_feat = feat
                best_thresh = thresh

    return best_feat, best_thresh, best_gain


class _Node:
    """
    A single node in the decision tree.

    Either a split node (has feature/threshold/children) or a leaf
    (has a class prediction).
    """

    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.is_leaf = False
        self.prediction = None
        # store class counts at this node so we can update incrementally
        self.class_counts = None
        self.n_samples = 0


class DecisionTreeClassifier:
    """
    Depth-limited decision tree that supports partial_fit for streaming.

    On each partial_fit call the tree is rebuilt from scratch using all
    data accumulated so far. This is the simplest correct approach for
    an incremental tree since proper online tree-growing algorithms
    (like Hoeffding trees) are significantly more complex.

    The accumulated data buffer is kept in memory, so this works best
    for datasets that fit in RAM. For the streaming simulation use case
    in this assignment that is fine.

    Parameters
    ----------
    max_depth : int, default 5
    min_samples_split : int, default 2
        Minimum samples needed at a node to attempt a split.
    criterion : str, default 'gini'
        'gini' or 'entropy'.
    max_features : int or None, default None
        Features to consider per split. None means all features.
        Set this to sqrt(n_features) for Random Forest usage.
    random_state : int or None, default None
    """

    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        criterion="gini",
        max_features=None,
        random_state=None,
    ):
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self._criterion_fn = _gini if criterion == "gini" else _entropy
        self.root = None
        self.classes_ = None
        self.fitted_ = False
        # buffer stores all data seen across partial_fit calls
        self._X_buf = None
        self._y_buf = None

    def partial_fit(self, X, y, classes=None):
        """
        Update the tree with a new chunk of data.

        Appends the chunk to the internal buffer then rebuilds the
        tree from scratch. classes can be passed on the first call
        to declare all expected labels upfront.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        classes : array-like or None
            All possible class labels. Useful when early chunks may
            not contain all classes.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")

        # handle NaNs in X by replacing with column means from current buffer
        if np.isnan(X).any():
            col_means = np.nanmean(
                X if self._X_buf is None else np.vstack([self._X_buf, X]),
                axis=0
            )
            nan_mask = np.isnan(X)
            X = X.copy()
            X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        # accumulate data
        if self._X_buf is None:
            self._X_buf = X.copy()
            self._y_buf = y.copy()
        else:
            self._X_buf = np.vstack([self._X_buf, X])
            self._y_buf = np.concatenate([self._y_buf, y])

        # update known classes
        seen = np.unique(self._y_buf)
        if self.classes_ is not None:
            seen = np.union1d(seen, self.classes_)
        if classes is not None:
            seen = np.unique(np.concatenate([seen, np.asarray(classes, dtype=int)]))
        self.classes_ = seen

        # rebuild tree from accumulated buffer
        encoded_y = np.searchsorted(self.classes_, self._y_buf)
        self.root = self._build(self._X_buf, encoded_y, depth=0)
        self.fitted_ = True
        return self

    def fit(self, X, y, classes=None):
        """Full fit on a single dataset. Clears any previous state."""
        self._X_buf = None
        self._y_buf = None
        self.fitted_ = False
        self.classes_ = None
        return self.partial_fit(X, y, classes=classes)

    def _build(self, X, y, depth):
        """
        Recursively build the tree from X and y.

        Returns a _Node that is either a leaf or a split node.
        """
        node = _Node()
        node.n_samples = len(y)
        counts = np.bincount(y.astype(int), minlength=len(self.classes_))
        node.class_counts = counts

        # leaf conditions: max depth hit, too few samples, or pure node
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            node.is_leaf = True
            node.prediction = int(self.classes_[np.argmax(counts)])
            return node

        feat, thresh, gain = _best_split(
            X, y, self._criterion_fn, self.max_features, self._rng
        )

        # no valid split found, make this a leaf
        if feat is None or gain <= 0:
            node.is_leaf = True
            node.prediction = int(self.classes_[np.argmax(counts)])
            return node

        node.feature = feat
        node.threshold = thresh
        left_mask = X[:, feat] <= thresh
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)

        Raises
        ------
        RuntimeError
            If called before partial_fit.
        """
        if not self.fitted_:
            raise RuntimeError("call partial_fit before predict")
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x, self.root) for x in X])

    def _predict_one(self, x, node):
        """Walk the tree for a single sample."""
        if node.is_leaf:
            return node.prediction
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict_proba(self, X):
        """
        Class probability estimates based on leaf class counts.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
        """
        if not self.fitted_:
            raise RuntimeError("call partial_fit before predict_proba")
        X = np.asarray(X, dtype=float)
        n_classes = len(self.classes_)
        proba = np.zeros((len(X), n_classes))
        for i, x in enumerate(X):
            node = self.root
            while not node.is_leaf:
                if x[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            counts = node.class_counts[:n_classes].astype(float)
            total = counts.sum()
            proba[i] = counts / total if total > 0 else np.ones(n_classes) / n_classes
        return proba

    def get_depth(self):
        """
        Actual depth of the fitted tree.

        Returns
        -------
        int
        """
        if self.root is None:
            return 0
        return self._depth(self.root)

    def _depth(self, node):
        """Recursively compute tree depth."""
        if node.is_leaf:
            return 0
        return 1 + max(self._depth(node.left), self._depth(node.right))
