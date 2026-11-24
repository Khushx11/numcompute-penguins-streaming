"""
ensemble.py

Ensemble classifiers built on top of DecisionTreeClassifier.
Supports Random Forest and Bagging, both with partial_fit for streaming.
The shared EnsembleClassifier interface means you can swap between
single trees and ensembles without changing the rest of the pipeline.
"""

import numpy as np
from numcompute_stream.tree import DecisionTreeClassifier


class RandomForestClassifier:
    """
    Random Forest: an ensemble of decision trees where each tree sees
    a bootstrap sample of the data and a random subset of features.

    Supports partial_fit so the forest can be updated chunk by chunk
    as new data arrives. Each call to partial_fit rebuilds every tree
    on the full accumulated buffer with a fresh bootstrap sample.

    Parameters
    ----------
    n_estimators : int, default 10
        Number of trees in the forest.
    max_depth : int, default 5
    min_samples_split : int, default 2
    criterion : str, default 'gini'
    max_features : int, str, or None, default 'sqrt'
        Features to consider per split in each tree.
        'sqrt' means sqrt(n_features), 'log2' means log2(n_features),
        an int is used directly, None means all features.
    random_state : int or None, default None
    """

    def __init__(
        self,
        n_estimators=10,
        max_depth=5,
        min_samples_split=2,
        criterion="gini",
        max_features="sqrt",
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self.estimators_ = []
        self.classes_ = None
        self.fitted_ = False
        self._X_buf = None
        self._y_buf = None

    def _resolve_max_features(self, n_features):
        """Work out the actual number of features to use per split."""
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        if self.max_features is None:
            return n_features
        return int(self.max_features)

    def partial_fit(self, X, y, classes=None):
        """
        Update the forest with a new chunk of data.

        Appends the chunk to the buffer then rebuilds all trees using
        bootstrap samples from the full accumulated dataset.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        classes : array-like or None

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

        # accumulate data across chunks
        if self._X_buf is None:
            self._X_buf = X.copy()
            self._y_buf = y.copy()
        else:
            self._X_buf = np.vstack([self._X_buf, X])
            self._y_buf = np.concatenate([self._y_buf, y])

        seen = np.unique(self._y_buf)
        if self.classes_ is not None:
            seen = np.union1d(seen, self.classes_)
        if classes is not None:
            seen = np.unique(np.concatenate([seen, np.asarray(classes, dtype=int)]))
        self.classes_ = seen

        n_samples, n_features = self._X_buf.shape
        mf = self._resolve_max_features(n_features)

        # rebuild all trees on fresh bootstrap samples
        self.estimators_ = []
        for i in range(self.n_estimators):
            # bootstrap sample with replacement
            indices = self._rng.integers(0, n_samples, size=n_samples)
            X_boot = self._X_buf[indices]
            y_boot = self._y_buf[indices]

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                max_features=mf,
                random_state=int(self._rng.integers(0, 2**31)),
            )
            tree.partial_fit(X_boot, y_boot, classes=self.classes_)
            self.estimators_.append(tree)

        self.fitted_ = True
        return self

    def fit(self, X, y, classes=None):
        """Full fit on a single dataset. Clears previous state."""
        self._X_buf = None
        self._y_buf = None
        self.fitted_ = False
        self.classes_ = None
        return self.partial_fit(X, y, classes=classes)

    def predict(self, X):
        """
        Majority vote across all trees.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        if not self.fitted_:
            raise RuntimeError("call partial_fit before predict")
        X = np.asarray(X, dtype=float)
        # collect predictions from every tree then take the majority vote
        all_preds = np.array([tree.predict(X) for tree in self.estimators_])
        # all_preds shape: (n_estimators, n_samples)
        n_samples = X.shape[0]
        result = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            votes = all_preds[:, i]
            counts = np.bincount(np.searchsorted(self.classes_, votes), minlength=len(self.classes_))
            result[i] = int(self.classes_[np.argmax(counts)])
        return result

    def predict_proba(self, X):
        """
        Average class probabilities across all trees.

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
        probas = np.array([tree.predict_proba(X) for tree in self.estimators_])
        # average across trees
        return probas.mean(axis=0)


class BaggingClassifier:
    """
    Bagging ensemble: like Random Forest but without feature subsampling.

    Each tree gets a bootstrap sample of the data but considers all
    features at every split. Good as a baseline to show the value of
    the extra feature randomness in Random Forest.

    Parameters
    ----------
    n_estimators : int, default 10
    max_depth : int, default 5
    min_samples_split : int, default 2
    criterion : str, default 'gini'
    random_state : int or None, default None
    """

    def __init__(
        self,
        n_estimators=10,
        max_depth=5,
        min_samples_split=2,
        criterion="gini",
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self.estimators_ = []
        self.classes_ = None
        self.fitted_ = False
        self._X_buf = None
        self._y_buf = None

    def partial_fit(self, X, y, classes=None):
        """
        Update the bagging ensemble with a new chunk of data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        classes : array-like or None

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

        if self._X_buf is None:
            self._X_buf = X.copy()
            self._y_buf = y.copy()
        else:
            self._X_buf = np.vstack([self._X_buf, X])
            self._y_buf = np.concatenate([self._y_buf, y])

        seen = np.unique(self._y_buf)
        if self.classes_ is not None:
            seen = np.union1d(seen, self.classes_)
        if classes is not None:
            seen = np.unique(np.concatenate([seen, np.asarray(classes, dtype=int)]))
        self.classes_ = seen

        n_samples = len(self._X_buf)
        self.estimators_ = []

        for i in range(self.n_estimators):
            indices = self._rng.integers(0, n_samples, size=n_samples)
            X_boot = self._X_buf[indices]
            y_boot = self._y_buf[indices]

            # no max_features restriction, that is the only difference vs RF
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                max_features=None,
                random_state=int(self._rng.integers(0, 2**31)),
            )
            tree.partial_fit(X_boot, y_boot, classes=self.classes_)
            self.estimators_.append(tree)

        self.fitted_ = True
        return self

    def fit(self, X, y, classes=None):
        """Full fit on a single dataset."""
        self._X_buf = None
        self._y_buf = None
        self.fitted_ = False
        self.classes_ = None
        return self.partial_fit(X, y, classes=classes)

    def predict(self, X):
        """
        Majority vote across all trees.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        if not self.fitted_:
            raise RuntimeError("call partial_fit before predict")
        X = np.asarray(X, dtype=float)
        all_preds = np.array([tree.predict(X) for tree in self.estimators_])
        n_samples = X.shape[0]
        result = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            votes = all_preds[:, i]
            counts = np.bincount(np.searchsorted(self.classes_, votes), minlength=len(self.classes_))
            result[i] = int(self.classes_[np.argmax(counts)])
        return result

    def predict_proba(self, X):
        """
        Average class probabilities across all trees.

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
        probas = np.array([tree.predict_proba(X) for tree in self.estimators_])
        return probas.mean(axis=0)


class EnsembleClassifier:
    """
    Unified wrapper around RandomForestClassifier and BaggingClassifier.

    Lets you switch between ensemble methods with a single parameter
    while keeping the same partial_fit/predict interface throughout
    the rest of the pipeline.

    Parameters
    ----------
    method : str, default 'random_forest'
        'random_forest' or 'bagging'.
    **kwargs
        Passed directly to the underlying classifier.

    Examples
    --------
    clf = EnsembleClassifier(method='random_forest', n_estimators=20)
    clf.partial_fit(X_chunk, y_chunk)
    preds = clf.predict(X_test)

    clf2 = EnsembleClassifier(method='bagging', n_estimators=10)
    """

    def __init__(self, method="random_forest", **kwargs):
        if method not in ("random_forest", "bagging"):
            raise ValueError("method must be 'random_forest' or 'bagging'")
        self.method = method
        if method == "random_forest":
            self._clf = RandomForestClassifier(**kwargs)
        else:
            self._clf = BaggingClassifier(**kwargs)

    def partial_fit(self, X, y, classes=None):
        """Delegate to the underlying classifier."""
        self._clf.partial_fit(X, y, classes=classes)
        return self

    def fit(self, X, y, classes=None):
        """Full fit, delegates to underlying classifier."""
        self._clf.fit(X, y, classes=classes)
        return self

    def predict(self, X):
        """Predict using the underlying classifier."""
        return self._clf.predict(X)

    def predict_proba(self, X):
        """Probability estimates from the underlying classifier."""
        return self._clf.predict_proba(X)

    @property
    def classes_(self):
        return self._clf.classes_

    @property
    def fitted_(self):
        return self._clf.fitted_
