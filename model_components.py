"""Fold-local imbalance weighting shared by training and inference."""
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.class_weight import compute_sample_weight


class FoldWeightedClassifier(ClassifierMixin, BaseEstimator):
    """Compute imbalance weights from the current training fold only."""
    def __init__(self, estimator, balanced=False):
        self.estimator = estimator
        self.balanced = balanced

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        weights = compute_sample_weight('balanced', y) if self.balanced else None
        self.estimator_.fit(X, y, sample_weight=weights)
        self.classes_ = self.estimator_.classes_
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
