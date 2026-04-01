from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


class FocalLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Practical focal-like classifier for tabular imbalance.

    It first fits a balanced logistic model, estimates sample hardness,
    then refits a second logistic model with focal-style sample weights.
    """

    def __init__(self, C: float = 1.0, gamma: float = 2.0, max_iter: int = 2000, random_state: int | None = None):
        self.C = C
        self.gamma = gamma
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            raise ValueError("FocalLogisticRegression requires at least two classes.")

        warmup = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver="liblinear",
            class_weight="balanced",
            random_state=self.random_state,
        )
        warmup.fit(X, y)
        proba = warmup.predict_proba(X)[:, 1]
        pt = np.where(y == 1, proba, 1.0 - proba)

        class_weight = {
            0: counts.max() / max(counts[0], 1),
            1: counts.max() / max(counts[1], 1),
        }
        alpha_t = np.where(y == 1, class_weight[1], class_weight[0]).astype(float)
        focal_weights = alpha_t * np.power(1.0 - np.clip(pt, 1e-6, 1.0 - 1e-6), self.gamma)

        self.model_ = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver="liblinear",
            random_state=self.random_state,
        )
        self.model_.fit(X, y, sample_weight=focal_weights)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def decision_function(self, X):
        return self.model_.decision_function(X)

    def predict(self, X):
        return self.model_.predict(X)
