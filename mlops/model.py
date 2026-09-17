"""A tiny logistic-regression classifier in pure Python (no numpy)."""
from __future__ import annotations
import math


class LogisticRegression:
    def __init__(self, lr=0.1, epochs=300, n_features=2):
        self.w = [0.0] * n_features
        self.b = 0.0
        self.lr = lr
        self.epochs = epochs

    @staticmethod
    def _sigmoid(z):
        z = max(-30.0, min(30.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def _p(self, x):
        return self._sigmoid(self.b + sum(w * xi for w, xi in zip(self.w, x)))

    def fit(self, X, y):
        if X and len(self.w) != len(X[0]):
            self.w = [0.0] * len(X[0])
        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                err = self._p(xi) - yi
                self.w = [w - self.lr * err * x for w, x in zip(self.w, xi)]
                self.b -= self.lr * err
        return self

    def predict(self, X):
        return [1 if self._p(x) >= 0.5 else 0 for x in X]

    def accuracy(self, X, y):
        pred = self.predict(X)
        return sum(1 for a, b in zip(pred, y) if a == b) / max(1, len(y))
