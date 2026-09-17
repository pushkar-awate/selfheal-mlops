"""Synthetic binary-classification data with controllable feature drift.

The label depends on the *latent* signal (lat1 + lat2 > 0), but we only observe
features shifted by `drift`. A model trained at drift=0 therefore degrades as
drift grows - until it is retrained on freshly labeled, drifted data.
"""
from __future__ import annotations
import random


def make_batch(n=300, drift=0.0, seed=0):
    r = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        lat1, lat2 = r.gauss(0, 1), r.gauss(0, 1)
        label = 1 if (lat1 + lat2) > 0 else 0
        X.append([lat1 + drift, lat2 + drift])   # observed features are shifted
        y.append(label)
    return X, y


def feature_means(X):
    cols = list(zip(*X))
    return [sum(c) / len(c) for c in cols]


def drift_score(X, baseline_means):
    """Mean absolute shift of feature means vs the training baseline."""
    m = feature_means(X)
    return sum(abs(a - b) for a, b in zip(m, baseline_means)) / len(m)
