"""
Simple ensemble regressor wrapper.

Stores a list of underlying regressors and a set of weights,
and exposes a scikit-learn-like `predict` method that returns
the weighted average of the base model predictions.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np


class EnsembleRegressor:
    def __init__(self, models: Sequence, weights: Iterable[float] | None = None):
        if not models:
            raise ValueError("EnsembleRegressor requires at least one base model.")
        self.models: List = list(models)

        if weights is None:
            w = np.ones(len(self.models), dtype="float64")
        else:
            w = np.asarray(list(weights), dtype="float64")
            if w.shape[0] != len(self.models):
                raise ValueError("weights length must match number of models.")

        # Normalize weights to sum to 1
        total = np.sum(w)
        if total <= 0:
            raise ValueError("Sum of weights must be positive.")
        self.weights = w / total

    def predict(self, X):
        """
        Predict by taking a weighted average over all base model predictions.
        """
        preds = []
        for m in self.models:
            preds.append(m.predict(X))
        # Shape: (n_models, n_samples)
        stacked = np.vstack(preds)
        # Weighted average over models axis
        return np.dot(self.weights, stacked)

