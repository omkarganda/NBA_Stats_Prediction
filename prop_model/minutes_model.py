"""
MinutesModel: learn a predictive distribution for minutes.

Baseline approach:
- Fit three GradientBoostingRegressor models: median (0.5 quantile), lower (0.1), upper (0.9).
- Use features that are easy to compute from public data (see feature scripts).

This is a pragmatic baseline; you can swap in LightGBM/QuantileRegressor or
hierarchical/time-varying models later while keeping the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


QUANTILES = {"p10": 0.1, "p50": 0.5, "p90": 0.9}


@dataclass
class MinutesModel:
    lower: GradientBoostingRegressor
    median: GradientBoostingRegressor
    upper: GradientBoostingRegressor
    feature_names: Optional[list] = None

    @classmethod
    def train(cls, X, y, feature_names: Optional[list] = None) -> "MinutesModel":
        models: Dict[str, GradientBoostingRegressor] = {}
        for name, q in QUANTILES.items():
            gb = GradientBoostingRegressor(loss="quantile", alpha=q, random_state=42)
            gb.fit(X, y)
            models[name] = gb
        return cls(
            lower=models["p10"],
            median=models["p50"],
            upper=models["p90"],
            feature_names=feature_names,
        )

    def predict_distribution(self, X) -> Dict[str, np.ndarray]:
        """Return dict with p10, p50, p90 arrays."""
        return {
            "p10": self.lower.predict(X),
            "p50": self.median.predict(X),
            "p90": self.upper.predict(X),
        }

    def predict_mean_std(self, X) -> Dict[str, np.ndarray]:
        """
        Convert quantile predictions into mean/std approximation
        by assuming a symmetric distribution (rough approximation).
        """
        q = self.predict_distribution(X)
        mean = q["p50"]
        std = (q["p90"] - q["p10"]) / 2.56  # approx 10th->90th span of Normal
        std = np.maximum(std, 1e-3)
        return {"mean": mean, "std": std}

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "lower": self.lower,
                "median": self.median,
                "upper": self.upper,
                "feature_names": self.feature_names,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "MinutesModel":
        obj = joblib.load(path)
        return cls(
            lower=obj["lower"],
            median=obj["median"],
            upper=obj["upper"],
            feature_names=obj.get("feature_names"),
        )
