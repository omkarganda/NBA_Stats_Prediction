"""
StatModel: Negative Binomial GLM for count-like stats (PTS/REB/AST/3PM).

Assumptions:
- Input features X (design matrix) already include useful covariates.
- Optional offset (e.g., log(minutes) or log(possessions)).
- Uses statsmodels GLM with NB family and log link.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class StatModel:
    model: sm.GLM
    result: sm.GLM
    feature_names: list
    offset_col: Optional[str] = None

    @classmethod
    def train(
        cls,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: list,
        offset_col: Optional[str] = None,
    ) -> "StatModel":
        y = df[target_col].astype(float).values
        X = df[feature_cols].astype(float)
        X = sm.add_constant(X, has_constant="add")
        offset = None
        if offset_col:
            offset = df[offset_col].astype(float).values
        glm_nb = sm.GLM(
            y,
            X,
            family=sm.families.NegativeBinomial(),
            offset=offset,
        )
        res = glm_nb.fit()
        return cls(model=glm_nb, result=res, feature_names=feature_cols, offset_col=offset_col)

    def predict_mean(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_names].astype(float)
        X = sm.add_constant(X, has_constant="add")
        offset = None
        if self.offset_col:
            offset = df[self.offset_col].astype(float).values
        mu = self.result.predict(X, offset=offset)
        return mu

    def predict_nb_params(self, df: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """
        Returns predicted mean and overdispersion alpha (scalar from fitted model).
        """
        mu = self.predict_mean(df)
        # statsmodels stores alpha in scale for NB GLM
        alpha = float(self.result.scale)
        return mu, alpha

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "model": self.model,
                "result": self.result,
                "feature_names": self.feature_names,
                "offset_col": self.offset_col,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "StatModel":
        obj = joblib.load(path)
        return cls(
            model=obj["model"],
            result=obj["result"],
            feature_names=obj["feature_names"],
            offset_col=obj.get("offset_col"),
        )
