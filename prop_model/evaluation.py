"""
Evaluation utilities for probability forecasts.
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss


def brier_score(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    return float(brier_score_loss(y_true, p_pred))


def log_loss_score(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    return float(log_loss(y_true, p_pred, eps=1e-9))


def reliability_bins(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10):
    prob_true, prob_pred = calibration_curve(y_true, p_pred, n_bins=n_bins, strategy="quantile")
    return prob_true, prob_pred


def edge_metrics(
    y_true: np.ndarray,
    p_model: np.ndarray,
    p_market: np.ndarray,
):
    """
    Compute realized ROI vs model and market-implied probabilities.
    Assumes stake=1, American odds not needed (pure prob ROI using fair odds = 1/p).
    """
    fair_odds_model = 1.0 / np.clip(p_model, 1e-6, 1 - 1e-6)
    fair_odds_mkt = 1.0 / np.clip(p_market, 1e-6, 1 - 1e-6)

    profit_model = y_true * (fair_odds_model - 1.0) - (1 - y_true)
    profit_mkt = y_true * (fair_odds_mkt - 1.0) - (1 - y_true)

    return {
        "roi_model": float(profit_model.mean()),
        "roi_market": float(profit_mkt.mean()),
    }
