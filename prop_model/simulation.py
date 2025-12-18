"""
Simulation helpers to turn minutes and per-minute rates into prop win probabilities.
"""

from __future__ import annotations

import numpy as np
from typing import Literal, Tuple


def nb2_params(mean: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert NB2 parameterization (mean, alpha) to (r, p) for numpy negative binomial.
    Var = mean + alpha * mean^2
    r = 1/alpha, p = 1 / (1 + alpha * mean)
    """
    p = 1.0 / (1.0 + alpha * mean)
    r = 1.0 / alpha
    return r, p


def sample_nb(mean: np.ndarray, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Sample from NB2 given mean and alpha."""
    r, p = nb2_params(mean, alpha)
    return rng.negative_binomial(r, p)


def simulate_prop_probability(
    minutes_mean: float,
    minutes_std: float,
    rate_per_min: float,
    alpha: float,
    line: float,
    direction: Literal["over", "under"] = "over",
    num_sims: int = 5000,
    seed: int = 42,
) -> dict:
    """
    Simulate win probability for a prop by sampling minutes and NB stat counts.

    Args:
        minutes_mean: expected minutes mean.
        minutes_std: expected minutes std (use minutes model quantiles to derive).
        rate_per_min: expected stat per minute (mu/min).
        alpha: NB overdispersion (from stat model fit).
        line: sportsbook line.
        direction: "over" or "under".
        num_sims: Monte Carlo samples.
    """
    rng = np.random.default_rng(seed)
    minutes = rng.normal(minutes_mean, minutes_std, size=num_sims)
    minutes = np.clip(minutes, 0.0, 48.0)

    mu = minutes * rate_per_min
    # Avoid zero mean instability
    mu = np.maximum(mu, 1e-4)

    samples = sample_nb(mu, alpha, rng)

    if direction == "over":
        win = samples > line
    else:
        win = samples < line

    p_win = float(win.mean())
    return {
        "p_model_win": p_win,
        "mu_samples": float(mu.mean()),
        "stat_mean_sim": float(samples.mean()),
        "minutes_mean_sim": float(minutes.mean()),
    }
