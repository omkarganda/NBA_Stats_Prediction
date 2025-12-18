"""
Devigging methods and odds utilities.

Implements common devig approaches for two-outcome prop markets (over/under):
- multiplicative
- additive
- shin
- power (defaults to exponent=0.5)

All functions expect American odds inputs for over/under and return vig-free probs.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple


def implied_prob_american(odds: float) -> float:
    """Convert American odds to implied probability (includes vig)."""
    if odds == 0:
        raise ValueError("Odds cannot be zero.")
    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    return 100.0 / (odds + 100.0)


def american_from_prob(p: float) -> int:
    """Convert probability (0,1) to American odds."""
    if not 0 < p < 1:
        raise ValueError("Probability must be in (0,1).")
    profit = (1.0 / p) - 1.0
    if profit >= 1.0:
        return int(round(100.0 * profit))
    return int(round(-100.0 / profit))


def _normalize_probs(p_over: float, p_under: float) -> Tuple[float, float]:
    total = p_over + p_under
    if total <= 0:
        raise ValueError("Non-positive total probability; check odds inputs.")
    return p_over / total, p_under / total


def devig_multiplicative(odds_over: float, odds_under: float) -> Dict[str, float]:
    """
    Spread vig proportionally (standard multiplicative normalization).
    """
    p_over = implied_prob_american(odds_over)
    p_under = implied_prob_american(odds_under)
    p_over_nv, p_under_nv = _normalize_probs(p_over, p_under)
    return {"p_over": p_over_nv, "p_under": p_under_nv, "hold": p_over + p_under - 1.0}


def devig_additive(odds_over: float, odds_under: float) -> Dict[str, float]:
    """
    Spread vig by subtracting half the overround equally.
    """
    p_over = implied_prob_american(odds_over)
    p_under = implied_prob_american(odds_under)
    overround = p_over + p_under - 1.0
    adj = overround / 2.0
    p_over_nv = max(1e-9, p_over - adj)
    p_under_nv = max(1e-9, p_under - adj)
    p_over_nv, p_under_nv = _normalize_probs(p_over_nv, p_under_nv)
    return {"p_over": p_over_nv, "p_under": p_under_nv, "hold": overround}


def devig_shin(odds_over: float, odds_under: float) -> Dict[str, float]:
    """
    Shin method (accounts for favorite-longshot bias). Closed-form for 2 outcomes.
    """
    p1 = implied_prob_american(odds_over)
    p2 = implied_prob_american(odds_under)
    s = p1 + p2
    if s <= 1:
        return {"p_over": p1, "p_under": p2, "hold": s - 1.0}
    # Solve for z (insider trading parameter)
    # z = 2*(s-1)/(1 + sqrt(1 - 4*p1*p2/(s**2)))
    denom = s * s
    inner = 1.0 - (4.0 * p1 * p2) / denom
    inner = max(inner, 1e-9)
    z = 2.0 * (s - 1.0) / (1.0 + math.sqrt(inner))
    p_over_nv = (p1 - z / 2.0) / (1.0 - z)
    p_under_nv = (p2 - z / 2.0) / (1.0 - z)
    p_over_nv, p_under_nv = _normalize_probs(p_over_nv, p_under_nv)
    return {"p_over": p_over_nv, "p_under": p_under_nv, "hold": s - 1.0}


def devig_power(odds_over: float, odds_under: float, exponent: float = 0.5) -> Dict[str, float]:
    """
    Power method: raise implied probs to exponent before normalizing.
    exponent < 1 reduces favorite-longshot bias allocation.
    """
    p_over = implied_prob_american(odds_over)
    p_under = implied_prob_american(odds_under)
    p_over_pow = p_over**exponent
    p_under_pow = p_under**exponent
    p_over_nv, p_under_nv = _normalize_probs(p_over_pow, p_under_pow)
    return {"p_over": p_over_nv, "p_under": p_under_nv, "hold": p_over + p_under - 1.0}


def devig_all(odds_over: float, odds_under: float) -> Dict[str, Dict[str, float]]:
    """
    Convenience helper returning all devig variants.
    """
    return {
        "multiplicative": devig_multiplicative(odds_over, odds_under),
        "additive": devig_additive(odds_over, odds_under),
        "shin": devig_shin(odds_over, odds_under),
        "power": devig_power(odds_over, odds_under),
    }
