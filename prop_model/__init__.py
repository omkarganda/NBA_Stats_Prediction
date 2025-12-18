# Core library for NBA prop-betting pipeline.
# Modules are intentionally small and focused; scripts under scripts/ orchestrate them.

from .devig import devig_all, implied_prob_american, american_from_prob
from .minutes_model import MinutesModel
from .stat_model import StatModel
from .simulation import simulate_prop_probability
from .evaluation import (
    brier_score,
    log_loss_score,
    reliability_bins,
    edge_metrics,
)

__all__ = [
    "devig_all",
    "implied_prob_american",
    "american_from_prob",
    "MinutesModel",
    "StatModel",
    "simulate_prop_probability",
    "brier_score",
    "log_loss_score",
    "reliability_bins",
    "edge_metrics",
]
