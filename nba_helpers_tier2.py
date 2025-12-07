"""
nba_tier2_model.py

Tier-2 NBA player stat model:

- Pulls multi-season game logs with nba_api
- Parses minutes as numeric
- Adds PTS, REB, AST, FG3M (3PM), PRA
- Applies recency weights to games
- Fits a weighted linear regression: stat ~ minutes
- Assumes Normal residuals for distribution
- Computes over/under probabilities & EV vs book odds

Requires:
    pip install nba_api pandas numpy
"""

from typing import List, Dict, Any, Optional, Literal
from math import erf, sqrt

import numpy as np
import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------

def find_player_id(name: str) -> dict:
    """Fuzzy search for a player by name and return best match info dict."""
    cand = players.find_players_by_full_name(name)
    if not cand:
        raise ValueError(f"No player found for name: {name}")
    return cand[0]


def get_player_gamelog(
    player_name: str,
    season: str,
    season_type: str = "Regular Season",
) -> pd.DataFrame:
    """Return game log DF for a player in a single season."""
    info = find_player_id(player_name)
    gl = playergamelog.PlayerGameLog(
        player_id=info["id"],
        season=season,
        season_type_all_star=season_type,
    )
    df = gl.get_data_frames()[0].copy()
    df["SEASON"] = season
    return df


def get_player_gamelog_multi_seasons(
    player_name: str,
    seasons: List[str],
    season_type: str = "Regular Season",
) -> pd.DataFrame:
    """
    Concatenate game logs from multiple seasons for a player.
    Sorted by date ascending.
    """
    frames = []
    for season in seasons:
        df_season = get_player_gamelog(player_name, season, season_type)
        frames.append(df_season)
    if not frames:
        raise ValueError("No seasons provided or no data returned.")
    df_all = pd.concat(frames, ignore_index=True)

    # Normalize GAME_DATE and sort
    df_all["GAME_DATE"] = pd.to_datetime(df_all["GAME_DATE"])
    df_all = df_all.sort_values("GAME_DATE").reset_index(drop=True)
    return df_all


# ---------------------------------------------------------------------
# Minutes & stat engineering
# ---------------------------------------------------------------------

def _parse_minutes(min_val) -> float:
    """Convert NBA API MIN field to numeric minutes as float."""
    if pd.isna(min_val):
        return 0.0

    if isinstance(min_val, (int, float)):
        return float(min_val)

    s = str(min_val)
    if ":" in s:
        m, sec = s.split(":")
        return int(m) + int(sec) / 60.0

    try:
        return float(s)
    except ValueError:
        return 0.0


def add_numeric_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """Add MIN_NUM column (minutes as float)."""
    df = df.copy()
    df["MIN_NUM"] = df["MIN"].apply(_parse_minutes)
    return df


def add_basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DF has:

        - PTS
        - REB
        - AST
        - FG3M  (3-pointers made)
        - PRA = PTS + REB + AST

    Returns a copy.
    """
    df = df.copy()
    # playergamelog already has PTS, REB, AST, FG3M
    if "PRA" not in df.columns:
        df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    return df


# ---------------------------------------------------------------------
# Recency weighting
# ---------------------------------------------------------------------

def add_recency_weights(
    df: pd.DataFrame,
    half_life_games: float = 20.0,
    weight_col: str = "RECENCY_W",
) -> pd.DataFrame:
    """
    Add an exponentially decaying recency weight.

    - Oldest game gets smallest weight
    - Newest game gets weight 1.0
    - half_life_games ~ number of games over which weight halves

    We operate purely on row index (chronological order), which is
    usually fine for this use case.
    """
    df = df.sort_values("GAME_DATE").reset_index(drop=True).copy()
    n = len(df)
    if n == 0:
        raise ValueError("Empty DataFrame in add_recency_weights.")

    # index 0 = oldest, n-1 = newest
    idx = np.arange(n)
    age_from_newest = (n - 1) - idx  # 0 for newest, n-1 for oldest

    # Exponential decay: w = 0.5 ** (age / half_life)
    w = 0.5 ** (age_from_newest / half_life_games)

    # Normalize to mean 1 for numerical stability (optional)
    w = w / w.mean()

    df[weight_col] = w
    return df


# ---------------------------------------------------------------------
# Odds / probability utilities
# ---------------------------------------------------------------------

def american_to_prob(odds: int) -> float:
    """Convert American odds to implied probability (no vig)."""
    if odds == 0:
        raise ValueError("Odds cannot be 0.")
    if odds < 0:
        return (-odds) / ((-odds) + 100)
    else:
        return 100 / (odds + 100)


def prob_to_american(p: float) -> int:
    """Convert probability to fair American odds."""
    if not (0 < p < 1):
        raise ValueError("Probability must be between 0 and 1.")
    profit_per_unit = (1 / p) - 1  # decimal_odds - 1
    if profit_per_unit >= 1:
        american = 100 * profit_per_unit
    else:
        american = -100 / profit_per_unit
    return int(round(american))


def expected_value_per_unit(p: float, odds: int) -> float:
    """
    Expected profit per 1 unit staked given:
      - p: win probability
      - odds: American odds
    """
    if odds < 0:
        profit_if_win = 100 / (-odds)
    else:
        profit_if_win = odds / 100.0
    return p * profit_if_win - (1 - p)


def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    """Normal CDF using error function."""
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / sigma
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def normal_tail_prob(
    line: float,
    mu: float,
    sigma: float,
    direction: Literal["over", "under"] = "over",
    continuity: bool = True,
) -> float:
    """
    P(stat > line) or P(stat < line) under Normal(mu, sigma^2).

    continuity=True and integer line -> applies +/- 0.5 correction.
    """
    if continuity and float(line).is_integer():
        threshold = line + (0.5 if direction == "over" else -0.5)
    else:
        threshold = line

    if direction == "over":
        # P(X > L) = 1 - F(L)
        return 1.0 - _normal_cdf(threshold, mu, sigma)
    else:
        # P(X < L) = F(L)
        return _normal_cdf(threshold, mu, sigma)


# ---------------------------------------------------------------------
# Minutes-aware regression
# ---------------------------------------------------------------------

STAT_MAP = {
    "points": "PTS",
    "rebounds": "REB",
    "assists": "AST",
    "threes": "FG3M",
    "3pm": "FG3M",
    "pra": "PRA",
}


def prepare_player_dataset(
    player_name: str,
    seasons: List[str],
    season_type: str = "Regular Season",
    half_life_games: float = 20.0,
) -> pd.DataFrame:
    """
    Pull multi-season logs and prepare:

      - GAME_DATE as datetime
      - MIN_NUM
      - PTS, REB, AST, FG3M, PRA
      - RECENCY_W

    Returns chronological DF.
    """
    df = get_player_gamelog_multi_seasons(player_name, seasons, season_type)
    df = add_numeric_minutes(df)
    df = add_basic_stats(df)
    df = add_recency_weights(df, half_life_games=half_life_games, weight_col="RECENCY_W")
    return df


def fit_minutes_model(
    df: pd.DataFrame,
    stat_col: str,
    min_minutes: float = 10.0,
    min_games: int = 12,
    weight_col: str = "RECENCY_W",
) -> Dict[str, Any]:
    """
    Fit a weighted linear model:

        stat = alpha + beta * minutes + epsilon

    using weighted least squares with recency weights.

    Returns dict with:
        - alpha, beta
        - sigma (residual std dev)
        - n_games_used
    """
    df = df.copy()
    df = df[df["MIN_NUM"] >= min_minutes].reset_index(drop=True)

    if len(df) < min_games:
        raise ValueError(
            f"Not enough games ({len(df)}) after filtering by min_minutes={min_minutes} "
            f"to fit model for {stat_col}."
        )

    y = df[stat_col].values.astype(float)
    m = df["MIN_NUM"].values.astype(float)
    w = df[weight_col].values.astype(float)

    # Design matrix X: [1, MIN_NUM]
    X = np.column_stack([np.ones_like(m), m])

    # Weighted Least Squares solution:
    # beta_hat = (X^T W X)^(-1) X^T W y
    W = np.diag(w)
    XtW = X.T @ W
    XtWX = XtW @ X
    XtWy = XtW @ y

    beta_hat = np.linalg.inv(XtWX) @ XtWy
    alpha, beta = beta_hat

    # Residuals & weighted sigma
    y_pred = X @ beta_hat
    resid = y - y_pred
    # Weighted MSE
    mse = np.average(resid ** 2, weights=w)
    sigma = float(np.sqrt(mse))

    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "sigma": sigma,
        "n_games_used": int(len(df)),
    }


def predict_stat_distribution(
    minutes: float,
    model_params: Dict[str, Any],
) -> Dict[str, float]:
    """
    Given minutes and a fitted minutes-model, return Normal(mu, sigma).
    """
    alpha = model_params["alpha"]
    beta = model_params["beta"]
    sigma = model_params["sigma"]

    mu = alpha + beta * minutes
    return {"mu": float(mu), "sigma": float(sigma)}


# ---------------------------------------------------------------------
# High-level evaluation for a stat (points / rebounds / assists / threes / PRA)
# ---------------------------------------------------------------------

def evaluate_tier2_stat_bet(
    player_name: str,
    stat_type: Literal["points", "rebounds", "assists", "threes", "3pm", "pra"],
    line: float,
    odds: int,
    expected_minutes: float,
    seasons: List[str],
    season_type: str = "Regular Season",
    half_life_games: float = 20.0,
    min_minutes_for_fit: float = 10.0,
    min_games_for_fit: int = 12,
    direction: Literal["over", "under"] = "over",
    opponent_factor: float = 1.0,
) -> Dict[str, Any]:
    """
    Full pipeline for one stat:

      - Pull multi-season data
      - Build minutes-aware, recency-weighted Normal model
      - Predict mean & sigma at expected_minutes
      - Optionally apply a rough opponent adjustment on the mean:
            mu_adj = opponent_factor * mu
      - Compute P(stat > line) or P(stat < line)
      - Compute edge vs book and EV per unit

    opponent_factor:
        Rough multiplier on the mean (μ) to reflect matchup (pace/defense).
        Example:
            1.05 -> +5% expected production (fast pace / weak defense)
            0.95 -> -5% expected production (slow pace / strong defense)
    """
    stat_col = STAT_MAP[stat_type.lower()]  # map to DF column

    # Prepare dataset
    df = prepare_player_dataset(
        player_name=player_name,
        seasons=seasons,
        season_type=season_type,
        half_life_games=half_life_games,
    )

    # Fit minutes model for this stat
    model_params = fit_minutes_model(
        df,
        stat_col=stat_col,
        min_minutes=min_minutes_for_fit,
        min_games=min_games_for_fit,
        weight_col="RECENCY_W",
    )

    # Predict distribution at expected_minutes
    dist = predict_stat_distribution(expected_minutes, model_params)
    mu_raw, sigma = dist["mu"], dist["sigma"]

    # Apply opponent adjustment to mean only (rough hack)
    mu_adj = mu_raw * float(opponent_factor)

    # Probability of winning the bet (over/under) with adjusted mean
    p_model = normal_tail_prob(
        line=line,
        mu=mu_adj,
        sigma=sigma,
        direction=direction,
        continuity=True,
    )

    # Book's implied probability
    p_book = american_to_prob(odds)

    # Edge & fair odds
    edge = p_model - p_book
    fair_odds = prob_to_american(p_model)
    ev = expected_value_per_unit(p_model, odds)

    return {
        "player_name": player_name,
        "stat_type": stat_type,
        "stat_column": stat_col,
        "seasons_used": seasons,
        "season_type": season_type,
        "line": line,
        "direction": direction,
        "odds": odds,
        "expected_minutes": expected_minutes,
        "opponent_factor": opponent_factor,
        "model_params": model_params,
        "distribution": {
            "mu_raw": mu_raw,
            "mu_adjusted": mu_adj,
            "sigma": sigma,
        },
        "probabilities": {
            "p_model_win": p_model,
            "p_book_win_implied": p_book,
        },
        "analytics": {
            "edge_prob_points": edge,
            "fair_odds": fair_odds,
            "ev_per_unit": ev,
        },
    }
