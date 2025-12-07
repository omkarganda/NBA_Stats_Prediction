from typing import Optional, Literal, Dict, Any
from math import erf, sqrt

import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog


# -------------------------
# Basic player + gamelog helpers
# -------------------------

def find_player_id(name: str) -> dict:
    """
    Fuzzy search for a player by name and return best match info dict.

    Example:
        info = find_player_id("Nikola Jokic")
        info["id"] -> player ID
    """
    cand = players.find_players_by_full_name(name)
    if not cand:
        raise ValueError(f"No player found for name: {name}")
    return cand[0]


def get_player_gamelog(
    name: str,
    season: str = "2024-25",
    season_type: str = "Regular Season",
) -> pd.DataFrame:
    """
    Return a DataFrame of all games for a player in a given season.
    Sorted by date ascending.
    """
    player_info = find_player_id(name)
    player_id = player_info["id"]

    gl = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season,
        season_type_all_star=season_type,
    )
    df = gl.get_data_frames()[0]
    df = df.sort_values("GAME_DATE")
    return df


# -------------------------
# Minutes parsing + PRA
# -------------------------

def _parse_minutes(min_val) -> float:
    """
    Convert NBA API MIN field to numeric minutes as float.

    - If 'MIN' is like '32:15' -> 32.25
    - If it's already numeric -> cast to float.
    """
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
    """
    Ensure DF has a numeric 'MIN_NUM' column with minutes as float.
    """
    df = df.copy()
    if "MIN_NUM" not in df.columns:
        df["MIN_NUM"] = df["MIN"].apply(_parse_minutes)
    return df


def add_pra(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add PRA = PTS + REB + AST.
    """
    df = df.copy()
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    return df


# -------------------------
# Filtering helpers
# -------------------------

def filter_gamelog(
    df: pd.DataFrame,
    home: Optional[bool] = None,
    last_n: Optional[int] = None,
    min_minutes: Optional[float] = None,
) -> pd.DataFrame:
    """
    Filter a game log DataFrame by:
      - home: True for home only, False for away only, None for all
      - last_n: keep only last N games (after other filters)
      - min_minutes: keep only games with MIN_NUM >= threshold

    Assumes 'MATCHUP' and 'MIN' columns from NBA API.
    """
    df = df.copy()
    df = add_numeric_minutes(df)

    # Home / away filter from MATCHUP string:
    # "DEN vs PHO" = home, "DEN @ PHO" = away
    if home is True:
        df = df[df["MATCHUP"].str.contains(" vs ", na=False)]
    elif home is False:
        df = df[df["MATCHUP"].str.contains(" @ ", na=False)]

    if min_minutes is not None:
        df = df[df["MIN_NUM"] >= float(min_minutes)]

    # Important: keep chronological order, then take last N
    df = df.sort_values("GAME_DATE")

    if last_n is not None:
        df = df.tail(int(last_n))

    return df


# -------------------------
# Probability + odds utilities
# -------------------------

def american_to_prob(odds: int) -> float:
    """
    Convert American odds to implied break-even probability (no vig).

    Example:
        -115 -> ~0.535
        +150 -> 0.4
    """
    if odds == 0:
        raise ValueError("Odds cannot be 0.")

    if odds < 0:
        return (-odds) / ((-odds) + 100)
    else:
        return 100 / (odds + 100)


def prob_to_american(p: float) -> int:
    """
    Convert a fair probability (0<p<1) to fair American odds (rounded).

    Example:
        p = 0.6 -> around -150
        p = 0.4 -> around +150
    """
    if not (0 < p < 1):
        raise ValueError("Probability must be between 0 and 1 (exclusive).")

    # Decimal odds
    profit_per_unit = (1 / p) - 1  # decimal_odds - 1

    if profit_per_unit >= 1:  # decimal_odds >= 2.0 -> underdog -> positive odds
        american = 100 * profit_per_unit
    else:
        american = -100 / profit_per_unit

    return int(round(american))


def expected_value_per_unit(p: float, odds: int) -> float:
    """
    Expected profit per 1 unit staked given:
      - p: your win probability
      - odds: American odds

    EV = p * profit_if_win - (1 - p) * 1
    where stake = 1 unit.
    """
    if odds < 0:
        profit_if_win = 100 / (-odds)
    else:
        profit_if_win = odds / 100.0

    ev = p * profit_if_win - (1 - p) * 1.0
    return ev


def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    """
    Standard normal CDF using error function.

    This is the core math a lot of real-world models build on:
    we approximate discrete stats as Normal around mean and variance.
    """
    if sigma <= 0:
        # If no variance in data, distribution is degenerate at mu.
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / sigma
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


# -------------------------
# PRA probabilities
# -------------------------

def pra_empirical_probs(df: pd.DataFrame, line: float) -> Dict[str, float]:
    """
    Empirical over/under/push probabilities from observed PRA.

    Returns dict with:
      - p_over, p_under, p_push
      - n_games
      - mean_pra, std_pra, min_pra, max_pra
    """
    if "PRA" not in df.columns:
        raise ValueError("DataFrame must have 'PRA' column. Run add_pra() first.")

    n = len(df)
    if n == 0:
        raise ValueError("No games in DataFrame after filtering.")

    pra_vals = df["PRA"]

    over_mask = pra_vals > line
    push_mask = pra_vals == line
    under_mask = pra_vals < line

    over_count = int(over_mask.sum())
    push_count = int(push_mask.sum())
    under_count = int(under_mask.sum())

    return {
        "line": line,
        "n_games": n,
        "mean_pra": pra_vals.mean(),
        "std_pra": pra_vals.std(),
        "min_pra": pra_vals.min(),
        "max_pra": pra_vals.max(),
        "p_over": over_count / n,
        "p_under": under_count / n,
        "p_push": push_count / n,
    }


def pra_normal_over_prob(df: pd.DataFrame, line: float, continuity: bool = True) -> Dict[str, float]:
    """
    Approximate P(PRA > line) using a Normal distribution fitted to PRA.

    This mirrors what a lot of pricing models do:
    - Estimate mean and variance
    - Treat stat as Gaussian
    - Integrate tail probability over the line

    If continuity=True and line is integer, we use line + 0.5 to better
    approximate discrete PRA.
    """
    if "PRA" not in df.columns:
        raise ValueError("DataFrame must have 'PRA' column. Run add_pra() first.")

    pra_vals = df["PRA"]
    mu = pra_vals.mean()
    sigma = pra_vals.std(ddof=1)

    if continuity and float(line).is_integer():
        threshold = line + 0.5
    else:
        threshold = line

    # P(PRA > line) = 1 - CDF(threshold)
    p_over = 1.0 - _normal_cdf(threshold, mu, sigma)

    return {
        "line": line,
        "n_games": len(df),
        "mean_pra": mu,
        "std_pra": sigma,
        "p_over_normal": p_over,
    }


# -------------------------
# High-level evaluation API
# -------------------------

def evaluate_pra_bet(
    player_name: str,
    line: float,
    odds: int,
    season: str = "2024-25",
    season_type: str = "Regular Season",
    home: Optional[bool] = None,
    last_n: Optional[int] = None,
    min_minutes: Optional[float] = None,
    method: Literal["empirical", "normal", "blend"] = "blend",
) -> Dict[str, Any]:
    """
    High-level function:

    Given:
      - player_name (e.g. "Nikola Jokic")
      - PRA line (e.g. 43.5)
      - sportsbook odds for OVER in American format (e.g. -115)
      - optional filters: home/away, last_n games, min_minutes

    Returns a dict with:
      - filter_info
      - empirical stats: p_over, p_under, p_push, mean/std PRA
      - normal-approx p_over_normal
      - chosen_model_prob (based on 'method')
      - sportsbook_implied_prob
      - edge (chosen_prob - implied_prob)
      - fair_odds_model (American)
      - ev_per_unit (expected profit per 1 unit staked)
    """
    # Pull + prep data
    df = get_player_gamelog(player_name, season=season, season_type=season_type)
    df = add_pra(df)
    df = filter_gamelog(df, home=home, last_n=last_n, min_minutes=min_minutes)

    if df.empty:
        raise ValueError("No games after applying filters. Loosen filters.")

    # Empirical stats
    emp = pra_empirical_probs(df, line=line)

    # Normal approximation
    norm = pra_normal_over_prob(df, line=line)

    p_emp = emp["p_over"]
    p_norm = norm["p_over_normal"]

    if method == "empirical":
        p_model = p_emp
    elif method == "normal":
        p_model = p_norm
    elif method == "blend":
        # Simple blend: you can tweak this later (e.g., weight by sample size)
        p_model = 0.5 * p_emp + 0.5 * p_norm
    else:
        raise ValueError(f"Unknown method: {method}")

    p_book = american_to_prob(odds)
    edge = p_model - p_book
    fair_odds = prob_to_american(p_model)
    ev = expected_value_per_unit(p_model, odds)

    result: Dict[str, Any] = {
        "player_name": player_name,
        "season": season,
        "season_type": season_type,
        "pra_line": line,
        "sportsbook_odds_over": odds,
        "filters": {
            "home": home,
            "last_n": last_n,
            "min_minutes": min_minutes,
        },
        "empirical": emp,
        "normal": norm,
        "model": {
            "method": method,
            "p_model_over": p_model,
        },
        "sportsbook": {
            "implied_prob_over": p_book,
        },
        "analytics": {
            "edge_prob_points": edge,          # p_model - p_book
            "fair_odds_over": fair_odds,       # model's fair odds
            "ev_per_unit_staked": ev,          # expected profit for 1 unit stake
        },
    }

    return result
