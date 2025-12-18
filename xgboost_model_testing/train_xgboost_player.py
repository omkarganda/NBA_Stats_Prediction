"""
train_xgb_player_models.py

Scaffold for a league-wide player stat model using XGBoost.

Pipeline:

1. Load raw player-game logs (league-wide) from nba_api or CSV.
2. Build a player-game feature table with:
   - Player rolling stats (minutes, pts, reb, ast, 3pm, rates)
   - Team offensive context
   - Basic opponent defensive context
   - Game context (home/away, rest)
   - Recency weights
3. Train:
   - Minutes model: MIN ~ features
   - Stat models: PTS / REB / AST / FG3M / PRA ~ features (+ minutes)
4. Save models for later inference.

NOTE: This is a scaffold, not production-ready code. You will:
- Expand feature sets,
- Plug in better train/val/test splits,
- Add Vegas features / injuries, etc.
"""

import os
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from joblib import dump

from nba_api.stats.endpoints import leaguegamelog, playergamelogs
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import warnings

# Silence just this specific, known-benign warning
warnings.filterwarnings(
    "ignore",
    message="Mean of empty slice",
    category=RuntimeWarning,
    module="numpy.lib._nanfunctions_impl",
)


# -------------------------
# 0. Config
# -------------------------

SEASONS = ["2024-25","2025-26"]
SEASON_TYPE = "Regular Season"

TARGET_STATS = ["PTS", "REB", "AST", "FG3M", "PRA"]
RANDOM_STATE = 42

# Minimum actual minutes to include a game in stat-model training
# (minutes model still trains on all games).
STAT_TRAIN_MIN_MINUTES = 20.0


def _get_player_group_series(df: pd.DataFrame) -> pd.Series:
    """
    Return a Series to use as the grouping key for 'per-player' operations.

    Prefer PLAYER_ID if present; fall back to PLAYER_NAME.
    Raise a clear error if neither is found.
    """
    if "PLAYER_ID" in df.columns:
        return df["PLAYER_ID"]
    if "PLAYER_NAME" in df.columns:
        return df["PLAYER_NAME"]
    raise KeyError(
        f"No PLAYER_ID or PLAYER_NAME column found in DataFrame.\n"
        f"Columns: {list(df.columns)}"
    )

# -------------------------
# 1. Data loading
# -------------------------

def add_recency_weights(df: pd.DataFrame, half_life_days: float = 120.0) -> pd.DataFrame:
    """
    Add RECENCY_WEIGHT based on GAME_DATE across the whole dataset.

    Newer games get weight ~1, older games decay towards 0.
    We guard against empty / all-NaN inputs so we don't trigger
    NumPy 'mean of empty slice' warnings ourselves.
    """
    df = df.copy()

    # If no rows or no valid GAME_DATE, default weight = 1.0
    if df.empty or "GAME_DATE" not in df.columns or df["GAME_DATE"].isna().all():
        df["RECENCY_WEIGHT"] = 1.0
        return df

    max_date = df["GAME_DATE"].max()
    age_days = (max_date - df["GAME_DATE"]).dt.days.astype("float64")

    # Exponential decay: newer games -> age_days small -> weight closer to 1
    w = 0.5 ** (age_days / half_life_days)

    mean_w = w.mean()
    if not np.isfinite(mean_w) or mean_w == 0:
        df["RECENCY_WEIGHT"] = 1.0
    else:
        df["RECENCY_WEIGHT"] = w / mean_w

    return df


def fetch_league_player_logs(seasons: List[str], season_type: str) -> pd.DataFrame:
    """
    Fetch per-player game logs for all players, all teams using nba_api PlayerGameLogs.

    This endpoint returns columns like:
    ['SEASON_YEAR', 'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID',
     'TEAM_ABBREVIATION', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'PTS', 'REB', 'AST', 'FG3M', ...]
    """
    frames = []
    for season in seasons:
        print(f"Fetching league player logs for season {season}...")
        pgl = playergamelogs.PlayerGameLogs(
            season_nullable=season,
            season_type_nullable=season_type,
            player_id_nullable=None,  # all players
        )
        df_season = pgl.get_data_frames()[0].copy()
        df_season["SEASON"] = season

        # Normalize GAME_DATE to datetime once here
        df_season["GAME_DATE"] = pd.to_datetime(df_season["GAME_DATE"])

        frames.append(df_season)

    df_all = pd.concat(frames, ignore_index=True)

    # Sanity check: we MUST have PLAYER_ID and PLAYER_NAME
    required_cols = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "MATCHUP", "PTS", "REB", "AST", "FG3M", "MIN"]
    missing = [c for c in required_cols if c not in df_all.columns]
    if missing:
        raise RuntimeError(f"Expected columns missing from PlayerGameLogs: {missing}. Got: {list(df_all.columns)}")

    # PRA
    if "PRA" not in df_all.columns:
        df_all["PRA"] = df_all["PTS"] + df_all["REB"] + df_all["AST"]

    # Sort for downstream rolling features
    df_all = df_all.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    return df_all



def parse_minutes(min_val) -> float:
    """Convert 'MM:SS' or numeric to float minutes."""
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


def parse_matchup(matchup: str) -> Tuple[str, str, int]:
    """
    Parse MATCHUP strings like:
      - "DEN vs HOU"
      - "DEN vs. HOU"
      - "DEN @ HOU"

    Returns (team_abbr, opp_abbr, is_home) where is_home = 1 for vs / vs.,
    and 0 for @. Falls back to (None, None, 0) if the pattern is unrecognized.
    """
    if not isinstance(matchup, str):
        return None, None, 0

    s = matchup.strip()

    # Normalize common 'vs.' variations to 'vs'
    s = s.replace("vs.", "vs").replace("VS.", "VS").replace("Vs.", "Vs")

    if " vs " in s:
        t, o = s.split(" vs ", 1)
        return t.strip(), o.strip(), 1
    if " @ " in s:
        t, o = s.split(" @ ", 1)
        return t.strip(), o.strip(), 0

    return None, None, 0


# -------------------------
# 2. Feature engineering
# -------------------------

def add_basic_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add MIN_NUM, TEAM_ABBREVIATION, OPP_TEAM_ABBREVIATION, IS_HOME."""
    df = df.copy()
    df["MIN_NUM"] = df["MIN"].apply(parse_minutes)

    team_abbr = []
    opp_abbr = []
    is_home = []
    for m in df["MATCHUP"]:
        t, o, h = parse_matchup(m)
        team_abbr.append(t)
        opp_abbr.append(o)
        is_home.append(h)
    df["TEAM_ABBREVIATION"] = team_abbr
    df["OPP_TEAM_ABBREVIATION"] = opp_abbr
    df["IS_HOME"] = is_home

    return df


def add_player_rolling_features(
    df: pd.DataFrame,
    window_short: int = 5,
    window_long: int = 10,
) -> pd.DataFrame:
    """
    For each player, add rolling features (time-respecting) for:
      - MIN, PTS, REB, AST, FG3M
      - per-min rates
      - volatility (std dev)
    Windows: L5, L10; shifted so they only see past games.
    """
    # Defensive copy
    df = df.copy()

    # ✅ Hard assertion: we *must* have some player identifier here
    if "PLAYER_ID" not in df.columns and "PLAYER_NAME" not in df.columns:
        raise KeyError(
            f"add_player_rolling_features: expected PLAYER_ID or PLAYER_NAME in df, "
            f"but got columns: {list(df.columns)}"
        )

    # Sort by date first, then get the grouping Series on the sorted frame
    df = df.sort_values(["GAME_DATE"]).reset_index(drop=True)

    # Use a Series as the group key (PLAYER_ID if available, else PLAYER_NAME)
    group_series = _get_player_group_series(df)

    def _add_for_window(g: pd.DataFrame, w: int) -> pd.DataFrame:
        # Always sort by GAME_DATE within each player group
        g = g.sort_values("GAME_DATE").reset_index(drop=True)

        # Rolling mean/std for core stats, shifted to avoid leakage
        roll_cols = ["MIN_NUM", "PTS", "REB", "AST", "FG3M"]
        # If we have extra shooting stats, include them as well
        for extra in ["FGA", "FTA", "TOV"]:
            if extra in g.columns:
                roll_cols.append(extra)

        roll = g[roll_cols].rolling(
            window=w,
            min_periods=1,
        ).agg(["mean", "std"]).shift(1)

        # (col, agg) -> f"{col}_{agg}_L{w}"
        roll.columns = [f"{c}_{stat}_L{w}" for c, stat in roll.columns]

        g = pd.concat([g, roll], axis=1)
        return g

    # Group by Series instead of column name string -> avoids KeyError('PLAYER_ID') path
    # We compute rolling windows for L3, L5, L10 (or whatever short/long are set to),
    # always using time-respecting (shifted) stats.
    windows = sorted(set([3, window_short, window_long]))
    for w in windows:
        df = df.groupby(group_series, group_keys=False).apply(
            lambda g, w=w: _add_for_window(g, w)
        )

    # Per-minute rates from short window
    df["PTS_PER_MIN_L5"] = df["PTS_mean_L5"] / df["MIN_NUM_mean_L5"].replace(0, np.nan)
    df["REB_PER_MIN_L5"] = df["REB_mean_L5"] / df["MIN_NUM_mean_L5"].replace(0, np.nan)
    df["AST_PER_MIN_L5"] = df["AST_mean_L5"] / df["MIN_NUM_mean_L5"].replace(0, np.nan)
    df["FG3M_PER_MIN_L5"] = df["FG3M_mean_L5"] / df["MIN_NUM_mean_L5"].replace(0, np.nan)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Role stability proxy: volatility of minutes in the short window
    if "MIN_NUM_std_L5" in df.columns:
        df["ROLE_STABILITY_L5"] = df["MIN_NUM_std_L5"]

    # Shooting volume rates if inputs are available
    if "FGA_mean_L5" in df.columns:
        df["FGA_PER_MIN_L5"] = df["FGA_mean_L5"] / df["MIN_NUM_mean_L5"].replace(0, np.nan)
    if "FTA_mean_L5" in df.columns:
        df["FTA_PER_MIN_L5"] = df["FTA_mean_L5"] / df["MIN_NUM_mean_L5"].replace(0, np.nan)

    return df


def add_player_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each player, add DAYS_SINCE_PRIOR and simple schedule flags.
    """
    df = df.copy()

    if "PLAYER_ID" not in df.columns and "PLAYER_NAME" not in df.columns:
        raise KeyError(
            f"add_player_rest_features: expected PLAYER_ID or PLAYER_NAME in df, "
            f"but got columns: {list(df.columns)}"
        )

    df = df.sort_values(["GAME_DATE"]).reset_index(drop=True)
    group_series = _get_player_group_series(df)

    def _add_rest(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("GAME_DATE").reset_index(drop=True)
        g["DAYS_SINCE_PRIOR"] = (g["GAME_DATE"] - g["GAME_DATE"].shift(1)).dt.days
        g["DAYS_SINCE_PRIOR"] = g["DAYS_SINCE_PRIOR"].fillna(g["DAYS_SINCE_PRIOR"].median())
        g["IS_B2B"] = (g["DAYS_SINCE_PRIOR"] == 1).astype(int)

        # Dense schedule flags (3-in-4, 4-in-6) based on calendar days
        dates = g["GAME_DATE"].values
        games_last_4 = []
        games_last_6 = []
        for i, d in enumerate(dates):
            d4 = d - np.timedelta64(3, "D")
            d6 = d - np.timedelta64(5, "D")
            mask4 = (dates >= d4) & (dates <= d)
            mask6 = (dates >= d6) & (dates <= d)
            games_last_4.append(mask4.sum())
            games_last_6.append(mask6.sum())
        g["GAMES_LAST_4D"] = games_last_4
        g["GAMES_LAST_6D"] = games_last_6
        g["IS_3IN4"] = (g["GAMES_LAST_4D"] >= 3).astype(int)
        g["IS_4IN6"] = (g["GAMES_LAST_6D"] >= 4).astype(int)
        return g

    df = df.groupby(group_series, group_keys=False).apply(
        _add_rest
    )
    return df





def build_team_game_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a team-game aggregation from player rows.
    This is used to compute team & opponent context.
    """
    # A unique game id (leaguegamelog has GAME_ID already)
    agg_dict = {
        "PTS": "sum",
        "REB": "sum",
        "AST": "sum",
        "FG3M": "sum",
    }
    # Add richer box-score stats if available
    for col in ["FGM", "FGA", "FG3A", "FTM", "FTA", "OREB", "DREB", "TOV"]:
        if col in df.columns:
            agg_dict[col] = "sum"

    team_game = df.groupby(["SEASON", "GAME_ID", "TEAM_ABBREVIATION"]).agg(agg_dict).reset_index()

    # sort by team & game date via merge
    team_game = team_game.merge(
        df[["GAME_ID", "TEAM_ABBREVIATION", "GAME_DATE"]].drop_duplicates(),
        on=["GAME_ID", "TEAM_ABBREVIATION"],
        how="left",
    )

    # Per-game advanced team metrics (offense-focused)
    if all(c in team_game.columns for c in ["FGA", "OREB", "TOV", "FTA"]):
        team_game["TEAM_POSS"] = (
            team_game["FGA"] - team_game["OREB"] + team_game["TOV"] + 0.44 * team_game["FTA"]
        )
    else:
        team_game["TEAM_POSS"] = np.nan

    # Offensive rating: points per 100 possessions
    team_game["TEAM_OFF_RTG"] = 100.0 * team_game["PTS"] / team_game["TEAM_POSS"].replace(0, np.nan)

    # Pace proxy: possessions per game (relative scale is what matters)
    team_game["TEAM_PACE"] = team_game["TEAM_POSS"]

    # Shooting profile
    if "FGA" in team_game.columns and "FG3A" in team_game.columns:
        team_game["TEAM_3P_RATE"] = team_game["FG3A"] / team_game["FGA"].replace(0, np.nan)
    else:
        team_game["TEAM_3P_RATE"] = np.nan

    if "FG3M" in team_game.columns and "FG3A" in team_game.columns:
        team_game["TEAM_3P_PCT"] = team_game["FG3M"] / team_game["FG3A"].replace(0, np.nan)
    else:
        team_game["TEAM_3P_PCT"] = np.nan

    # Turnover rate per possession
    if "TOV" in team_game.columns:
        team_game["TEAM_TOV_RATE"] = team_game["TOV"] / team_game["TEAM_POSS"].replace(0, np.nan)
    else:
        team_game["TEAM_TOV_RATE"] = np.nan

    # Offensive rebounding rate (approximate)
    if "OREB" in team_game.columns and "REB" in team_game.columns:
        team_game["TEAM_ORB_RATE"] = team_game["OREB"] / team_game["REB"].replace(0, np.nan)
    else:
        team_game["TEAM_ORB_RATE"] = np.nan

    team_game = team_game.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)
    return team_game


def add_team_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling team-level offensive context and basic opponent context.

    We:
      - Build team-game aggregates
      - Compute rolling team PTS/REB/AST/FG3M (L10) per team
      - Merge those back to player rows as TEAM_* features
      - Merge opponent team rolling stats as OPP_TEAM_* features
    """
    df = df.copy()
    team_game = build_team_game_table(df)

    def _add_team_roll(g: pd.DataFrame, w: int) -> pd.DataFrame:
        g = g.sort_values("GAME_DATE").reset_index(drop=True)

        # Basic box stats rolling means
        base_cols = ["PTS", "REB", "AST", "FG3M"]
        roll_basic = g[base_cols].rolling(window=w, min_periods=1).mean().shift(1)
        roll_basic.columns = [f"TEAM_{c}_L{w}" for c in base_cols]

        # Advanced rate metrics
        if "TEAM_OFF_RTG" in g.columns:
            g[f"TEAM_OFF_RTG_L{w}"] = (
                g["TEAM_OFF_RTG"].rolling(window=w, min_periods=1).mean().shift(1)
            )
        if "TEAM_PACE" in g.columns:
            g[f"TEAM_PACE_L{w}"] = (
                g["TEAM_PACE"].rolling(window=w, min_periods=1).mean().shift(1)
            )
        if "TEAM_3P_RATE" in g.columns:
            g[f"TEAM_3P_RATE_L{w}"] = (
                g["TEAM_3P_RATE"].rolling(window=w, min_periods=1).mean().shift(1)
            )
        if "TEAM_3P_PCT" in g.columns:
            g[f"TEAM_3P_PCT_L{w}"] = (
                g["TEAM_3P_PCT"].rolling(window=w, min_periods=1).mean().shift(1)
            )
        if "TEAM_TOV_RATE" in g.columns:
            g[f"TEAM_TOV_RATE_L{w}"] = (
                g["TEAM_TOV_RATE"].rolling(window=w, min_periods=1).mean().shift(1)
            )
        if "TEAM_ORB_RATE" in g.columns:
            g[f"TEAM_ORB_RATE_L{w}"] = (
                g["TEAM_ORB_RATE"].rolling(window=w, min_periods=1).mean().shift(1)
            )

        g = pd.concat([g, roll_basic], axis=1)
        return g

    # Rolling window of 10 games for team context
    team_game = team_game.groupby("TEAM_ABBREVIATION", group_keys=False).apply(
        lambda g: _add_team_roll(g, 10),
        include_groups=True,
    )

    # In some pandas versions, the group key can end up in the index; flatten if needed
    if "TEAM_ABBREVIATION" not in team_game.columns and "TEAM_ABBREVIATION" in team_game.index.names:
        team_game = team_game.reset_index(level="TEAM_ABBREVIATION")

    # Columns we want to merge back to player rows
    team_cols = [
        "TEAM_PTS_L10",
        "TEAM_REB_L10",
        "TEAM_AST_L10",
        "TEAM_FG3M_L10",
        "TEAM_OFF_RTG_L10",
        "TEAM_PACE_L10",
        "TEAM_3P_RATE_L10",
        "TEAM_3P_PCT_L10",
        "TEAM_TOV_RATE_L10",
        "TEAM_ORB_RATE_L10",
    ]
    team_cols_existing = [c for c in team_cols if c in team_game.columns]

    df = df.merge(
        team_game[
            ["GAME_ID", "TEAM_ABBREVIATION", "GAME_DATE"] + team_cols_existing
        ],
        on=["GAME_ID", "TEAM_ABBREVIATION", "GAME_DATE"],
        how="left",
    )

    # Build opponent team mapping: TEAM_* -> OPP_TEAM_* for the other side of the game
    opp_team = team_game[
        ["GAME_ID", "TEAM_ABBREVIATION"] + team_cols_existing
    ].rename(
        columns={
            "TEAM_ABBREVIATION": "OPP_TEAM_ABBREVIATION",
            "TEAM_PTS_L10": "OPP_TEAM_PTS_L10",
            "TEAM_REB_L10": "OPP_TEAM_REB_L10",
            "TEAM_AST_L10": "OPP_TEAM_AST_L10",
            "TEAM_FG3M_L10": "OPP_TEAM_FG3M_L10",
            "TEAM_OFF_RTG_L10": "OPP_TEAM_OFF_RTG_L10",
            "TEAM_PACE_L10": "OPP_TEAM_PACE_L10",
            "TEAM_3P_RATE_L10": "OPP_TEAM_3P_RATE_L10",
            "TEAM_3P_PCT_L10": "OPP_TEAM_3P_PCT_L10",
            "TEAM_TOV_RATE_L10": "OPP_TEAM_TOV_RATE_L10",
            "TEAM_ORB_RATE_L10": "OPP_TEAM_ORB_RATE_L10",
        }
    )

    df = df.merge(
        opp_team,
        on=["GAME_ID", "OPP_TEAM_ABBREVIATION"],
        how="left",
    )

    return df



def build_feature_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature-building pipeline.

    We:
      - Start from a minimal, explicitly selected set of columns
        so we NEVER lose PLAYER_ID / PLAYER_NAME by accident.
      - Add basic fields (MIN_NUM, TEAM_ABBREVIATION, OPP_TEAM_ABBREVIATION, IS_HOME)
      - Add player rolling features
      - Add rest / B2B flags
      - Add team and opponent context
      - Add recency weights
      - Drop very early games with no rolling context
      - Add simple role flags
    """
    # 🔹 1. Start from a minimal, safe subset of columns
    base_cols = [
        "SEASON",
        "SEASON_YEAR",
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "GAME_ID",
        "GAME_DATE",
        "MATCHUP",
        "MIN",          # string like '28:34'
        "PTS",
        "REB",
        "AST",
        "FG3M",
        "PRA",
    ]

    missing = [c for c in base_cols if c not in df_raw.columns]
    if missing:
        raise RuntimeError(
            f"build_feature_table: expected columns {missing} missing from df_raw.\n"
            f"Available columns: {list(df_raw.columns)}"
        )

    # Required base columns
    df = df_raw[base_cols].copy()

    # Optional box-score columns used for richer features if present
    optional_cols = [
        "FGM",
        "FGA",
        "FG3A",
        "FTM",
        "FTA",
        "OREB",
        "DREB",
        "STL",
        "BLK",
        "TOV",
        "PF",
    ]
    for col in optional_cols:
        if col in df_raw.columns and col not in df.columns:
            df[col] = df_raw[col]

    # Ensure GAME_DATE is datetime
    if not np.issubdtype(df["GAME_DATE"].dtype, np.datetime64):
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # 🔹 2. Add basic helper fields (MIN_NUM, IS_HOME, OPP_TEAM_ABBREVIATION)
    df = add_basic_fields(df)

    # 🔹 3. Add player rolling features (per-player)
    df = add_player_rolling_features(df, window_short=5, window_long=10)

    # 🔹 4. Add player rest / B2B features
    df = add_player_rest_features(df)

    # 🔹 5. Add team & opponent context features
    df = add_team_context_features(df)

    # 🔹 6. Add recency weights
    df = add_recency_weights(df, half_life_days=120.0)

    # 🔹 7. Drop early games with no rolling context
    df = df.dropna(
        subset=[
            "MIN_NUM_mean_L5",
            "PTS_mean_L5",
            "REB_mean_L5",
            "AST_mean_L5",
            "FG3M_mean_L5",
        ]
    ).reset_index(drop=True)

    # 🔹 8. Simple role flags (you can refine these later)
    df["IS_STARTER_RECENT"] = (df["MIN_NUM_mean_L5"] >= 28).astype(int)

    # Bench scorer: high per-minute scoring but lower minutes
    if "PTS_PER_MIN_L5" in df.columns:
        df["BENCH_SCORER_FLAG"] = (
            (df["PTS_PER_MIN_L5"] >= 0.8) & (df["MIN_NUM_mean_L5"] <= 24)
        ).astype(int)
    else:
        df["BENCH_SCORER_FLAG"] = 0

    # Player share of team production over recent window
    if "TEAM_PTS_L10" in df.columns:
        df["PLAYER_PTS_SHARE_L10"] = df["PTS_mean_L10"] / df["TEAM_PTS_L10"].replace(0, np.nan)
    if "TEAM_REB_L10" in df.columns:
        df["PLAYER_REB_SHARE_L10"] = df["REB_mean_L10"] / df["TEAM_REB_L10"].replace(0, np.nan)
    if "TEAM_AST_L10" in df.columns:
        df["PLAYER_AST_SHARE_L10"] = df["AST_mean_L10"] / df["TEAM_AST_L10"].replace(0, np.nan)

    # Approximate usage-rate style feature from L5 shooting/turnovers and team possessions
    if all(c in df.columns for c in ["FGA_mean_L5", "FTA_mean_L5", "TOV_mean_L5", "TEAM_PACE_L10"]):
        df["USG_EST_L5"] = (
            df["FGA_mean_L5"]
            + 0.44 * df["FTA_mean_L5"]
            + df["TOV_mean_L5"]
        ) / df["TEAM_PACE_L10"].replace(0, np.nan)

    return df



# -------------------------
# 3. Train/test split
# -------------------------

def time_based_split(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15):
    """
    Simple time-based split by GAME_DATE.
    """
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    n = len(df)
    train_end = int(train_frac * n)
    val_end = int((train_frac + val_frac) * n)

    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]

    return df_train, df_val, df_test


# -------------------------
# 4. XGBoost training utilities
# -------------------------

def train_xgb_regressor(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    sample_weight_col: str = "RECENCY_WEIGHT",
    params: Dict[str, Any] = None,
) -> XGBRegressor:
    """
    Generic XGBRegressor training helper.
    """
    if params is None:
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 1500,
            "min_child_weight": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "random_state": RANDOM_STATE,
        }

    X_train = df_train[feature_cols].values
    y_train = df_train[target_col].values
    w_train = df_train[sample_weight_col].values

    X_val = df_val[feature_cols].values
    y_val = df_val[target_col].values
    w_val = df_val[sample_weight_col].values

    model = XGBRegressor(**params)
    model.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        sample_weight_eval_set=[w_val],
        verbose=False,
    )

    return model


# -------------------------
# 5. Model training orchestration
# -------------------------

def get_minutes_feature_cols(df: pd.DataFrame) -> List[str]:
    """Select a subset of features for minutes model."""
    cols = [
        # Player form & role
        "MIN_NUM_mean_L3",
        "MIN_NUM_mean_L5",
        "MIN_NUM_mean_L10",
        "MIN_NUM_std_L5",
        "PTS_mean_L5",
        "REB_mean_L5",
        "AST_mean_L5",
        "ROLE_STABILITY_L5",
        "IS_STARTER_RECENT",
        "BENCH_SCORER_FLAG",
        # Schedule & game context
        "DAYS_SINCE_PRIOR",
        "IS_B2B",
        "IS_3IN4",
        "IS_4IN6",
        "IS_HOME",
        # Team context
        "TEAM_PTS_L10",
        "TEAM_REB_L10",
        "TEAM_AST_L10",
        "TEAM_FG3M_L10",
        "TEAM_OFF_RTG_L10",
        "TEAM_PACE_L10",
        # Opponent context
        "OPP_TEAM_PTS_L10",
        "OPP_TEAM_REB_L10",
        "OPP_TEAM_AST_L10",
        "OPP_TEAM_FG3M_L10",
        "OPP_TEAM_OFF_RTG_L10",
        "OPP_TEAM_PACE_L10",
    ]
    # Keep only those that exist (defensive)
    return [c for c in cols if c in df.columns]


def get_stat_feature_cols(df: pd.DataFrame) -> List[str]:
    """Select a subset of features for stat models (includes minutes features)."""
    base = get_minutes_feature_cols(df)
    extra = [
        # Per-minute scoring / usage proxies
        "PTS_PER_MIN_L5",
        "REB_PER_MIN_L5",
        "AST_PER_MIN_L5",
        "FG3M_PER_MIN_L5",
        "FGA_PER_MIN_L5",
        "FTA_PER_MIN_L5",
        "USG_EST_L5",
        # Longer window production
        "PTS_mean_L10",
        "REB_mean_L10",
        "AST_mean_L10",
        "FG3M_mean_L10",
        # Player share of team stats
        "PLAYER_PTS_SHARE_L10",
        "PLAYER_REB_SHARE_L10",
        "PLAYER_AST_SHARE_L10",
        # Additional team rates that might matter more for box stats
        "TEAM_3P_RATE_L10",
        "TEAM_3P_PCT_L10",
        "TEAM_TOV_RATE_L10",
        "TEAM_ORB_RATE_L10",
        "OPP_TEAM_3P_RATE_L10",
        "OPP_TEAM_3P_PCT_L10",
        "OPP_TEAM_TOV_RATE_L10",
        "OPP_TEAM_ORB_RATE_L10",
    ]
    cols = base + [c for c in extra if c in df.columns]
    return cols


def main():
    # 1. Load raw data
    df_raw = fetch_league_player_logs(SEASONS, SEASON_TYPE)

    # 2. Build feature table
    df_feat = build_feature_table(df_raw)
    # Ensure there are no NaNs before training
    if df_feat.isna().any().any():
        nan_cols = df_feat.columns[df_feat.isna().any()].tolist()
        print("Warning: NaNs found in feature table; filling with 0 for columns:", nan_cols)
        df_feat = df_feat.fillna(0)

    # 3. Train/val/test split
    df_train, df_val, df_test = time_based_split(df_feat)

    # 4. Train minutes model
    minutes_features = get_minutes_feature_cols(df_feat)
    print("Minutes features:", minutes_features)

    minutes_model = train_xgb_regressor(
        df_train,
        df_val,
        feature_cols=minutes_features,
        target_col="MIN_NUM",
        sample_weight_col="RECENCY_WEIGHT",
    )

    # Optionally inspect minutes RMSE on test set
    X_test_min = df_test[minutes_features].values
    y_test_min = df_test["MIN_NUM"].values
    y_pred_min = minutes_model.predict(X_test_min)
    print("Test minutes RMSE:", np.sqrt(np.mean((y_pred_min - y_test_min) ** 2)))

    # 5. Train stat models (only on games where player actually played enough minutes)
    stat_features = get_stat_feature_cols(df_feat)
    print("Stat features:", stat_features)

    # Filter train/val/test for stat models to high-minute games
    df_train_stats = df_train[df_train["MIN_NUM"] >= STAT_TRAIN_MIN_MINUTES].reset_index(drop=True)
    df_val_stats = df_val[df_val["MIN_NUM"] >= STAT_TRAIN_MIN_MINUTES].reset_index(drop=True)
    df_test_stats = df_test[df_test["MIN_NUM"] >= STAT_TRAIN_MIN_MINUTES].reset_index(drop=True)

    stat_models: Dict[str, XGBRegressor] = {}

    for stat in TARGET_STATS:
        print(f"\nTraining model for {stat} (MIN_NUM >= {STAT_TRAIN_MIN_MINUTES}) ...")
        model = train_xgb_regressor(
            df_train_stats,
            df_val_stats,
            feature_cols=stat_features,
            target_col=stat,
            sample_weight_col="RECENCY_WEIGHT",
        )
        stat_models[stat] = model

        X_test = df_test_stats[stat_features].values
        y_test = df_test_stats[stat].values
        y_pred = model.predict(X_test)
        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
        mae = np.mean(np.abs(y_pred - y_test))
        print(f"Test {stat} RMSE: {rmse:.3f}, MAE: {mae:.3f}")

    # 6. Save models and feature table for inference scripts
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)

    dump(minutes_model, os.path.join(models_dir, "minutes_model.joblib"))
    for stat, model in stat_models.items():
        dump(model, os.path.join(models_dir, f"stat_model_{stat.lower()}.joblib"))

    # Save the feature table to reuse schema/context at inference (in the script directory)
    df_feat.to_pickle(os.path.join(os.path.dirname(__file__), "df_feat.pkl"))


if __name__ == "__main__":
    main()
