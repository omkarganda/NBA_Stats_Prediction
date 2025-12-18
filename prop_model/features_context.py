"""
Context features: rest, back-to-back, home/away, and optional market totals/spreads.

Inputs:
    games_df columns (minimum):
        - GAME_ID
        - GAME_DATE
        - TEAM_ABBR
        - OPP_ABBR
        - HOME (bool/int)
        - TEAM_SCORE (optional)
        - OPP_SCORE (optional)
        - SPREAD (optional, team-centric, e.g., -5.5 if favorite)
        - TOTAL (optional, game total)
"""

from __future__ import annotations

import pandas as pd


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if not pd.api.types.is_datetime64_any_dtype(df["GAME_DATE"]):
        df = df.copy()
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df


def build_rest_features(games_df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_datetime(games_df)
    df = df.sort_values(["TEAM_ABBR", "GAME_DATE"])
    frames = []
    for team, grp in df.groupby("TEAM_ABBR"):
        g = grp.copy()
        g["prev_game_date"] = g["GAME_DATE"].shift(1)
        g["days_since_last"] = (g["GAME_DATE"] - g["prev_game_date"]).dt.days
        g["is_b2b"] = (g["days_since_last"] == 1).astype(int)
        g["is_3in4"] = ((g["GAME_DATE"] - g["GAME_DATE"].shift(2)).dt.days <= 4).astype(int)
        g["is_4in6"] = ((g["GAME_DATE"] - g["GAME_DATE"].shift(3)).dt.days <= 6).astype(int)
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def add_market_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure market features exist; fill with NaN if missing.
    """
    out = df.copy()
    for col in ["SPREAD", "TOTAL"]:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def build_context_features(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns games_df with rest flags, spread/total columns ensured.
    """
    rest_df = build_rest_features(games_df)
    rest_df = add_market_context(rest_df)
    return rest_df
