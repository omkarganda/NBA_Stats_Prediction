"""
Player-level rolling features (rate and volatility) from game logs.

Expected input columns:
- PLAYER_ID
- GAME_DATE (datetime or string parsable)
- MIN, PTS, REB, AST, FG3M, FGA, FTA, FG3A, PF, TOV
"""

from __future__ import annotations

import pandas as pd


ROLL_DEFAULTS = [5, 10, 20]


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if not pd.api.types.is_datetime64_any_dtype(df["GAME_DATE"]):
        df = df.copy()
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df


def build_player_rolling_features(df: pd.DataFrame, windows=ROLL_DEFAULTS) -> pd.DataFrame:
    df = _ensure_datetime(df)
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"])
    frames = []
    for pid, grp in df.groupby("PLAYER_ID"):
        g = grp.copy()
        for w in windows:
            g[f"min_mean_L{w}"] = g["MIN"].rolling(w, min_periods=1).mean()
            g[f"min_std_L{w}"] = g["MIN"].rolling(w, min_periods=1).std().fillna(0.0)

            for col in ["PTS", "REB", "AST", "FG3M"]:
                g[f"{col.lower()}_mean_L{w}"] = g[col].rolling(w, min_periods=1).mean()
                g[f"{col.lower()}_std_L{w}"] = g[col].rolling(w, min_periods=1).std().fillna(0.0)

            # Rates per minute (guard against zero minutes)
            m = g["MIN"].clip(lower=1e-3)
            for col in ["PTS", "REB", "AST", "FG3M", "FGA", "FTA", "FG3A", "PF", "TOV"]:
                if col not in g.columns:
                    continue
                rate = g[col] / m
                g[f"{col.lower()}_per_min_L{w}"] = rate.rolling(w, min_periods=1).mean()

        frames.append(g)
    out = pd.concat(frames, ignore_index=True)
    return out
