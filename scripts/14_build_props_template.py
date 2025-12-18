"""
Build a placeholder props CSV for today's slate:
- Uses cached game logs (parquet) under data/raw_nba to compute season averages.
- Filters players with Avg_Min >= 25.
- Outputs a CSV with required columns for 02_ingest_market_props.py, leaving line/odds empty.

Usage:
  python scripts/14_build_props_template.py --season 2025-26 --out data/raw_market/props_template.csv

Notes:
- This does NOT fetch real sportsbook props/odds (requires a provider). It builds a template you can fill.
- If you have multiple parquet game logs in data/raw_nba, it picks the newest season by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import re


REQUIRED_COLS = [
    "player_name",
    "market",
    "line",
    "odds_over",
    "odds_under",
    "game_date",
    "team_abbr",
    "opp_abbr",
    "book",
]


def parse_args():
    ap = argparse.ArgumentParser(description="Build placeholder props CSV from cached logs.")
    ap.add_argument("--season", type=str, default=None, help="Season string, e.g., 2025-26. If omitted, pick latest parquet.")
    ap.add_argument("--logs-dir", type=Path, default=Path("data/raw_nba"), help="Dir containing game_logs_*.parquet.")
    ap.add_argument("--min-minutes", type=float, default=25.0, help="Minimum average minutes filter.")
    ap.add_argument("--out", type=Path, default=Path("data/raw_market/props_template.csv"))
    return ap.parse_args()


def pick_latest_season(logs_dir: Path) -> str:
    candidates = list(logs_dir.glob("game_logs_*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No game_logs_*.parquet found in {logs_dir}")
    seasons = []
    for p in candidates:
        m = re.search(r"game_logs_(\d{4}-\d{2})\.parquet", p.name)
        if m:
            seasons.append(m.group(1))
    if not seasons:
        raise RuntimeError(f"No season patterns found in {logs_dir}")
    seasons = sorted(seasons)
    return seasons[-1]


def load_logs(logs_dir: Path, season: str) -> pd.DataFrame:
    path = logs_dir / f"game_logs_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run ingest first.")
    df = pd.read_parquet(path)
    df["SEASON"] = season
    return df


def main():
    args = parse_args()
    logs_dir = args.logs_dir
    season = args.season or pick_latest_season(logs_dir)

    df = load_logs(logs_dir, season)
    df["MIN_NUM"] = pd.to_numeric(df["MIN"], errors="coerce")
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]

    count_col = "GAME_ID" if "GAME_ID" in df.columns else "Game_ID" if "Game_ID" in df.columns else None
    if count_col is None:
        raise KeyError("No GAME_ID or Game_ID column found in logs; cannot count games.")

    agg = (
        df.groupby(["PLAYER_ID", "SEASON", "TEAM_ABBR"], as_index=False)
        .agg(
            avg_min=("MIN_NUM", "mean"),
            avg_pts=("PTS", "mean"),
            avg_reb=("REB", "mean"),
            avg_ast=("AST", "mean"),
            avg_fg3m=("FG3M", "mean"),
            avg_pra=("PRA", "mean"),
            games=(count_col, "count"),
        )
    )

    filtered = agg[agg["avg_min"] >= args.min_minutes].copy()
    if filtered.empty:
        raise RuntimeError("No players meet the minutes threshold; check season or ingest data.")

    # Build template rows for markets of interest
    markets = ["PTS", "REB", "AST", "FG3M", "PRA"]
    rows = []
    today = pd.Timestamp.today().normalize().date().isoformat()
    for _, row in filtered.iterrows():
        name = str(row["PLAYER_ID"])  # Placeholder; PLAYER name not in logs; caller may map later.
        team = row["TEAM_ABBR"]
        for mkt in markets:
            rows.append(
                {
                    "player_name": name,
                    "market": mkt,
                    "line": None,
                    "odds_over": None,
                    "odds_under": None,
                    "game_date": today,
                    "team_abbr": team,
                    "opp_abbr": "",
                    "book": "",
                }
            )

    out_df = pd.DataFrame(rows, columns=REQUIRED_COLS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"[ok] wrote template props CSV to {args.out} with {len(out_df)} rows for season {season}")


if __name__ == "__main__":
    main()
