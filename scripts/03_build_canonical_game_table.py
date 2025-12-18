"""
Build canonical training rows by joining market props to NBA data.

Inputs:
- game logs parquet (from 01_ingest_nba_game_logs.py)
- rosters parquet for target season(s)
- market props parquet (from 02_ingest_market_props.py)

Output:
- parquet with one row per (player, game, market) ready for feature building.

Usage:
  python scripts/03_build_canonical_game_table.py --nba data/raw_nba --market data/raw_market/props.parquet --out data/canonical/training_rows.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser(description="Join market props to NBA logs/rosters.")
    ap.add_argument("--nba", type=Path, required=True, help="Dir with game_logs_*.parquet and rosters_*.parquet")
    ap.add_argument("--market", type=Path, required=True, help="Parquet with market props")
    ap.add_argument("--out", type=Path, required=True, help="Output parquet")
    return ap.parse_args()


def load_rosters(nba_dir: Path) -> pd.DataFrame:
    frames = []
    for p in nba_dir.glob("rosters_*.parquet"):
        frames.append(pd.read_parquet(p))
    if not frames:
        raise RuntimeError("No rosters parquet files found.")
    rosters = pd.concat(frames, ignore_index=True)
    rosters["PLAYER_UP"] = rosters["PLAYER"].str.upper()
    return rosters


def load_logs(nba_dir: Path) -> pd.DataFrame:
    frames = []
    for p in nba_dir.glob("game_logs_*.parquet"):
        frames.append(pd.read_parquet(p))
    if not frames:
        raise RuntimeError("No game log parquet files found.")
    logs = pd.concat(frames, ignore_index=True)
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
    return logs


def main():
    args = parse_args()
    rosters = load_rosters(args.nba)
    logs = load_logs(args.nba)
    market = pd.read_parquet(args.market)
    market["PLAYER_UP"] = market["player_name"].str.upper()

    # Map player_id via roster name match + team + season if present
    m = market.merge(
        rosters[["PLAYER_UP", "PLAYER_ID", "TEAM_ABBR", "SEASON"]],
        on="PLAYER_UP",
        how="left",
        suffixes=("", "_roster"),
    )

    missing_id = m["PLAYER_ID"].isna().sum()
    if missing_id:
        print(f"[warn] {missing_id} rows missing PLAYER_ID after roster merge. Consider adding better mapping.")

    # Attach last available game date for sanity check
    logs_small = logs[["PLAYER_ID", "GAME_DATE", "TEAM_ABBR", "SEASON"]].copy()
    logs_small = logs_small.rename(columns={"TEAM_ABBR": "TEAM_ABBR_LOG"})
    m = m.merge(
        logs_small.groupby("PLAYER_ID")["GAME_DATE"].max().reset_index().rename(columns={"GAME_DATE": "last_game_date"}),
        on="PLAYER_ID",
        how="left",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    m.to_parquet(args.out, index=False)
    print(f"[ok] wrote {args.out} with {len(m)} rows")


if __name__ == "__main__":
    main()
