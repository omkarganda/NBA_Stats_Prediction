"""
Build rest/travel/market context features at team-game level.

Usage:
  python scripts/05_build_context_features.py --schedule data/schedule.csv --out data/features/features_context.parquet

Expected columns in schedule CSV/Parquet:
  GAME_ID, GAME_DATE, TEAM_ABBR, OPP_ABBR, HOME (1/0), SPREAD (optional), TOTAL (optional)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from prop_model.features_context import build_context_features


def parse_args():
    ap = argparse.ArgumentParser(description="Build context features (rest, b2b, spread, total).")
    ap.add_argument("--schedule", type=Path, required=True, help="CSV/Parquet with team-game rows.")
    ap.add_argument("--out", type=Path, required=True, help="Output parquet path.")
    return ap.parse_args()


def load_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def main():
    args = parse_args()
    df = load_any(args.schedule)
    feats = build_context_features(df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(args.out, index=False)
    print(f"[ok] wrote {args.out} with {len(feats)} rows")


if __name__ == "__main__":
    main()
