"""
Build player rolling features from game logs.

Usage:
  python scripts/04_build_player_features.py --logs data/raw_nba/game_logs_2024-25.parquet --out data/features/features_player.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from prop_model.features_player import build_player_rolling_features


def parse_args():
    ap = argparse.ArgumentParser(description="Build player rolling features.")
    ap.add_argument("--logs", type=Path, required=True, help="Parquet of game logs (can be multi-season concatenated).")
    ap.add_argument("--out", type=Path, required=True, help="Output parquet for player features.")
    return ap.parse_args()


def main():
    args = parse_args()
    df = pd.read_parquet(args.logs)
    feats = build_player_rolling_features(df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(args.out, index=False)
    print(f"[ok] wrote {args.out} with {len(feats)} rows")


if __name__ == "__main__":
    main()
