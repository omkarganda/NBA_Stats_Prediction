"""
Train a probabilistic minutes model (quantile GBMs).

Inputs:
  - features parquet with player-game rows including target column MIN
  - feature columns specified via --feature-cols

Usage:
  python scripts/07_train_minutes_model.py --features data/features/features_player.parquet --feature-cols min_mean_L5,min_mean_L10,min_std_L5 --out models/minutes_model.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from prop_model.minutes_model import MinutesModel


def parse_args():
    ap = argparse.ArgumentParser(description="Train minutes quantile model.")
    ap.add_argument("--features", type=Path, required=True, help="Parquet with player-game features including MIN.")
    ap.add_argument("--feature-cols", type=str, required=True, help="Comma-separated feature columns.")
    ap.add_argument("--out", type=Path, required=True, help="Output path for model (joblib).")
    return ap.parse_args()


def main():
    args = parse_args()
    df = pd.read_parquet(args.features)
    feature_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    X = df[feature_cols].values
    y = df["MIN"].values
    model = MinutesModel.train(X, y, feature_names=feature_cols)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)
    print(f"[ok] saved minutes model to {args.out}")


if __name__ == "__main__":
    main()
