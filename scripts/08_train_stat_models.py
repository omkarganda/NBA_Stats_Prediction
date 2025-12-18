"""
Train Negative Binomial GLM for a given stat (PTS/REB/AST/3PM).

Inputs:
  - parquet with player-game rows and engineered features
  - target column (e.g., PTS)
  - feature columns and optional offset (e.g., log_minutes)

Usage:
  python scripts/08_train_stat_models.py --data data/features/features_player.parquet --target PTS --feature-cols min_mean_L5,pts_per_min_L5 --offset-col log_minutes --out models/stat_pts.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from prop_model.stat_model import StatModel


def parse_args():
    ap = argparse.ArgumentParser(description="Train NB GLM stat model.")
    ap.add_argument("--data", type=Path, required=True, help="Parquet with features + target.")
    ap.add_argument("--target", type=str, required=True, help="Target column (PTS/REB/AST/FG3M).")
    ap.add_argument("--feature-cols", type=str, required=True, help="Comma-separated feature columns.")
    ap.add_argument("--offset-col", type=str, default=None, help="Optional offset column (e.g., log_minutes).")
    ap.add_argument("--out", type=Path, required=True, help="Output joblib path.")
    return ap.parse_args()


def main():
    args = parse_args()
    df = pd.read_parquet(args.data).copy()

    # Add log_minutes offset if requested
    if args.offset_col and args.offset_col == "log_minutes" and "log_minutes" not in df.columns:
        df["log_minutes"] = np.log(df["MIN"].clip(lower=1e-3))

    feature_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    model = StatModel.train(df, target_col=args.target, feature_cols=feature_cols, offset_col=args.offset_col)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)
    print(f"[ok] saved stat model to {args.out}")


if __name__ == "__main__":
    main()
