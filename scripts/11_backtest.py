"""
Backtest probability forecasts vs outcomes.

Requires a parquet with:
  - y_true (1 if over wins, 0 otherwise)
  - p_model_win
  - p_market (devigged, optional; if absent, skip market ROI)

Usage:
  python scripts/11_backtest.py --data data/backtest/backtest_set.parquet --out reports/backtest_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from prop_model.evaluation import brier_score, log_loss_score, reliability_bins, edge_metrics


def parse_args():
    ap = argparse.ArgumentParser(description="Backtest probability forecasts.")
    ap.add_argument("--data", type=Path, required=True, help="Parquet with y_true, p_model_win, optional p_market.")
    ap.add_argument("--out", type=Path, required=True, help="CSV summary output.")
    return ap.parse_args()


def main():
    args = parse_args()
    df = pd.read_parquet(args.data)
    y = df["y_true"].values
    p = df["p_model_win"].values

    summary = {
        "brier": brier_score(y, p),
        "log_loss": log_loss_score(y, p),
        "n": len(df),
    }

    if "p_market" in df.columns:
        em = edge_metrics(y, p_model=p, p_market=df["p_market"].values)
        summary.update(em)

    prob_true, prob_pred = reliability_bins(y, p, n_bins=10)
    calib_df = pd.DataFrame({"prob_pred_bin": prob_pred, "prob_true_bin": prob_true})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(args.out, index=False)
    calib_df.to_csv(args.out.with_name(args.out.stem + "_calibration.csv"), index=False)
    print(f"[ok] wrote {args.out} and calibration csv")


if __name__ == "__main__":
    main()
