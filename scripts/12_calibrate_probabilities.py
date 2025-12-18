"""
Calibrate probability forecasts with isotonic or Platt scaling.

Inputs:
  - parquet with y_true and p_model_win

Outputs:
  - parquet with calibrated probabilities (p_calibrated)
  - serialized calibrator (joblib)

Usage:
  python scripts/12_calibrate_probabilities.py --data data/backtest/backtest_set.parquet --method isotonic --out data/backtest/calibrated.parquet --model-out models/calibrator.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def parse_args():
    ap = argparse.ArgumentParser(description="Calibrate probabilities.")
    ap.add_argument("--data", type=Path, required=True, help="Parquet with y_true and p_model_win.")
    ap.add_argument("--method", type=str, default="isotonic", choices=["isotonic", "sigmoid"])
    ap.add_argument("--out", type=Path, required=True, help="Parquet output with p_calibrated.")
    ap.add_argument("--model-out", type=Path, required=True, help="Joblib path for calibrator.")
    return ap.parse_args()


def main():
    args = parse_args()
    df = pd.read_parquet(args.data)
    y = df["y_true"].values
    p = df["p_model_win"].values

    if args.method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(p, y)
        p_cal = calibrator.predict(p)
    else:  # sigmoid / Platt
        lr = LogisticRegression()
        lr.fit(p.reshape(-1, 1), y)
        calibrator = lr
        p_cal = lr.predict_proba(p.reshape(-1, 1))[:, 1]

    df["p_calibrated"] = p_cal

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, args.model_out)
    print(f"[ok] wrote {args.out} and calibrator {args.model_out}")


if __name__ == "__main__":
    main()
