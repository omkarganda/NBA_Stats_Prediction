"""
Transform injury feed into model-ready features.

Expected input columns (example):
  player_name, player_id (optional), status (OUT/DOUBTFUL/QUESTIONABLE/PROBABLE/ACTIVE),
  report_date, notes

Outputs basic availability flags and days-since-last-status.

Usage:
  python scripts/06_build_injury_features.py --input data/raw_injuries.csv --out data/features/features_injury.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser(description="Build injury availability features.")
    ap.add_argument("--input", type=Path, required=True, help="CSV/Parquet injury feed.")
    ap.add_argument("--out", type=Path, required=True, help="Output parquet.")
    return ap.parse_args()


def load_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


STATUS_MAP = {
    "OUT": 0,
    "DOUBTFUL": 0.1,
    "QUESTIONABLE": 0.25,
    "PROBABLE": 0.75,
    "ACTIVE": 1.0,
}


def main():
    args = parse_args()
    df = load_any(args.input)
    df["report_date"] = pd.to_datetime(df["report_date"])
    df = df.sort_values(["player_name", "report_date"])
    df["status_up"] = df["status"].str.upper()
    df["availability_score"] = df["status_up"].map(STATUS_MAP).fillna(0.5)
    df["days_since_report"] = (pd.Timestamp.utcnow().normalize() - df["report_date"]).dt.days

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"[ok] wrote {args.out} with {len(df)} rows")


if __name__ == "__main__":
    main()
