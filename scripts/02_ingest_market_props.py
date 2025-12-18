"""
Normalize market prop data (lines + both-side odds) into a canonical parquet.

Inputs:
- CSV/Parquet from your odds provider with at least:
    player_name, market (PTS/REB/AST/3PM/PRA), line, odds_over, odds_under,
    game_date, team_abbr, opp_abbr, book, ts (timestamp)

Usage:
  python scripts/02_ingest_market_props.py --input raw_props.csv --out data/raw_market/props.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

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
    ap = argparse.ArgumentParser(description="Normalize market prop data to parquet.")
    ap.add_argument("--input", type=Path, required=True, help="CSV/Parquet input from provider.")
    ap.add_argument("--out", type=Path, required=True, help="Output parquet path.")
    return ap.parse_args()


def load_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        candidates = list(Path(".").glob("**/*.csv"))
        hint = ""
        if "path/to/your/raw_props.csv" in str(path):
            hint = " (replace the placeholder with your actual props file path)"
        raise FileNotFoundError(
            f"Input file not found: {path.resolve()}{hint}. "
            f"CSV files seen under cwd: {[str(p) for p in candidates[:5]]}..."
        )
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def main():
    args = parse_args()
    df = load_any(args.input)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required cols: {missing}")

    df["game_date"] = pd.to_datetime(df["game_date"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"[ok] wrote {args.out} with {len(df)} rows")


if __name__ == "__main__":
    main()
