"""
Devig, compute edge/EV, and rank props.

Inputs:
  - parquet from 09_run_simulation.py containing p_model_win
  - columns: odds_over, odds_under, line, market, player_name, p_model_win

Usage:
  python scripts/10_price_and_rank_edges.py --preds data/preds/model_probs.parquet --method multiplicative --out data/preds/priced.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from prop_model.devig import devig_multiplicative, devig_additive, devig_shin, devig_power, american_from_prob


METHODS = {
    "multiplicative": devig_multiplicative,
    "additive": devig_additive,
    "shin": devig_shin,
    "power": lambda o, u: devig_power(o, u, exponent=0.5),
}


def parse_args():
    ap = argparse.ArgumentParser(description="Devig and rank edges.")
    ap.add_argument("--preds", type=Path, required=True, help="Parquet with model probabilities and market odds.")
    ap.add_argument("--method", type=str, default="multiplicative", choices=list(METHODS.keys()))
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    df = pd.read_parquet(args.preds)
    devig_fn = METHODS[args.method]

    rows = []
    for _, row in df.iterrows():
        devig = devig_fn(row["odds_over"], row["odds_under"])
        p_over = devig["p_over"]
        p_model = row["p_model_win"]
        edge = p_model - p_over if row.get("direction", "over") == "over" else (1 - p_model) - devig["p_under"]
        fair_odds_model = american_from_prob(p_model)
        out = row.to_dict()
        out.update(
            {
                "p_over_devig": p_over,
                "p_under_devig": devig["p_under"],
                "hold": devig["hold"],
                "edge": edge,
                "fair_odds_model": fair_odds_model,
            }
        )
        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values("edge", ascending=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(f"[ok] wrote {args.out} with {len(out_df)} rows (sorted by edge)")


if __name__ == "__main__":
    main()
