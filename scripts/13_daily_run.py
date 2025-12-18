"""
Daily orchestration: simulate props, devig, and rank edges.

This script assumes you already prepared:
- props parquet (with feature columns for minutes + stat models, odds_over/odds_under/line/market/direction)
- trained minutes model
- trained stat model for the selected market

Usage example:
  python scripts/13_daily_run.py \
    --props data/today/props.parquet \
    --minutes-model models/minutes_model.joblib \
    --stat-model models/stat_pts.joblib \
    --minutes-cols min_mean_L5,min_std_L5 \
    --stat-feature-cols pts_per_min_L5,min_mean_L5 \
    --devig-method multiplicative \
    --out reports/daily_props.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from prop_model.devig import devig_multiplicative, devig_additive, devig_shin, devig_power, american_from_prob
from prop_model.minutes_model import MinutesModel
from prop_model.simulation import simulate_prop_probability
from prop_model.stat_model import StatModel


DEVIG = {
    "multiplicative": devig_multiplicative,
    "additive": devig_additive,
    "shin": devig_shin,
    "power": lambda o, u: devig_power(o, u, exponent=0.5),
}


def parse_args():
    ap = argparse.ArgumentParser(description="Daily prop pricing runner.")
    ap.add_argument("--props", type=Path, required=True)
    ap.add_argument("--minutes-model", type=Path, required=True)
    ap.add_argument("--stat-model", type=Path, required=True)
    ap.add_argument("--minutes-cols", type=str, required=True)
    ap.add_argument("--stat-feature-cols", type=str, required=True)
    ap.add_argument("--devig-method", type=str, default="multiplicative", choices=list(DEVIG.keys()))
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    df = pd.read_parquet(args.props)
    minutes_cols = [c.strip() for c in args.minutes_cols.split(",") if c.strip()]
    stat_feature_cols = [c.strip() for c in args.stat_feature_cols.split(",") if c.strip()]

    minutes_model: MinutesModel = joblib.load(args.minutes_model)
    stat_model: StatModel = joblib.load(args.stat_model)

    X_min = df[minutes_cols].values
    minutes_pred = minutes_model.predict_mean_std(X_min)
    df["minutes_mean"] = minutes_pred["mean"]
    df["minutes_std"] = minutes_pred["std"]

    missing = [c for c in stat_feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing stat feature columns: {missing}")

    df_stat = df.copy()
    df_stat["MIN"] = df["minutes_mean"]
    df_stat["log_minutes"] = np.log(df_stat["MIN"].clip(lower=1e-3))

    mu_game, alpha = stat_model.predict_nb_params(df_stat)
    df["mu_game"] = mu_game
    df["alpha"] = alpha
    df["rate_per_min"] = df["mu_game"] / df["minutes_mean"].clip(lower=1e-3)

    devig_fn = DEVIG[args.devig_method]
    results = []
    for _, row in df.iterrows():
        sim = simulate_prop_probability(
            minutes_mean=row["minutes_mean"],
            minutes_std=row["minutes_std"],
            rate_per_min=row["rate_per_min"],
            alpha=row["alpha"],
            line=row["line"],
            direction=row.get("direction", "over"),
            num_sims=4000,
        )
        devig = devig_fn(row["odds_over"], row["odds_under"])
        p_model = sim["p_model_win"]
        p_over = devig["p_over"]
        p_under = devig["p_under"]
        direction = row.get("direction", "over")
        edge = p_model - p_over if direction == "over" else (1 - p_model) - p_under
        fair_odds_model = american_from_prob(p_model)

        out = row.to_dict()
        out.update(sim)
        out.update(
            {
                "p_over_devig": p_over,
                "p_under_devig": p_under,
                "hold": devig["hold"],
                "edge": edge,
                "fair_odds_model": fair_odds_model,
            }
        )
        results.append(out)

    out_df = pd.DataFrame(results).sort_values("edge", ascending=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(f"[ok] wrote {args.out} with {len(out_df)} rows (sorted by edge)")


if __name__ == "__main__":
    main()
