"""
Run simulation to price props using minutes model + stat model + Monte Carlo.

Inputs:
  - props parquet with feature columns for minutes model and stat model
  - minutes model joblib
  - stat model joblib
  - columns: player_name, market, line, direction, [feature cols...]

Outputs:
  - parquet with p_model_win, mu_sim, edge placeholders (no devig yet)

Usage:
  python scripts/09_run_simulation.py --props data/canonical/training_rows.parquet --minutes-model models/minutes_model.joblib --stat-model models/stat_pts.joblib --minutes-cols min_mean_L5,min_std_L5 --out data/preds/model_probs.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from prop_model.simulation import simulate_prop_probability
from prop_model.stat_model import StatModel
from prop_model.minutes_model import MinutesModel


def parse_args():
    ap = argparse.ArgumentParser(description="Simulate prop probabilities.")
    ap.add_argument("--props", type=Path, required=True, help="Parquet with props + features.")
    ap.add_argument("--minutes-model", type=Path, required=True)
    ap.add_argument("--stat-model", type=Path, required=True)
    ap.add_argument("--minutes-cols", type=str, required=True, help="Comma-separated columns for minutes model.")
    ap.add_argument("--stat-feature-cols", type=str, required=True, help="Comma-separated columns for stat model.")
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

    # Predict per-game mean at minutes_mean, approximate rate-per-minute
    # Sanity check feature presence for stat model
    missing = [c for c in stat_feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing stat feature columns: {missing}")

    df_stat = df.copy()
    df_stat["MIN"] = df["minutes_mean"]
    df_stat["log_minutes"] = np.log(df_stat["MIN"].clip(lower=1e-3))
    mu, alpha = stat_model.predict_nb_params(df_stat)
    df["mu_game"] = mu
    df["alpha"] = alpha
    df["rate_per_min"] = df["mu_game"] / df["minutes_mean"].clip(lower=1e-3)

    rows = []
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
        out = row.to_dict()
        out.update(sim)
        rows.append(out)

    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(f"[ok] wrote {args.out} with {len(out_df)} rows")


if __name__ == "__main__":
    main()
