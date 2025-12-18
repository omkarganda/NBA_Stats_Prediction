# Usage Guide: Prop-Betting Pipeline

This follows `PRD_PROP_BETTING.md` and matches the scripts in `scripts/`.

## Environment
Install deps:
```
pip install -r requirements.txt
```

## Data expectations
- **NBA logs/rosters**: pulled from `nba_api` (scripts/01_ingest_nba_game_logs.py).
- **Market props**: you must supply both sides of odds for each prop (over/under). Use your provider/scrape and normalize via `scripts/02_ingest_market_props.py`.
- **Schedule**: team-game rows with GAME_DATE, TEAM_ABBR, OPP_ABBR, HOME, optional SPREAD/TOTAL (for context features).
- **Injuries**: provider feed with status per player; normalized via `scripts/06_build_injury_features.py`.

## Pipeline (minimal runnable path)
1) **Ingest NBA data (resumable, retries)**
```
python scripts/01_ingest_nba_game_logs.py --seasons 2024-25,2025-26 --out data/raw_nba --sleep 0.75 --max-retries 4 --timeout 60
```
- If you interrupt or get timeouts, rerun the same command; it reuses existing parquet and only refetches missing teams/players.

2) **Normalize market props**
```
python scripts/02_ingest_market_props.py --input raw_props.csv --out data/raw_market/props.parquet
```

3) **Canonical join (props + IDs)**
```
python scripts/03_build_canonical_game_table.py --nba data/raw_nba --market data/raw_market/props.parquet --out data/canonical/training_rows.parquet
```

4) **Player features**
```
python scripts/04_build_player_features.py --logs data/raw_nba/game_logs_2024-25.parquet --out data/features/features_player.parquet
```

5) **Context features**
```
python scripts/05_build_context_features.py --schedule data/schedule.csv --out data/features/features_context.parquet
```

6) **Injury features (optional but recommended)**
```
python scripts/06_build_injury_features.py --input data/raw_injuries.csv --out data/features/features_injury.parquet
```

7) **Train minutes model** (choose feature columns from features_player)
```
python scripts/07_train_minutes_model.py --features data/features/features_player.parquet \
  --feature-cols min_mean_L5,min_std_L5,min_mean_L10 \
  --out models/minutes_model.joblib
```

8) **Train stat models** (per market, e.g., PTS)
```
python scripts/08_train_stat_models.py --data data/features/features_player.parquet \
  --target PTS \
  --feature-cols min_mean_L5,pts_per_min_L5,pts_per_min_L10 \
  --offset-col log_minutes \
  --out models/stat_pts.joblib
```

9) **Simulate probabilities for props**
```
python scripts/09_run_simulation.py \
  --props data/canonical/training_rows.parquet \
  --minutes-model models/minutes_model.joblib \
  --stat-model models/stat_pts.joblib \
  --minutes-cols min_mean_L5,min_std_L5 \
  --stat-feature-cols min_mean_L5,pts_per_min_L5 \
  --out data/preds/model_probs.parquet
```

10) **Devig and rank edges**
```
python scripts/10_price_and_rank_edges.py --preds data/preds/model_probs.parquet \
  --method multiplicative \
  --out data/preds/priced.parquet
```

11) **Backtest (needs y_true outcomes)**
```
python scripts/11_backtest.py --data data/backtest/backtest_set.parquet --out reports/backtest_summary.csv
```

12) **Calibrate probabilities**
```
python scripts/12_calibrate_probabilities.py \
  --data data/backtest/backtest_set.parquet \
  --method isotonic \
  --out data/backtest/calibrated.parquet \
  --model-out models/calibrator.joblib
```

13) **Daily run (end-to-end once models are trained)**
```
python scripts/13_daily_run.py \
  --props data/today/props.parquet \
  --minutes-model models/minutes_model.joblib \
  --stat-model models/stat_pts.joblib \
  --minutes-cols min_mean_L5,min_std_L5 \
  --stat-feature-cols pts_per_min_L5,min_mean_L5 \
  --devig-method multiplicative \
  --out reports/daily_props.parquet
```

## Notes and next steps
- You must supply **historical props** with both over/under odds to devig and backtest realistically.
- Feature choices matter: start simple (rolling minutes, per-minute rates) and expand with context/injury features.
- For joint PRA modeling, extend `prop_model.stat_model` to handle multivariate simulation or simulate PTS/REB/AST separately with correlated residuals.
- Calibration should be evaluated on a held-out, time-based set before deployment.
