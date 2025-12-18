# PRD: NBA Player Prop-Betting Projection & Backtesting System

## 0) Summary
Build a production-style pipeline that:
1) ingests NBA game + context data and historical sportsbook player props/odds,
2) trains probabilistic models for **minutes** and **stat distributions**,
3) produces calibrated win probabilities for over/under markets,
4) removes vig correctly, and
5) backtests on strict time splits (with CLV-style market benchmarking).

This PRD is designed to replace the current “Tier-2: stat ~ minutes + Normal residuals” baseline with a system that more closely resembles real prop-betting workflows.

---

## 1) Goals / Non-goals

### Goals
- **Prop markets**: PTS, REB, AST, 3PM, PRA (extensible to others: blocks, steals, turnovers).
- Output a **full distribution** (or a calibrated CDF) per player-stat, not only a mean.
- Build a **backtesting + calibration loop** with honest out-of-sample evaluation.
- Implement **de-vig** to compare model probabilities to market-implied probabilities correctly.
- Produce **daily slate outputs** (CSV/JSON) with edges, EV, and risk controls.

### Non-goals (initial version)
- Real-time in-game/live betting.
- Perfect injury forecasting (we model availability/minutes uncertainty, not medical outcomes).
- Proprietary tracking data dependencies (we allow optional integration, but keep a public-data path).

---

## 2) Key real-world constraints (must-have)

### 2.1 Historical sportsbook prop lines are required
NBA’s public endpoints do **not** provide historical book lines/odds. Without this you cannot:
- backtest “edge/EV” realistically,
- measure calibration versus the actual market,
- estimate if you beat closing.

You need one of:
- a paid odds/props API (typical in real-world workflows),
- stored historical scrapes (if legal/ToS-compliant for you),
- exchange/market data sources (where available).

### 2.2 Market efficiency
Most “public-data-only” player models will not beat sharp closing lines consistently. The system must:
- benchmark against **closing line** and **consensus**,
- track CLV proxies,
- treat the market itself as a strong baseline/prior.

---

## 3) Data inputs (with recommended sources)

### 3.1 NBA game data (public)
- `nba_api`: player game logs, team stats, schedule (already used in repo).
- Optional: play-by-play (possession-level) via `nba_api` endpoints (heavier).
- Optional: `pbpstats`-style derived data (if you have a pipeline; not required for v1).

### 3.2 Team/opponent context (public-ish)
- Pace proxies, offensive/defensive rating (team-level per season, rolling windows).
- Home/away, rest days, back-to-back, 3-in-4, travel distance/time zones (derive from schedule + coordinates).
- Vegas context: game total + spread (requires odds provider; highly predictive).

### 3.3 Injuries / availability
Minimum viable:
- a daily injury status feed (OUT/DOUBTFUL/QUESTIONABLE/PROBABLE),
- starter/bench tags,
- days since return.
This typically requires a provider (or carefully maintained scraping).

### 3.4 Props / odds (required)
For each offered player prop:
- `player_id`, `game_id` or (date + teams), `market` (PTS/REB/AST/3PM/PRA),
- `line` (e.g., 24.5),
- `odds_over`, `odds_under`,
- `timestamp` and ideally `closing_timestamp`,
- sportsbook name (to study book-specific bias/hold).

---

## 4) Modeling strategy (recommended v1 → v2)

### 4.1 Two-stage model (v1 baseline that is “real world shaped”)
1) **Minutes model**: predict a *distribution* of minutes for tonight.
2) **Stat model conditional on minutes**:
   - model per-minute/opportunity rates and convert to a per-game distribution,
   - integrate over minutes uncertainty via simulation.

This fixes the largest realism gap: “minutes as fixed user input.”

### 4.2 Distribution choices by stat (practical defaults)
- **PTS / REB / AST**: Negative Binomial (overdispersed count) is often more realistic than Normal.
  - Model: `E[y] = exp(η)` with `η = f(features) + log(minutes)` (offset-style) or directly simulate from a rate.
- **3PM**:
  - Option A: two-part model: 3PA (count) + 3P% (bounded), simulate makes.
  - Option B: NB on 3PM with minutes + role + opponent perimeter defense features.
- **PRA**:
  - Simulate `PTS, REB, AST` jointly with correlated noise, then sum.
  - Simplest v1: simulate independently then calibrate; best v2: multivariate copula / correlated residuals.

### 4.3 Time-varying + shrinkage (v2 upgrades)
- Hierarchical partial pooling across players (rookies/small samples).
- State-space / time-varying coefficients for role changes.
- Mixture models for “starter vs bench role” regimes.

### 4.4 Market-aware priors (high ROI)
- Use market totals/spreads and even *the prop line itself* as a feature/baseline.
- Goal is not to “ignore the market,” but to exploit systematic errors (injury timing, role changes, book lag).

---

## 5) Feature set (expanded, organized)

### 5.1 Player role & opportunity (offense)
**Core minutes/role**
- rolling minutes (L5/L10/L20), starter indicator, rotation stability (variance of minutes),
- % of games above thresholds (e.g., >28, >32 minutes),
- foul trouble propensity (PF per minute, early foul rates if you have PBP).

**Usage and shot volume proxies (from box score)**
- FGA, 3PA, FTA per minute and per game (rolling windows),
- “aggression” features: FTA/FGA, 3PA share,
- assists: potential proxies like AST per minute, TOV per minute, AST:TOV,
- rebounds: OREB/DREB split, rebound rate proxies (REB per minute).

**Efficiency**
- TS% (needs points + FGA + FTA),
- eFG% (needs FGM/FG3M/FGA),
- FT% (stability proxy; but noisy).

**Form & volatility**
- rolling mean + rolling std for each stat,
- skew/outlier rate (how often player produces spike games),
- consistency metrics (e.g., median absolute deviation).

### 5.2 Opponent & matchup (defense)
Team-level opponent context (rolling windows):
- opponent pace, defensive rating, rebound rate allowed,
- opponent 3PA allowed and 3PM allowed,
- opponent foul rate (FTA allowed).

Position/role defense (if you can build it):
- opponent allowed by position (PG/SG/SF/PF/C) for each stat,
- opponent rim protection proxies (BLK, DREB), perimeter defense proxies (opp 3PA, 3P% allowed).

Lineup interaction (optional but powerful):
- teammate availability impacts usage/minutes (e.g., star OUT increases secondary scorer usage),
- on/off style features: when Player A is out, Player B’s minutes/usage changes.

### 5.3 Game context
- Home/away.
- Rest: days since last game; back-to-back; 3-in-4; 4-in-6.
- Travel: distance, time zones crossed, altitude (Denver), long road trips.
- Game environment from market: team implied total, spread, total (highly predictive).
- Blowout risk: spread magnitude + team strength → affects minutes distributions.

### 5.4 Injury & availability patterns
Inputs (from injury feed):
- status (OUT/DOUBTFUL/QUESTIONABLE/PROBABLE), minutes restriction tags (if available).
Derived features:
- “first game back” indicator and days since return,
- historical minutes cap patterns after injury (player-specific),
- coach-specific tendencies (if you can label coaching changes).

### 5.5 Market microstructure (props)
- time-to-tip, line movement velocity,
- book identity (some books shade unders/overs differently),
- hold (vig) and devigged “true p”,
- closing-line as benchmark target for “beat the market.”

---

## 6) Backtesting, calibration, and reporting (must-have)

### 6.1 Splits
- Strict **time-based** splits (train on past → validate on future).
- Walk-forward evaluation by date/week.

### 6.2 Metrics
Probability quality:
- Brier score, log loss, reliability (calibration) curve by market.
Betting performance:
- ROI, EV realized vs predicted, hit rate, drawdown,
- stratified performance by edge bucket, player type (starter/bench), market (PTS/REB/AST/3PM/PRA).
Market benchmark:
- CLV proxy: compare your model edge at open vs close (requires timestamped props).

### 6.3 Calibration
Per market (PTS, REB, …) and optionally per player archetype:
- isotonic regression or Platt scaling on out-of-sample predictions,
- check calibration drift across season phases.

---

## 7) Vig removal (devig) requirements

### 7.1 Inputs
You need **both sides**: `odds_over` and `odds_under` for the same line.

### 7.2 Methods to implement (evaluate and choose)
- Multiplicative (proportional) normalization
- Additive normalization
- Shin method
- Power method

Implementation should:
- return devigged `p_over`, `p_under` (sum to 1),
- compute implied hold for diagnostics,
- be unit-tested (edge cases: -10000 lines, small holds, etc.).

---

## 8) System design: scripts (one concept per script)

Directory proposal:
- `prop_model/` (Python package with reusable code)
- `scripts/` (thin CLIs that call `prop_model`)
- `data/` (raw/processed/features/models)
- `reports/` (backtests, calibration plots, daily outputs)

Each script below is intentionally narrow. The “business logic” lives in `prop_model/`.

### Script 01 — NBA data ingest
- **File**: `scripts/01_ingest_nba_game_logs.py`
- **Purpose**: download and cache player game logs, rosters, schedule.
- **Inputs**: seasons list, season type, cache path.
- **Outputs**: parquet/csv partitions by season/date; metadata file.
- **Libraries**: `nba_api`, `pandas`, `pyarrow` (recommended), `tenacity`.

### Script 02 — Market data ingest (props/odds)
- **File**: `scripts/02_ingest_market_props.py`
- **Purpose**: ingest prop lines + both-side odds with timestamps.
- **Inputs**: provider config (API key), date range, book filter.
- **Outputs**: normalized table `market_props.parquet` with unique IDs.
- **Libraries**: `requests`, `pandas`, `pyarrow`, `pydantic`, provider SDK (varies).
- **Notes**: implement adapter interface so you can swap providers.

### Script 03 — Canonical IDs + joins
- **File**: `scripts/03_build_canonical_game_table.py`
- **Purpose**: align NBA game IDs/dates/teams with market rows; resolve player ID mapping.
- **Inputs**: nba tables + market props.
- **Outputs**: `training_rows.parquet` with one row per (player, game, market, line).
- **Libraries**: `pandas` or `polars`, `duckdb` (recommended for joins).

### Script 04 — Feature engineering: player/role
- **File**: `scripts/04_build_player_features.py`
- **Purpose**: rolling windows + rate features derived from player logs.
- **Outputs**: `features_player.parquet` keyed by (player_id, game_date/game_id).
- **Libraries**: `pandas`/`polars`, `numpy`.

### Script 05 — Feature engineering: opponent/game context
- **File**: `scripts/05_build_context_features.py`
- **Purpose**: opponent rolling defense, pace proxies, rest/travel, market totals/spreads (if available).
- **Outputs**: `features_context.parquet`.
- **Libraries**: `pandas`/`polars`, `numpy`; optional `geopy` for travel.

### Script 06 — Injury & availability features
- **File**: `scripts/06_build_injury_features.py`
- **Purpose**: transform injury feed into model-ready features and availability priors.
- **Outputs**: `features_injury.parquet`.
- **Libraries**: provider SDK; `pandas`; optional `rapidfuzz` for name matching.

### Script 07 — Minutes model training
- **File**: `scripts/07_train_minutes_model.py`
- **Purpose**: train a probabilistic minutes model (mean + uncertainty or mixture).
- **Inputs**: features + historical minutes.
- **Outputs**: serialized model + feature schema + calibration report.
- **Libraries**:
  - baseline: `scikit-learn`, `lightgbm`/`xgboost` (optional), `joblib`
  - probabilistic options: `statsmodels` or `pymc` (v2)

### Script 08 — Stat distribution model training (per market)
- **File**: `scripts/08_train_stat_models.py`
- **Purpose**: train distributional models for PTS/REB/AST/3PM, conditional on minutes + context.
- **Outputs**: one model per market (and maybe per player archetype) + diagnostics.
- **Libraries**:
  - GLM/NB: `statsmodels`, `scipy`
  - GBM: `scikit-learn`, `lightgbm`/`xgboost`
  - quantile regression alternative: `lightgbm` quantile objective

### Script 09 — Simulation engine (produce CDF/probabilities)
- **File**: `scripts/09_run_simulation.py`
- **Purpose**: for each prop row, sample minutes → sample stat → compute `P(over/under)`, push prob.
- **Outputs**: `model_probs.parquet` with `p_over`, `p_under`, distribution summaries.
- **Libraries**: `numpy`, `scipy`.

### Script 10 — Devig + edge + EV
- **File**: `scripts/10_price_and_rank_edges.py`
- **Purpose**: devig market odds, compare to model probabilities, compute edge/EV and rank.
- **Outputs**: daily card CSV + JSON for app.
- **Libraries**: `pandas`, `numpy`.

### Script 11 — Backtest runner
- **File**: `scripts/11_backtest.py`
- **Purpose**: walk-forward backtest, compute probability + betting metrics.
- **Outputs**: `reports/backtest_*.html` and `reports/backtest_*.parquet`.
- **Libraries**: `pandas`, `numpy`, `scikit-learn` (metrics), `matplotlib`/`plotly`.

### Script 12 — Calibration trainer
- **File**: `scripts/12_calibrate_probabilities.py`
- **Purpose**: fit isotonic/Platt models per market (and optionally per segment).
- **Outputs**: calibration objects + reliability plots + before/after metrics.
- **Libraries**: `scikit-learn`.

### Script 13 — Daily pipeline orchestration
- **File**: `scripts/13_daily_run.py`
- **Purpose**: end-to-end daily run: ingest today props → build features → predict → rank.
- **Outputs**: `reports/daily_YYYYMMDD.csv`, optional Streamlit-consumable artifact.
- **Libraries**: `pydantic-settings` (optional), `rich`, `joblib`.

---

## 9) Python environment / libraries

### 9.1 Baseline requirements (recommended for this PRD)
- `pandas`, `numpy`, `scipy`
- `pyarrow` (parquet IO)
- `duckdb` (fast joins, reproducible feature tables)
- `scikit-learn` (metrics, calibration, baselines)
- `statsmodels` (GLM/NB; great for distributional baselines)
- `tenacity` (retries for NBA endpoints and provider APIs)
- `joblib` (model serialization)
- `pydantic` (schemas/configs; reduces silent data issues)

### 9.2 Optional performance / modeling
- `polars` (faster feature pipelines than pandas for large tables)
- `lightgbm` or `xgboost` (GBM models)
- `optuna` (hyperparameter tuning)
- `pymc` (Bayesian hierarchical/time-varying, v2+)

---

## 10) Deliverables & acceptance criteria

### Deliverables
- A reproducible dataset builder that outputs a canonical training table.
- A minutes distribution model and stat distribution models.
- A devig module and a pricing module.
- A backtest report with:
  - Brier/log loss by market,
  - calibration plots,
  - ROI/EV curves by edge bucket,
  - benchmark vs naive baselines and (if available) closing.

### Acceptance criteria (v1)
- Walk-forward backtest runs end-to-end without manual intervention.
- Calibration improves reliability (measurable Brier/log loss improvement vs uncalibrated).
- Devigged market probabilities sum to ~1 and reported holds match expectation ranges.

---

## 11) References / web resources (starting points)
- Estimated Plus-Minus (EPM) methodology (SPM prior + RAPM): https://dunksandthrees.com/about/epm
- RAPM/SPM prior discussion (Synergy / Sportradar blog): https://sportradar.com/content-hub/blog-en-us/explaining-synergys-new-player-impact-stats/
- Devig method overviews (multiplicative/additive/Shin/power): https://help.outlier.bet/en/articles/8208129-how-to-devig-odds-comparing-the-methods
- NBA load management / schedule density notes (league release): https://www.nba.com/news/nba-sends-data-load-management-study

