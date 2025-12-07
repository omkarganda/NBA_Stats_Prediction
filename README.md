# NBA Stats Prediction

Streamlit app plus helper modules for researching NBA prop bets. It fetches live data from `nba_api`, shows who actually plays heavy minutes, and runs a minutes-aware Tier-2 model that prices points/rebounds/assists/3PM/PRA props versus sportsbook odds.

## Repository layout
- `app.py`: Streamlit UI. Pulls teams/rosters from `nba_api`, computes season averages, filters by minutes, and lets you configure props (line/odds/direction/minutes/opponent factor) for the Tier-2 model (`evaluate_tier2_stat_bet`). Caches NBA calls with `st.cache_data`.
- `nba_helpers_tier1.py`: Quick PRA helper that pulls a single season of games, supports filters (home/away, last N, min minutes), offers empirical and normal-approx probabilities, and returns edge/EV/fair odds.
- `nba_helpers_tier2.py`: Minutes-aware, multi-season model. Builds recency weights, fits weighted linear regression of stat ~ minutes, applies an optional opponent mean multiplier, and computes win prob, edge, fair odds, and EV for points/rebounds/assists/threes/PRA.
- `nba_eda.ipynb`: Small notebook that demos `nba_api` queries and both helper modules (Jokić/Sarr/Westbrook examples).
- `requirements.txt`: Minimal deps (`streamlit`, `pandas`, `numpy`, `nba_api`).
- `Dockerfile`: Streamlit container (Python 3.11 slim). Installs deps and runs `streamlit run app.py` on port 8501.
- `html/index.html` + `html/Dockerfile`: Tiny static "it works" page served by nginx (handy for Tailscale checks).
- `scripts/prefetch.py`: CLI to prefetch rosters and multi-season game logs to CSVs under `data/`.
- `data/`: Cache output from the prefetch script (`rosters_<season>.csv`, `game_logs_<season>.csv`, `metadata.json`).

## How the app works (`app.py`)
1) Sidebar picks team + season + min avg minutes for the roster table. NBA data is fetched through `nba_api` endpoints (`teams`, `commonteamroster`, `playergamelog`).
2) The table shows players meeting the minutes threshold with season averages (PTS/REB/AST/3PM/PRA).
3) You enter modeling seasons (comma-separated), select players from the table, and configure props with expected minutes, line/odds, direction, and an opponent multiplier.
4) For each prop, the Tier-2 model fits stat~minutes with recency weights, predicts a Normal distribution at your minutes, applies the opponent factor to the mean, and reports model win prob vs implied, edge (probability points), fair odds, and EV per unit.
5) Sidebar toggle lets you prefer cached CSVs from `data/` (prefetch) and fall back to live API when the cache is missing.

## Running locally
- Python 3.11+ recommended. Install deps: `pip install -r requirements.txt`.
- Launch Streamlit: `streamlit run app.py` (serve on http://localhost:8501 by default).
- NBA data is fetched live; you need outbound network access for `nba_api` calls.
- To prefetch and cache data: `python scripts/prefetch.py --seasons 2024-25,2025-26` (writes CSVs to `data/`; see `data/README.md` for scheduling).

## Docker
- Build app image: `docker build -t nba-stats-app .`
- Run: `docker run --rm -p 8501:8501 nba-stats-app`
- Static test page: `docker build -t nba-static html && docker run --rm -p 8080:80 nba-static`

## Notebook usage
`nba_eda.ipynb` expects the helper modules in the working directory and internet access for `nba_api`. Cells show how to:
- Look up a player and fetch a single-season game log.
- Run the Tier-1 PRA helper with filters.
- Run Tier-2 examples for PRA and multiple props.

## Key functions (quick reference)
- `nba_helpers_tier1.evaluate_pra_bet(...)`: Empirical/normal/blended PRA probabilities, edge, fair odds, EV with optional filters.
- `nba_helpers_tier2.evaluate_tier2_stat_bet(...)`: Minutes-aware, recency-weighted pricing for points/rebounds/assists/threes/PRA with opponent adjustment.
- `app.get_team_players_with_stats(...)`: Build the roster table filtered by average minutes.

## Notes / next ideas
- Opponent factor is a simple mean multiplier; refine with matchup-specific data if needed.
- Tier-2 currently assumes Normal residuals and linear minutes relationship; monitor when players have role changes or low-minute samples.
