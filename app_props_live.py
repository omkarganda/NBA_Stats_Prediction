import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from nba_helpers_tier2 import evaluate_tier2_stat_bet


# -----------------------------------------------
# Helpers for loading cached data
# -----------------------------------------------


def find_latest_season(logs_dir: Path) -> Optional[str]:
    seasons = []
    for p in logs_dir.glob("game_logs_*.parquet"):
        name = p.stem.replace("game_logs_", "")
        seasons.append(name)
    if not seasons:
        return None
    seasons = sorted(seasons)
    return seasons[-1]


def load_logs(logs_dir: Path, season: str) -> pd.DataFrame:
    path = logs_dir / f"game_logs_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run ingest first.")
    df = pd.read_parquet(path)
    df["SEASON"] = season
    return df


def load_rosters(logs_dir: Path, season: str) -> pd.DataFrame:
    path = logs_dir / f"rosters_{season}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    # Harmonize column names from nba_api roster
    if "PLAYER_ID" not in df.columns and "PLAYER_ID" in df.columns:
        pass
    return df


def build_player_avgs(logs: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    df = logs.copy()
    # Minutes numeric
    def parse_min(m):
        if pd.isna(m):
            return 0.0
        s = str(m)
        if ":" in s:
            mm, ss = s.split(":")
            try:
                return int(mm) + int(ss) / 60.0
            except ValueError:
                return 0.0
        try:
            return float(s)
        except ValueError:
            return 0.0

    df["MIN_NUM"] = df["MIN"].apply(parse_min)
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]

    count_col = "GAME_ID" if "GAME_ID" in df.columns else "Game_ID"
    agg = (
        df.groupby(["PLAYER_ID", "TEAM_ABBR"], as_index=False)
        .agg(
            avg_min=("MIN_NUM", "mean"),
            avg_pts=("PTS", "mean"),
            avg_reb=("REB", "mean"),
            avg_ast=("AST", "mean"),
            avg_fg3m=("FG3M", "mean"),
            avg_pra=("PRA", "mean"),
            games=(count_col, "count"),
        )
    )

    if not rosters.empty and "PLAYER_ID" in rosters.columns:
        name_map = rosters[["PLAYER_ID", "PLAYER", "TEAM_ABBR"]].drop_duplicates("PLAYER_ID")
        agg = agg.merge(name_map, on=["PLAYER_ID", "TEAM_ABBR"], how="left")
        agg["Player"] = agg["PLAYER"].fillna(agg["PLAYER_ID"].astype(str))
    else:
        agg["Player"] = agg["PLAYER_ID"].astype(str)

    return agg


# -----------------------------------------------
# Streamlit UI
# -----------------------------------------------


st.set_page_config(page_title="NBA Prop Entry (no props file)", layout="wide")
st.title("🏀 NBA Prop Entry (manual lines/odds, Tier-2 model)")
st.caption(
    "This dashboard pulls cached game logs, lists players averaging 25+ minutes this season, "
    "lets you enter props/odds by hand, and prices them with nba_helpers_tier2.py."
)

logs_dir = Path("data/raw_nba")
latest = find_latest_season(logs_dir)

if latest is None:
    st.error("No cached game logs found in data/raw_nba. Run ingest first (bash scripts/cron_ingest.sh).")
    st.stop()

season = st.sidebar.text_input("Season (YYYY-YY)", value=latest)
min_minutes_filter = st.sidebar.slider("Min avg minutes filter", 0.0, 40.0, 25.0, 1.0)
seasons_for_model = st.sidebar.text_input(
    "Seasons to use in model (comma-separated)", value=season, help="Example: 2023-24,2024-25,2025-26"
)
half_life_games = st.sidebar.number_input("Recency half-life (games)", 5.0, 40.0, 20.0, 1.0)
min_minutes_for_fit = st.sidebar.number_input("Min minutes to include in fit", 0.0, 40.0, 10.0, 1.0)
min_games_for_fit = st.sidebar.number_input("Min games to fit", 5, 50, 12, 1)

try:
    logs = load_logs(logs_dir, season)
    rosters = load_rosters(logs_dir, season)
except Exception as exc:
    st.error(f"Failed to load cached data for season {season}: {exc}")
    st.stop()

agg = build_player_avgs(logs, rosters)
agg = agg[agg["avg_min"] >= min_minutes_filter].sort_values("avg_min", ascending=False)

if agg.empty:
    st.warning("No players found for this season/minutes filter. Adjust the slider or check the season.")
    st.stop()

st.subheader(f"Players averaging ≥ {min_minutes_filter:.0f} minutes ({season})")
st.dataframe(
    agg[["Player", "TEAM_ABBR", "games", "avg_min", "avg_pts", "avg_reb", "avg_ast", "avg_fg3m", "avg_pra"]],
    use_container_width=True,
)

selected_players = st.multiselect(
    "Select players to enter props for",
    options=agg["Player"].tolist(),
    default=agg["Player"].head(5).tolist(),
)

if not selected_players:
    st.info("Select at least one player above.")
    st.stop()

model_seasons = [s.strip() for s in seasons_for_model.split(",") if s.strip()]
stat_options = ["points", "rebounds", "assists", "threes", "pra"]

st.subheader("Enter props (manual lines/odds)")
user_configs = []
for player in selected_players:
    row = agg[agg["Player"] == player].iloc[0]
    avg_min = float(row["avg_min"])
    with st.expander(f"{player} ({row['TEAM_ABBR']}) — season avg {avg_min:.1f} min", expanded=False):
        expected_minutes = st.number_input(
            f"Expected minutes for {player}",
            min_value=0.0,
            max_value=48.0,
            value=avg_min,
            step=1.0,
            key=f"exp_min_{player}",
        )
        opp_factor = st.slider(
            f"Opponent factor (μ multiplier) for {player}",
            min_value=0.80,
            max_value=1.20,
            value=1.00,
            step=0.01,
            key=f"opp_{player}",
            help="Rough adjustment; 1.00 = neutral, >1 boost, <1 reduce.",
        )
        chosen_stats = st.multiselect(
            f"Stats to evaluate for {player}",
            options=stat_options,
            default=["points", "rebounds", "assists", "pra"],
            key=f"stats_{player}",
        )
        for stat in chosen_stats:
            c1, c2, c3 = st.columns(3)
            with c1:
                line = st.number_input(f"{stat} line ({player})", value=0.0, step=0.5, key=f"line_{player}_{stat}")
            with c2:
                odds = st.number_input(f"{stat} odds ({player})", value=-110, step=5, key=f"odds_{player}_{stat}")
            with c3:
                direction = st.selectbox(
                    f"{stat} direction ({player})", options=["over", "under"], key=f"dir_{player}_{stat}"
                )
            user_configs.append(
                {
                    "player_name": str(player),
                    "team": row["TEAM_ABBR"],
                    "stat_type": stat,
                    "line": float(line),
                    "odds": int(odds),
                    "direction": direction,
                    "expected_minutes": float(expected_minutes),
                    "opponent_factor": float(opp_factor),
                }
            )

run_btn = st.button("Run model on entered props")

if not run_btn:
    st.stop()

results = []
for cfg in user_configs:
    if cfg["line"] == 0.0 and cfg["odds"] == 0:
        continue
    try:
        res = evaluate_tier2_stat_bet(
            player_name=cfg["player_name"],
            stat_type=cfg["stat_type"],
            line=cfg["line"],
            odds=cfg["odds"],
            expected_minutes=cfg["expected_minutes"],
            seasons=model_seasons,
            season_type="Regular Season",
            half_life_games=half_life_games,
            min_minutes_for_fit=min_minutes_for_fit,
            min_games_for_fit=min_games_for_fit,
            direction=cfg["direction"],
            opponent_factor=cfg["opponent_factor"],
        )
        dist = res["distribution"]
        probs = res["probabilities"]
        analytics = res["analytics"]
        results.append(
            {
                "player_name": res["player_name"],
                "team": cfg["team"],
                "stat_type": res["stat_type"],
                "line": res["line"],
                "direction": res["direction"],
                "odds": res["odds"],
                "expected_minutes": res["expected_minutes"],
                "seasons_used": ",".join(res["seasons_used"]),
                "mu_raw": dist["mu_raw"],
                "mu_adjusted": dist["mu_adjusted"],
                "sigma": dist["sigma"],
                "p_model_win": probs["p_model_win"],
                "p_book_win_implied": probs["p_book_win_implied"],
                "edge_prob_points": analytics["edge_prob_points"],
                "fair_odds": analytics["fair_odds"],
                "ev_per_unit": analytics["ev_per_unit"],
            }
        )
    except Exception as exc:
        results.append({"player_name": cfg["player_name"], "stat_type": cfg["stat_type"], "error": str(exc)})

res_df = pd.DataFrame(results)
st.subheader("Results")
st.dataframe(res_df, use_container_width=True)

csv_data = res_df.to_csv(index=False)
st.download_button(
    label="Download results CSV",
    data=csv_data,
    file_name=f"props_results_{datetime.date.today()}.csv",
    mime="text/csv",
)

st.caption(
    "Notes: Data comes from cached game logs in data/raw_nba. "
    "Seasons in the model field control which seasons are used per prop. "
    "Lines/odds are entered manually; no sportsbook feed is pulled."
)
