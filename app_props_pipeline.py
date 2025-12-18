import io
import datetime
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

from nba_helpers_tier2 import evaluate_tier2_stat_bet


st.set_page_config(page_title="NBA Prop Processor (Tier-2)", layout="wide")
st.title("🏀 NBA Prop Processor (Tier-2)")
st.caption(
    "Upload or paste your prop sheet (player, stat, line, odds, direction, expected minutes, seasons) "
    "and process them with the Tier-2 model from nba_helpers_tier2.py."
)


def parse_seasons(seasons_raw: str) -> List[str]:
    return [s.strip() for s in seasons_raw.split(",") if s.strip()]


required_cols = [
    "player_name",
    "stat_type",
    "line",
    "odds",
    "direction",
    "expected_minutes",
    "seasons",
]

st.sidebar.header("Input options")
default_path = Path("data/raw_market/props_template.csv")
use_local = False

if default_path.exists():
    use_local = st.sidebar.checkbox(
        f"Load local template: {default_path}", value=False, help="Reads from data/raw_market/props_template.csv"
    )

uploaded = st.sidebar.file_uploader("Or upload CSV with required columns", type=["csv"])

text_input = st.sidebar.text_area(
    "Or paste CSV rows (with header)",
    value="player_name,stat_type,line,odds,direction,expected_minutes,seasons\n"
    "1631210,points,20.5,-110,over,32,2025-26",
    height=180,
)

data_df = None

if use_local and default_path.exists():
    data_df = pd.read_csv(default_path)
elif uploaded is not None:
    data_df = pd.read_csv(uploaded)
elif text_input.strip():
    data_df = pd.read_csv(io.StringIO(text_input))


if data_df is None or data_df.empty:
    st.warning("No data loaded yet. Upload/paste CSV or toggle the local template.")
    st.stop()

missing = [c for c in required_cols if c not in data_df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

st.subheader("Input props")
st.dataframe(data_df, use_container_width=True)

run_btn = st.button("Run Tier-2 model on all props")

if not run_btn:
    st.stop()

results = []
for _, row in data_df.iterrows():
    try:
        seasons = row["seasons"]
        if isinstance(seasons, str):
            seasons_list = parse_seasons(seasons)
        else:
            seasons_list = list(seasons)

        res = evaluate_tier2_stat_bet(
            player_name=str(row["player_name"]),
            stat_type=str(row["stat_type"]),
            line=float(row["line"]),
            odds=int(row["odds"]),
            expected_minutes=float(row["expected_minutes"]),
            seasons=seasons_list,
            season_type="Regular Season",
            direction=str(row["direction"]),
        )
        dist = res["distribution"]
        probs = res["probabilities"]
        analytics = res["analytics"]
        results.append(
            {
                "player_name": res["player_name"],
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
        results.append({"player_name": row.get("player_name"), "stat_type": row.get("stat_type"), "error": str(exc)})

res_df = pd.DataFrame(results)

st.subheader("Results")
st.dataframe(res_df, use_container_width=True)

csv = res_df.to_csv(index=False)
st.download_button(
    label="Download results as CSV",
    data=csv,
    file_name=f"props_results_{datetime.date.today()}.csv",
    mime="text/csv",
)

st.caption(
    "Notes: uses the existing nba_helpers_tier2.py pipeline (recency-weighted minutes regression + Normal tail). "
    "Seasons are pulled per-row; ensure your CSV has appropriate seasons for each player."
)
