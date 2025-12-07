import datetime
import time
from pathlib import Path

import streamlit as st
import pandas as pd

from nba_api.stats.static import teams
from nba_api.stats.endpoints import commonteamroster, playergamelog

# Import Tier-2 model function
from nba_helpers_tier2 import evaluate_tier2_stat_bet


# ---------------------- Helpers ---------------------- #

def parse_minutes(min_val):
    """
    Convert NBA API MIN field to numeric minutes as float.

    Examples:
        "32:15" -> 32.25
        35      -> 35.0
        None    -> 0.0
    """
    if min_val is None or pd.isna(min_val):
        return 0.0

    if isinstance(min_val, (int, float)):
        return float(min_val)

    s = str(min_val)
    if ":" in s:
        m, sec = s.split(":")
        try:
            return int(m) + int(sec) / 60.0
        except ValueError:
            return 0.0

    try:
        return float(s)
    except ValueError:
        return 0.0


def with_retry(fn, retries=3, base_pause=1.5):
    """
    Retry helper for nba_api calls with exponential backoff.
    """
    last_err = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - network/endpoint errors
            last_err = exc
            sleep_for = base_pause * (2 ** attempt)
            time.sleep(sleep_for)
    raise last_err


def load_cached_roster(season: str, data_dir: str = "data"):
    """
    Load roster CSV if present (written by scripts/prefetch.py).
    """
    path = Path(data_dir) / f"rosters_{season}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    expected_cols = {"PLAYER", "PLAYER_ID", "POSITION", "NUM", "TEAM_ID"}
    if not expected_cols.issubset(set(df.columns)):
        return None
    return df


def load_cached_game_logs(season: str, data_dir: str = "data"):
    """
    Load per-player game logs CSV for a season if present.
    """
    path = Path(data_dir) / f"game_logs_{season}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "PLAYER_ID" not in df.columns:
        return None
    return df


@st.cache_data(show_spinner=False)
def get_teams_table():
    """Get all NBA teams as a DataFrame."""
    team_list = teams.get_teams()
    df = pd.DataFrame(team_list)
    return df[["id", "full_name", "abbreviation", "nickname", "city"]].sort_values(
        "full_name"
    )


@st.cache_data(show_spinner=True)
def get_team_roster(team_id: int, season: str, use_cache: bool, data_dir: str = "data"):
    """
    Get current roster for a team in a given season.

    Returns DataFrame with PLAYER and PLAYER_ID columns.
    """
    if use_cache:
        cached = load_cached_roster(season, data_dir=data_dir)
        if cached is not None:
            df_team = cached[cached["TEAM_ID"] == team_id]
            if not df_team.empty:
                return df_team[["PLAYER", "PLAYER_ID", "POSITION", "NUM"]]

    def fetch():
        roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season)
        return roster.get_data_frames()[0]

    df_roster = with_retry(fetch)
    return df_roster[["PLAYER", "PLAYER_ID", "POSITION", "NUM"]]


@st.cache_data(show_spinner=True)
def get_player_season_averages(player_id: int, season: str, use_cache: bool, data_dir: str = "data"):
    """
    Pull game logs for a player and compute season averages:

    - Games played
    - Avg minutes
    - Avg PTS, REB, AST, FG3M, PRA
    """
    if use_cache:
        gl_cached = load_cached_game_logs(season, data_dir=data_dir)
        if gl_cached is not None:
            df = gl_cached[gl_cached["PLAYER_ID"] == player_id].copy()
        else:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    if df.empty:
        def fetch():
            gl = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star="Regular Season",
            )
            return gl.get_data_frames()[0]

        try:
            df = with_retry(fetch)
        except Exception:
            return {
                "games": 0,
                "avg_min": 0.0,
                "pts": 0.0,
                "reb": 0.0,
                "ast": 0.0,
                "fg3m": 0.0,
                "pra": 0.0,
            }

    if df.empty:
        return {
            "games": 0,
            "avg_min": 0.0,
            "pts": 0.0,
            "reb": 0.0,
            "ast": 0.0,
            "fg3m": 0.0,
            "pra": 0.0,
        }

    df["MIN_NUM"] = df["MIN"].apply(parse_minutes)

    games_played = len(df)
    avg_min = df["MIN_NUM"].mean()
    pts = df["PTS"].mean()
    reb = df["REB"].mean()
    ast = df["AST"].mean()
    fg3m = df["FG3M"].mean()
    pra = pts + reb + ast

    return {
        "games": games_played,
        "avg_min": avg_min,
        "pts": pts,
        "reb": reb,
        "ast": ast,
        "fg3m": fg3m,
        "pra": pra,
    }


@st.cache_data(show_spinner=True)
def get_team_players_with_stats(
    team_id: int,
    season: str,
    min_avg_minutes: float = 25.0,
    use_cache: bool = False,
    data_dir: str = "data",
):
    """
    For a team + season, compute season averages for each roster player
    and return a DataFrame filtered by avg minutes >= min_avg_minutes.

    Columns:
        Player, Player_ID, Position, Jersey,
        Games_Played, Avg_Minutes,
        Avg_PTS, Avg_REB, Avg_AST, Avg_3PM, Avg_PRA
    """
    roster_df = get_team_roster(team_id, season, use_cache=use_cache, data_dir=data_dir)
    data_rows = []

    for _, row in roster_df.iterrows():
        player_name = row["PLAYER"]
        player_id = row["PLAYER_ID"]
        position = row["POSITION"]
        jersey = row["NUM"]

        stats = get_player_season_averages(
            player_id, season, use_cache=use_cache, data_dir=data_dir
        )
        games = stats["games"]
        if games == 0:
            continue

        data_rows.append(
            {
                "Player": player_name,
                "Player_ID": player_id,
                "Position": position,
                "Jersey": jersey,
                "Games_Played": games,
                "Avg_Minutes": stats["avg_min"],
                "Avg_PTS": stats["pts"],
                "Avg_REB": stats["reb"],
                "Avg_AST": stats["ast"],
                "Avg_3PM": stats["fg3m"],
                "Avg_PRA": stats["pra"],
            }
        )

    if not data_rows:
        return pd.DataFrame()

    df = pd.DataFrame(data_rows)

    # Filter by average minutes
    df = df[df["Avg_Minutes"] >= min_avg_minutes]

    # Sort & round
    df = df.sort_values("Avg_Minutes", ascending=False)
    df["Avg_Minutes"] = df["Avg_Minutes"].round(1)
    for col in ["Avg_PTS", "Avg_REB", "Avg_AST", "Avg_3PM", "Avg_PRA"]:
        df[col] = df[col].round(1)

    return df.reset_index(drop=True)


# ---------------------- Streamlit UI ---------------------- #

def main():
    st.set_page_config(
        page_title="NBA Minutes & Tier-2 Prop Model",
        layout="wide",
    )

    st.title("🏀 NBA Rotation + Tier-2 Prop Model")
    st.caption(
        "See who actually plays heavy minutes, then send them into a minutes-aware, "
        "multi-season Tier-2 model for points / rebounds / assists / 3PM / PRA."
    )

    # Sidebar: team + season + filter
    st.sidebar.header("Team & Season Settings")

    teams_df = get_teams_table()
    team_names = teams_df["full_name"].tolist()

    selected_team_name = st.sidebar.selectbox("Choose a team", team_names)

    default_season = "2025-26"
    season = st.sidebar.text_input("Display season (format: YYYY-YY)", value=default_season)
    use_cache = st.sidebar.checkbox(
        "Use cached data (if available)",
        value=True,
        help="Read from data/rosters_<season>.csv and data/game_logs_<season>.csv, fall back to live NBA API.",
    )

    min_avg_minutes = st.sidebar.slider(
        "Minimum average minutes (for table)",
        min_value=10.0,
        max_value=40.0,
        value=25.0,
        step=1.0,
    )

    # NEW: Seasons for the Tier-2 model (multi-season)
    st.sidebar.markdown("### Model seasons")
    model_seasons_raw = st.sidebar.text_input(
        "Seasons for Tier-2 model (comma-separated)",
        value=season,  # default to current display season, but you can type "2022-23,2023-24,2024-25"
        help='Example: "2022-23,2023-24,2024-25"',
    )
    model_seasons = [s.strip() for s in model_seasons_raw.split(",") if s.strip()]

    # Team info
    selected_team_row = teams_df[teams_df["full_name"] == selected_team_name].iloc[0]
    team_id = int(selected_team_row["id"])
    team_abbrev = selected_team_row["abbreviation"]

    st.subheader(f"{selected_team_name} ({team_abbrev}) — Season {season}")
    st.write(
        f"Players with **avg ≥ {min_avg_minutes:.0f} minutes per game** this season, "
        "plus their core stat averages."
    )

    # Fetch team stats table
    with st.spinner("Fetching roster and season stats..."):
        try:
            df_players = get_team_players_with_stats(
                team_id=team_id,
                season=season,
                min_avg_minutes=min_avg_minutes,
                use_cache=use_cache,
                data_dir="data",
            )
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return

    if df_players.empty:
        st.warning(
            "No players found matching the criteria. "
            "Try lowering the minimum average minutes or check the season format."
        )
        return

    # Show table with stats
    st.dataframe(
        df_players[
            [
                "Player",
                "Position",
                "Jersey",
                "Games_Played",
                "Avg_Minutes",
                "Avg_PTS",
                "Avg_REB",
                "Avg_AST",
                "Avg_3PM",
                "Avg_PRA",
            ]
        ],
        use_container_width=True,
    )

    st.markdown("---")

    # ---------------- Tier-2 Model Section ---------------- #

    st.header("🎯 Tier-2 Model: Evaluate Props for Tonight")

    # Game context (meta only for now)
    col_date, col_opp = st.columns([1, 2])
    with col_date:
        game_date = st.date_input(
            "Game date",
            value=datetime.date.today(),
        )
    with col_opp:
        opponent_text = st.text_input(
            "Opponent (optional, for your reference)",
            value="",
            help="Use this to remind yourself of matchup; opponent effect is handled via a rough multiplier below.",
        )

    st.caption(
        "Pick one or more players from the table above, set expected minutes for tonight, "
        "and enter betting lines & odds for each stat you care about."
    )

    # Player selection from the filtered table
    players_list = df_players["Player"].tolist()
    selected_players = st.multiselect(
        "Select players to send into the Tier-2 model",
        options=players_list,
    )

    if not selected_players:
        st.info("Select at least one player above to configure Tier-2 projections.")
        return

    # Default modeling settings
    if not model_seasons:
        st.warning("No seasons provided for the model; using display season only.")
        model_seasons = [season]

    half_life_games = 20.0
    min_minutes_for_fit = 15.0
    min_games_for_fit = 12

    st.markdown("### Configure props for selected players")

    bet_configs = []

    for player_name in selected_players:
        row = df_players[df_players["Player"] == player_name].iloc[0]
        avg_min = float(row["Avg_Minutes"])

        with st.expander(f"{player_name} — season avg {avg_min:.1f} min", expanded=False):
            # Expected minutes (prefilled with season avg)
            expected_minutes = st.number_input(
                f"Expected minutes for tonight — {player_name}",
                min_value=0.0,
                max_value=48.0,
                value=avg_min,
                step=1.0,
                key=f"exp_min_{player_name}",
            )

            st.write("**Season averages** (for context):")
            st.write(
                f"- PTS: {row['Avg_PTS']}  "
                f"- REB: {row['Avg_REB']}  "
                f"- AST: {row['Avg_AST']}  "
                f"- 3PM: {row['Avg_3PM']}  "
                f"- PRA: {row['Avg_PRA']}"
            )

            # NEW: Opponent adjustment factor
            st.markdown("**Opponent adjustment (rough)**")
            opp_factor = st.slider(
                f"Opponent factor (μ multiplier) for {player_name}",
                min_value=0.80,
                max_value=1.20,
                value=1.00,
                step=0.01,
                help=(
                    "Rough adjustment for pace/defense.\n"
                    "1.00 = neutral opponent\n"
                    ">1.00 = faster pace / weaker defense (boost stats)\n"
                    "<1.00 = slower pace / stronger defense (mute stats)"
                ),
                key=f"opp_factor_{player_name}",
            )

            # Choose stats to model
            stat_options = ["points", "rebounds", "assists", "threes", "pra"]
            chosen_stats = st.multiselect(
                f"Stats to evaluate for {player_name}",
                options=stat_options,
                default=["points", "rebounds", "assists", "pra"],
                key=f"stats_{player_name}",
            )

            for stat_type in chosen_stats:
                st.markdown(f"**{player_name} — {stat_type.upper()} prop**")
                c1, c2, c3 = st.columns([1, 1, 1])

                with c1:
                    line = st.number_input(
                        f"Line ({stat_type})",
                        value=0.0,
                        step=0.5,
                        key=f"line_{player_name}_{stat_type}",
                    )
                with c2:
                    odds = st.number_input(
                        f"Odds ({stat_type})",
                        value=-110,
                        step=5,
                        key=f"odds_{player_name}_{stat_type}",
                    )
                with c3:
                    direction = st.selectbox(
                        "Direction",
                        options=["over", "under"],
                        key=f"dir_{player_name}_{stat_type}",
                    )

                bet_configs.append(
                    {
                        "player_name": player_name,
                        "stat_type": stat_type,
                        "line": float(line),
                        "odds": int(odds),
                        "direction": direction,
                        "expected_minutes": float(expected_minutes),
                        "opponent_factor": float(opp_factor),
                    }
                )

    run_button = st.button("Run Tier-2 model for all configured props")

    if run_button:
        if not bet_configs:
            st.warning("You added no props (all lines were 0). Set some lines first.")
            return

        st.markdown("### 📊 Tier-2 Model Results")

        for config in bet_configs:
            if config["line"] == 0.0:
                continue

            player_name = config["player_name"]
            stat_type = config["stat_type"]
            line = config["line"]
            odds = config["odds"]
            direction = config["direction"]
            expected_minutes = config["expected_minutes"]
            opponent_factor = config["opponent_factor"]

            st.markdown(
                f"#### {player_name} — {stat_type.upper()} {direction} {line} @ {odds}"
            )

            try:
                result = evaluate_tier2_stat_bet(
                    player_name=player_name,
                    stat_type=stat_type,
                    line=line,
                    odds=odds,
                    expected_minutes=expected_minutes,
                    seasons=model_seasons,
                    season_type="Regular Season",
                    half_life_games=half_life_games,
                    min_minutes_for_fit=min_minutes_for_fit,
                    min_games_for_fit=min_games_for_fit,
                    direction=direction,
                    opponent_factor=opponent_factor,
                )
            except Exception as e:
                st.error(f"Error evaluating {player_name} {stat_type}: {e}")
                continue

            dist = result["distribution"]
            mu_raw = dist["mu_raw"]
            mu_adj = dist["mu_adjusted"]
            sigma = dist["sigma"]

            p_model = result["probabilities"]["p_model_win"]
            p_book = result["probabilities"]["p_book_win_implied"]
            edge = result["analytics"]["edge_prob_points"]
            fair_odds = result["analytics"]["fair_odds"]
            ev = result["analytics"]["ev_per_unit"]

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(
                    "Raw vs Adjusted mean (μ)",
                    f"{mu_raw:.2f} → {mu_adj:.2f}",
                    help="Raw = minutes-only model; Adjusted = after opponent factor.",
                )
            with c2:
                st.metric(
                    "Win prob (model vs book)",
                    f"{p_model:.3f} vs {p_book:.3f}",
                )
            with c3:
                st.metric(
                    "Edge & Fair Odds",
                    f"{edge*100:.1f} pct pts | {fair_odds}",
                )

            st.write(f"**σ (volatility)**: `{sigma:.2f}`")
            st.write(f"**EV per 1 unit staked**: `{ev:.3f}` units")

            st.caption(
                "This uses your chosen seasons, recency weighting, expected minutes, and a rough opponent "
                "adjustment (μ multiplier). It’s a guide, not a guarantee — combine it with your own "
                "injury/matchup judgment."
            )
            st.markdown("---")


if __name__ == "__main__":
    main()
