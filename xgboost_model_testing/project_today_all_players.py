"""
project_today_all_players.py

Run this after training to generate projections for every player in today's NBA games.

It:
- Loads the trained XGBoost models + saved feature table from train_xgboost_player.py
- Pulls today's games from NBA Scoreboard
- Builds a "future" feature row for each eligible player (based on recent history)
- Predicts minutes and PTS/REB/AST/FG3M/PRA
- Saves to player_projections_today.csv
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from joblib import load
from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.static import teams

from train_xgboost_player import (
    SEASONS,
    TARGET_STATS,
    get_minutes_feature_cols,
    get_stat_feature_cols,
)

# Base paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# Which injury statuses should be treated as "not playing"
INACTIVE_STATUSES = {"out"}  # extend to {"out", "doubtful"} if desired

# Optional: injury report library (from official team reports)
try:
    from nbainjuries import injury as nbainjury  # type: ignore
except Exception:  # pragma: no cover - library optional
    nbainjury = None


# -------------------------
# Helpers
# -------------------------

def _load_model_with_fallback(*candidates: str):
    """
    Load the first model found from a list of candidate filenames inside the models directory.
    Preference is determined by the order of arguments.
    """
    for name in candidates:
        path = MODELS_DIR / name
        if path.exists():
            print(f"Loading model: {path}")
            return load(path)
    raise FileNotFoundError(
        "None of the candidate model files found in 'models': " + ", ".join(candidates)
    )

def _today_scoreboard_date_str() -> str:
    """Scoreboard expects MM/DD/YYYY."""
    return datetime.today().strftime("%m/%d/%Y")


def get_today_matchups() -> List[Dict[str, str]]:
    """Return list of today's games with team abbreviations."""
    sb = scoreboardv2.ScoreboardV2(game_date=_today_scoreboard_date_str())
    game_header = sb.game_header.get_data_frame()
    if game_header.empty:
        return []

    id_to_abbr = {t["id"]: t["abbreviation"] for t in teams.get_teams()}

    matchups = []
    for _, row in game_header.iterrows():
        home_abbr = id_to_abbr.get(int(row["HOME_TEAM_ID"]))
        away_abbr = id_to_abbr.get(int(row["VISITOR_TEAM_ID"]))
        if not home_abbr or not away_abbr:
            continue

        # Some Scoreboard responses use GAME_DATE_EST, others GAME_DATE; fall back to today.
        game_date_val = row.get("GAME_DATE", None)
        if game_date_val is None or pd.isna(game_date_val):
            game_date_val = row.get("GAME_DATE_EST", None)
        if game_date_val is None or pd.isna(game_date_val):
            game_date_val = datetime.today().date()

        game_date = pd.to_datetime(game_date_val).normalize()

        matchups.append(
            {
                "game_id": row["GAME_ID"],
                "home_abbr": home_abbr,
                "away_abbr": away_abbr,
                "game_date": game_date,
            }
        )
    return matchups


def _normalize_injury_player_name(raw_name: str) -> str:
    """
    Injury reports use 'Last, First' format; convert to 'First Last'
    to match PLAYER_NAME from stats endpoints.
    """
    if not isinstance(raw_name, str):
        return ""
    parts = [p.strip() for p in raw_name.split(",")]
    if len(parts) == 2:
        last, first = parts
        return f"{first} {last}"
    return raw_name.strip()


def get_inactive_players_for_today() -> Set[Tuple[str, str]]:
    """
    Use nbainjuries (if available) to get today's official injury report
    and build a set of (team_abbr, player_name) pairs that should be
    excluded from projections.

    We treat players with 'Current Status' == 'Out' as not playing.
    You can later extend this to include 'Doubtful', 'Suspension', etc.
    """
    inactive: Set[Tuple[str, str]] = set()

    if nbainjury is None:
        # Library not installed; caller can decide how to handle.
        print(
            "Injury filtering skipped: nbainjuries package not installed. "
            "Install with `pip install nbainjuries` and ensure Java is available "
            "if you want to exclude OUT players automatically."
        )
        return inactive

    try:
        # Injury report snapshots are timestamped in ET; using "now"
        # will request the most recent snapshot. Ask nbainjuries to
        # return a DataFrame directly.
        now = datetime.now()
        inj_df = nbainjury.get_reportdata(now, return_df=True)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Warning: failed to fetch injury report as DataFrame ({exc}); not filtering by injuries.")
        return inactive

    if inj_df is None or getattr(inj_df, "empty", True):
        print("Injury filtering: no injury data returned for today.")
        return inactive

    # Expect columns like: 'Game Date', 'Game Time', 'Matchup',
    # 'Team', 'Player Name', 'Current Status', 'Reason'
    required = {"Team", "Player Name", "Current Status"}
    if not required.issubset(set(inj_df.columns)):
        print("Warning: injury report missing expected columns; not filtering by injuries.")
        return inactive

    # Map team full name -> abbreviation
    full_to_abbr = {t["full_name"]: t["abbreviation"] for t in teams.get_teams()}

    for _, row in inj_df.iterrows():
        status = str(row["Current Status"]).strip().lower()

        # Only drop players whose status is in configured inactive set
        # or whose status string contains any inactive keyword.
        if not any(s in status for s in INACTIVE_STATUSES):
            continue

        team_full = str(row["Team"])
        team_abbr = full_to_abbr.get(team_full)
        if not team_abbr:
            continue

        player_name = _normalize_injury_player_name(str(row["Player Name"]))
        if not player_name:
            continue

        inactive.add((team_abbr, player_name))

    return inactive


def get_team_players(df_season: pd.DataFrame, team_abbr: str, min_games: int = 3) -> List[int]:
    """
    Return PLAYER_IDs who have played at least min_games for this team in the season data.
    """
    subset = df_season[df_season["TEAM_ABBREVIATION"] == team_abbr]
    counts = subset.groupby("PLAYER_ID").size().reset_index(name="n")
    return counts[counts["n"] >= min_games]["PLAYER_ID"].astype(int).tolist()


def get_latest_player_row(df_season: pd.DataFrame, player_id: int) -> pd.Series | None:
    """Last game row for a player in this season."""
    g = df_season[df_season["PLAYER_ID"] == player_id].sort_values("GAME_DATE")
    if g.empty:
        return None
    return g.iloc[-1]


def build_team_context_map(df_season: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Map TEAM_ABBREVIATION -> latest row with team rolling features.
    """
    ctx_cols = ["TEAM_PTS_L10", "TEAM_REB_L10", "TEAM_AST_L10", "TEAM_FG3M_L10"]
    context = {}
    for team_abbr, g in df_season.groupby("TEAM_ABBREVIATION"):
        last = g.sort_values("GAME_DATE").iloc[-1]
        context[team_abbr] = last[ctx_cols]
    return context


def build_future_row(
    last_row: pd.Series,
    team_abbr: str,
    opp_abbr: str,
    game_id: str,
    game_date: pd.Timestamp,
    is_home: int,
    team_ctx_map: Dict[str, pd.Series],
) -> pd.Series:
    """
    Build a single-row Series for the upcoming game using the player's latest history.
    """
    row = last_row.copy()

    # Basic game identifiers
    row["GAME_ID"] = game_id
    row["GAME_DATE"] = game_date
    row["TEAM_ABBREVIATION"] = team_abbr
    row["OPP_TEAM_ABBREVIATION"] = opp_abbr
    row["IS_HOME"] = is_home
    row["MATCHUP"] = f"{team_abbr} vs {opp_abbr}" if is_home else f"{team_abbr} @ {opp_abbr}"

    # Rest / B2B
    last_game_date = pd.to_datetime(last_row["GAME_DATE"])
    days_since_prior = max(int((game_date - last_game_date).days), 0)
    row["DAYS_SINCE_PRIOR"] = days_since_prior
    row["IS_B2B"] = 1 if days_since_prior == 1 else 0

    # Latest team context
    team_ctx = team_ctx_map.get(team_abbr)
    if team_ctx is not None:
        for col in team_ctx.index:
            row[col] = team_ctx[col]

    opp_ctx = team_ctx_map.get(opp_abbr)
    if opp_ctx is not None:
        row["OPP_TEAM_PTS_L10"] = opp_ctx.get("TEAM_PTS_L10", np.nan)
        row["OPP_TEAM_REB_L10"] = opp_ctx.get("TEAM_REB_L10", np.nan)
        row["OPP_TEAM_AST_L10"] = opp_ctx.get("TEAM_AST_L10", np.nan)
        row["OPP_TEAM_FG3M_L10"] = opp_ctx.get("TEAM_FG3M_L10", np.nan)

    # This weight is ignored at inference but keep shape consistent
    row["RECENCY_WEIGHT"] = 1.0
    return row


# -------------------------
# Main
# -------------------------

def main():
    current_season = SEASONS[-1]

    # Load artifacts saved by training scripts
    df_feat = pd.read_pickle(BASE_DIR / "df_feat.pkl")

    # Prefer ensemble models, then tuned, then base models
    minutes_model = _load_model_with_fallback(
        "minutes_model_ensemble.joblib",
        "minutes_model_tuned.joblib",
        "minutes_model.joblib",
    )
    stat_models = {
        "PTS": _load_model_with_fallback(
            "stat_model_pts_ensemble.joblib",
            "stat_model_pts_tuned.joblib",
            "stat_model_pts.joblib",
        ),
        "REB": _load_model_with_fallback(
            "stat_model_reb_ensemble.joblib",
            "stat_model_reb_tuned.joblib",
            "stat_model_reb.joblib",
        ),
        "AST": _load_model_with_fallback(
            "stat_model_ast_ensemble.joblib",
            "stat_model_ast_tuned.joblib",
            "stat_model_ast.joblib",
        ),
        "FG3M": _load_model_with_fallback(
            "stat_model_fg3m_ensemble.joblib",
            "stat_model_fg3m_tuned.joblib",
            "stat_model_fg3m.joblib",
        ),
        "PRA": _load_model_with_fallback(
            "stat_model_pra_ensemble.joblib",
            "stat_model_pra_tuned.joblib",
            "stat_model_pra.joblib",
        ),
    }

    # Select season slice (fallback to all if empty)
    df_season = df_feat[df_feat["SEASON"] == current_season]
    if df_season.empty:
        print(f"No rows for season {current_season}; using all data instead.")
        df_season = df_feat

    minutes_features = get_minutes_feature_cols(df_feat)
    stat_features = get_stat_feature_cols(df_feat)
    feature_cols_needed = sorted(set(minutes_features + stat_features))

    # Build context maps
    team_ctx_map = build_team_context_map(df_season)

    matchups = get_today_matchups()
    if not matchups:
        print("No games found for today.")
        return

    rows = []
    today_iso = pd.Timestamp(datetime.today().date())

    for matchup in matchups:
        game_id = matchup["game_id"]
        home_abbr = matchup["home_abbr"]
        away_abbr = matchup["away_abbr"]
        game_date = matchup["game_date"]

        home_players = get_team_players(df_season, home_abbr, min_games=3)
        away_players = get_team_players(df_season, away_abbr, min_games=3)

        print(
            f"\nGAME_ID {game_id}: projecting players for "
            f"{away_abbr} @ {home_abbr} "
            f"(home={len(home_players)}, away={len(away_players)})"
        )

        def project_team(player_ids: List[int], team_abbr: str, opp_abbr: str, is_home: int):
            for pid in player_ids:
                last_row = get_latest_player_row(df_season, pid)
                if last_row is None:
                    continue

                try:
                    future_row = build_future_row(
                        last_row=last_row,
                        team_abbr=team_abbr,
                        opp_abbr=opp_abbr,
                        game_id=game_id,
                        game_date=game_date,
                        is_home=is_home,
                        team_ctx_map=team_ctx_map,
                    )
                except Exception as exc:
                    print(f"Skipping player {pid} ({team_abbr}) due to error: {exc}")
                    continue

                # Build DataFrame for prediction with all needed columns
                row_dict = {col: future_row.get(col, np.nan) for col in feature_cols_needed}
                row_df = pd.DataFrame([row_dict])

                # Predict minutes (pass DataFrame so feature names are preserved)
                mins_pred = float(minutes_model.predict(row_df[minutes_features])[0])

                # Predict stats (also with DataFrame input)
                stat_preds = {}
                for stat in TARGET_STATS:
                    model = stat_models[stat]
                    pred = float(model.predict(row_df[stat_features])[0])
                    stat_preds[stat.lower()] = pred

                rows.append(
                    {
                        "game_id": game_id,
                        "game_date": today_iso,
                        "team": team_abbr,
                        "opponent": opp_abbr,
                        "is_home": is_home,
                        "player_id": pid,
                        "player_name": last_row.get("PLAYER_NAME"),
                        "minutes_pred": mins_pred,
                        **stat_preds,
                    }
                )

        project_team(home_players, home_abbr, away_abbr, is_home=1)
        project_team(away_players, away_abbr, home_abbr, is_home=0)

    if not rows:
        print("No player projections generated.")
        return

    df_proj = pd.DataFrame(rows)

    # Optional: filter out players who are officially OUT on today's injury report.
    inactive_players = get_inactive_players_for_today()
    if inactive_players:
        before = len(df_proj)
        df_proj = df_proj[
            ~df_proj.apply(
                lambda r: (str(r["team"]), str(r["player_name"])) in inactive_players,
                axis=1,
            )
        ].reset_index(drop=True)
        after = len(df_proj)
        print(f"Filtered out {before - after} players marked OUT on the injury report.")
    out_path = "player_projections_today.csv"
    df_proj.to_csv(out_path, index=False)
    print(f"\nSaved {len(df_proj)} player projections to {out_path}")


if __name__ == "__main__":
    main()
