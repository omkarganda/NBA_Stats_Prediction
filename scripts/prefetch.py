"""
Prefetch NBA data from nba_api and store as CSVs for offline use.

What it does:
  - Pulls current rosters for a target season
  - Derives player IDs from those rosters
  - Downloads per-game logs for selected seasons
  - Writes CSVs under data/ and a metadata JSON with last-run info

Usage (examples):
  python scripts/prefetch.py
  python scripts/prefetch.py --seasons 2024-25,2025-26 --roster-season 2025-26
  python scripts/prefetch.py --data-dir data --sleep 0.75

Notes:
  - Respects a small sleep between API calls to avoid rate limits.
  - Requires network access for nba_api.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd
from nba_api.stats.endpoints import commonteamroster, playergamelog
from nba_api.stats.static import teams


DEFAULT_SEASONS = ["2024-25", "2025-26"]
DEFAULT_SEASON_TYPE = "Regular Season"
DEFAULT_SLEEP = 0.6  # seconds between requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefetch NBA data to CSVs.")
    parser.add_argument(
        "--seasons",
        type=str,
        default=",".join(DEFAULT_SEASONS),
        help="Comma-separated list of seasons to fetch (e.g., 2023-24,2024-25).",
    )
    parser.add_argument(
        "--roster-season",
        type=str,
        default=None,
        help="Season to use for rosters (defaults to last season in --seasons).",
    )
    parser.add_argument(
        "--season-type",
        type=str,
        default=DEFAULT_SEASON_TYPE,
        help='Season type passed to nba_api (e.g., "Regular Season").',
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory where CSVs and metadata.json will be written.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP,
        help="Seconds to sleep between API calls (helps avoid rate limits).",
    )
    return parser.parse_args()


def safe_sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def fetch_rosters(roster_season: str, data_dir: Path, sleep: float) -> pd.DataFrame:
    """
    Fetch rosters for all teams for the given season and write CSV.
    Returns the combined roster DataFrame.
    """
    rows = []
    team_list = teams.get_teams()
    for team_info in team_list:
        team_id = team_info["id"]
        abbrev = team_info["abbreviation"]
        try:
            roster = commonteamroster.CommonTeamRoster(
                team_id=team_id, season=roster_season
            ).get_data_frames()[0]
            roster["TEAM_ID"] = team_id
            roster["TEAM_ABBR"] = abbrev
            rows.append(roster)
        except Exception as exc:  # pragma: no cover - network/endpoint errors
            print(f"[warn] roster fetch failed for {abbrev} ({team_id}): {exc}")
        safe_sleep(sleep)

    if not rows:
        raise RuntimeError("No roster data fetched. Check network or season format.")

    df = pd.concat(rows, ignore_index=True)
    output_path = data_dir / f"rosters_{roster_season}.csv"
    df.to_csv(output_path, index=False)
    print(f"[ok] wrote {output_path} with {len(df)} rows")
    return df


def fetch_game_logs_for_season(
    season: str,
    player_ids: Iterable[int],
    season_type: str,
    data_dir: Path,
    sleep: float,
) -> None:
    """
    Fetch game logs for each player in a season and write one CSV per season.
    """
    frames: List[pd.DataFrame] = []
    for pid in player_ids:
        try:
            gl = playergamelog.PlayerGameLog(
                player_id=int(pid),
                season=season,
                season_type_all_star=season_type,
            ).get_data_frames()[0]
            if gl.empty:
                continue
            gl["PLAYER_ID"] = pid
            gl["SEASON"] = season
            frames.append(gl)
        except Exception as exc:  # pragma: no cover - network/endpoint errors
            print(f"[warn] game log fetch failed for player {pid} season {season}: {exc}")
        safe_sleep(sleep)

    if not frames:
        print(f"[warn] no game logs fetched for season {season}")
        return

    df = pd.concat(frames, ignore_index=True)
    # Ensure GAME_DATE is ISO-like for easier downstream parsing.
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.strftime("%Y-%m-%d")
    output_path = data_dir / f"game_logs_{season}.csv"
    df.to_csv(output_path, index=False)
    print(f"[ok] wrote {output_path} with {len(df)} rows")


def write_metadata(data_dir: Path, seasons: Sequence[str], roster_season: str) -> None:
    meta = {
        "last_run_utc": dt.datetime.utcnow().isoformat() + "Z",
        "seasons": list(seasons),
        "roster_season": roster_season,
    }
    path = data_dir / "metadata.json"
    path.write_text(json.dumps(meta, indent=2))
    print(f"[ok] wrote {path}")


def main() -> None:
    args = parse_args()
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    if not seasons:
        raise ValueError("Provide at least one season via --seasons.")

    roster_season = args.roster_season or seasons[-1]
    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[info] fetching rosters for {roster_season} "
        f"and game logs for seasons: {', '.join(seasons)}"
    )
    roster_df = fetch_rosters(roster_season, data_dir, args.sleep)
    player_ids = roster_df["PLAYER_ID"].dropna().unique().tolist()
    print(f"[info] found {len(player_ids)} unique player IDs from rosters")

    for season in seasons:
        fetch_game_logs_for_season(
            season=season,
            player_ids=player_ids,
            season_type=args.season_type,
            data_dir=data_dir,
            sleep=args.sleep,
        )

    write_metadata(data_dir, seasons, roster_season)
    print("[done] prefetch complete")


if __name__ == "__main__":
    main()
