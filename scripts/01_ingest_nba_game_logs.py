"""
Fetch NBA game logs via nba_api and store as parquet for model training.

Robust/resumable behavior:
- If roster/game log parquet already exists, reuse it and only fetch missing teams/players.
- Retries with exponential backoff for nba_api timeouts/rate limits.
- After partial/interrupted runs, re-run the script; it fills only missing pieces.

Usage:
  python scripts/01_ingest_nba_game_logs.py --seasons 2024-25,2025-26 --out data/raw_nba

Notes:
- Requires network access for nba_api.
- Timeouts on stats.nba.com are common; this script retries and resumes instead of failing.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Set

import pandas as pd
from nba_api.stats.endpoints import commonteamroster, playergamelog
from nba_api.stats.static import teams
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Ingest NBA game logs to parquet (resumable).")
    ap.add_argument("--seasons", type=str, required=True, help="Comma-separated seasons, e.g., 2024-25,2025-26")
    ap.add_argument("--season-type", type=str, default="Regular Season")
    ap.add_argument("--out", type=Path, default=Path("data/raw_nba"))
    ap.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between calls.")
    ap.add_argument("--max-retries", type=int, default=3, help="Retries per request on timeout/rate-limit.")
    ap.add_argument("--timeout", type=float, default=60.0, help="Request timeout seconds for nba_api calls.")
    return ap.parse_args()


def safe_sleep(s: float) -> None:
    if s > 0:
        time.sleep(s)


def load_existing(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def list_team_abbrs() -> Set[str]:
    return {t["abbreviation"] for t in teams.get_teams()}


def retry_wrap(max_attempts: int, desc: str):
    """Return a retry decorator with exponential backoff."""
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )


def fetch_team_roster(team_id: int, season: str, timeout: float) -> pd.DataFrame:
    return commonteamroster.CommonTeamRoster(team_id=team_id, season=season, timeout=timeout).get_data_frames()[0]


def fetch_player_logs(player_id: int, season: str, season_type: str, timeout: float) -> pd.DataFrame:
    return playergamelog.PlayerGameLog(
        player_id=int(player_id),
        season=season,
        season_type_all_star=season_type,
        timeout=timeout,
    ).get_data_frames()[0]


def fetch_rosters(season: str, existing: Optional[pd.DataFrame], max_retries: int, timeout: float, sleep: float) -> pd.DataFrame:
    have = set()
    if existing is not None and not existing.empty:
        have = set(existing["TEAM_ABBR"].unique())

    need = list(list_team_abbrs() - have)
    rows = [existing] if existing is not None and not existing.empty else []

    for t in teams.get_teams():
        team_id = t["id"]
        abbrev = t["abbreviation"]
        if abbrev not in need:
            continue
        try:
            fn = retry_wrap(max_retries, f"roster {abbrev}")(
                lambda: fetch_team_roster(team_id, season, timeout)
            )
            df = fn()
            df["TEAM_ABBR"] = abbrev
            rows.append(df)
            safe_sleep(sleep)
        except Exception as exc:  # pragma: no cover - network/endpoint errors
            print(f"[warn] roster fetch failed for {abbrev}: {exc}")

    if not rows:
        # Return empty DF to signal nothing fetched (e.g., future season)
        return pd.DataFrame()

    roster_df = pd.concat(rows, ignore_index=True)
    return roster_df


def fetch_logs_for_season(
    season: str,
    roster_df: pd.DataFrame,
    season_type: str,
    sleep: float,
    max_retries: int,
    timeout: float,
    existing_logs: Optional[pd.DataFrame],
) -> pd.DataFrame:
    frames = []
    if existing_logs is not None:
        if "SEASON" in existing_logs.columns:
            existing_logs = existing_logs[existing_logs["SEASON"] == season]
        frames.append(existing_logs)

    # Determine which players need fetching (missing or empty logs)
    existing_pids: Set[int] = set()
    if existing_logs is not None and not existing_logs.empty:
        existing_pids = set(existing_logs["PLAYER_ID"].unique())

    roster_pids = roster_df["PLAYER_ID"].dropna().unique().tolist()
    need_pids = [pid for pid in roster_pids if pid not in existing_pids]

    for pid in need_pids:
        team_abbr = roster_df.loc[roster_df["PLAYER_ID"] == pid, "TEAM_ABBR"].iloc[0]
        try:
            fn = retry_wrap(max_retries, f"logs pid {pid}")(
                lambda: fetch_player_logs(pid, season, season_type, timeout)
            )
            gl = fn()
            if gl.empty:
                continue
            gl["PLAYER_ID"] = pid
            gl["TEAM_ABBR"] = team_abbr
            gl["SEASON"] = season
            frames.append(gl)
        except Exception as exc:  # pragma: no cover - network/endpoint errors
            print(f"[warn] game log fetch failed for player {pid} season {season}: {exc}")
        safe_sleep(sleep)

    if not frames:
        # If no new frames and no existing_logs, return empty to let caller decide
        return pd.DataFrame()

    logs_df = pd.concat(frames, ignore_index=True)
    # Deduplicate in case of re-fetch
    if "GAME_ID" in logs_df.columns:
        logs_df = logs_df.drop_duplicates(subset=["PLAYER_ID", "GAME_ID"], keep="last")
    return logs_df


def main():
    args = parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    for season in seasons:
        print(f"[info] processing season {season}")
        roster_path = out_dir / f"rosters_{season}.parquet"
        existing_roster = load_existing(roster_path)
        roster_df = fetch_rosters(
            season=season,
            existing=existing_roster,
            max_retries=args.max_retries,
            timeout=args.timeout,
            sleep=args.sleep,
        )
        if roster_df.empty:
            print(f"[warn] roster fetch returned empty for season {season}; skipping logs.")
            roster_path.parent.mkdir(parents=True, exist_ok=True)
            roster_df.to_parquet(roster_path, index=False)
            continue
        roster_df.to_parquet(roster_path, index=False)
        print(f"[ok] wrote {roster_path} (rows={len(roster_df)})")

        logs_path = out_dir / f"game_logs_{season}.parquet"
        existing_logs = load_existing(logs_path)
        logs_df = fetch_logs_for_season(
            season=season,
            roster_df=roster_df,
            season_type=args.season_type,
            sleep=args.sleep,
            max_retries=args.max_retries,
            timeout=args.timeout,
            existing_logs=existing_logs,
        )
        if logs_df.empty:
            print(f"[warn] no game logs fetched for season {season} (roster rows={len(roster_df)}); keeping existing if any.")
            if existing_logs is not None:
                logs_df = existing_logs
        logs_df.to_parquet(logs_path, index=False)
        print(f"[ok] wrote {logs_path} with {len(logs_df)} rows")


if __name__ == "__main__":
    main()
