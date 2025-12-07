Cached data lives here when you run `python scripts/prefetch.py`.

Files produced:
- `rosters_<season>.csv`: Team rosters for the roster season (one row per player, includes TEAM_ID/TEAM_ABBR).
- `game_logs_<season>.csv`: All player game logs for that season with PLAYER_ID and SEASON columns.
- `metadata.json`: Last run timestamp (UTC) plus seasons/roster season used.

Cron-friendly command (12am Eastern / 05:00 UTC):
- `0 5 * * * cd /home/banana-bread/nba_stats_prediction && /usr/bin/env python3 scripts/prefetch.py --seasons 2024-25,2025-26 >> logs/prefetch.log 2>&1`

Notes:
- Adjust `--seasons` as the calendar rolls; set `--roster-season` if you want rosters from a different year than the game logs.
- Add a short sleep (default 0.6s) to reduce nba_api rate-limit risk; increase if you see warnings.
- If running in Docker, bind-mount `data/` and call `python scripts/prefetch.py` inside the container.
- The Streamlit app has a sidebar toggle to use these cached files first and only hit the live API when a cache is missing.
