#!/usr/bin/env bash
# Resumable daily ingest runner for NBA logs/rosters.
# - Computes current+next season strings.
# - Calls scripts/01_ingest_nba_game_logs.py with retries/resume.
# - Logs to logs/ingest_daily.log.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

LOG_DIR="$REPO_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/ingest_daily.log"

# Compute current season (YYYY-YY). We skip future season to avoid empty rosters/logs.
SEASONS=$(python - <<'PY'
import datetime
today = datetime.date.today()
start = today.year if today.month >= 10 else today.year - 1
curr = f"{start}-{(start + 1) % 100:02d}"
print(curr)
PY
)

echo "[info] $(date -Iseconds) starting ingest for seasons: $SEASONS" >> "$LOG_FILE"

python scripts/01_ingest_nba_game_logs.py \
  --seasons "$SEASONS" \
  --out data/raw_nba \
  --sleep 2.00 \
  --max-retries 10 \
  --timeout 120 \
  >> "$LOG_FILE" 2>&1

echo "[done] $(date -Iseconds) ingest complete" >> "$LOG_FILE"
