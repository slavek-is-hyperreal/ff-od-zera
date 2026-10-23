#!/usr/bin/env bash
# One sync pass for cron (decision D5c): pull results, confirm checkpoint hashes, log counts only.
# Usage (crontab, every 10 min):
#   */10 * * * * /vectorlegis_ssd_pool/uczelnia/eksperyment_ff_od_zera/kod/vast/sync_cron.sh HOST PORT PLAN.json
set -uo pipefail
HOST="$1"; PORT="$2"; PLAN="$3"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG="$ROOT/wyniki/sync_cron.log"
exec 9>"$ROOT/wyniki/.sync.lock"; flock -n 9 || exit 0   # skip if the previous pass is still running
{
  date '+%F %T'
  rsync -az --timeout=120 -e "ssh -p $PORT -o ConnectTimeout=30 -o BatchMode=yes" \
    "root@$HOST:/workspace/ff/wyniki/" "$ROOT/wyniki/" --exclude 'locks/' 2>&1 | grep -v -i 'welcome\|have fun'
  cd "$ROOT/kod" && "$ROOT/.venv/bin/python" -m ffexp.sync_verify "$ROOT/wyniki" | head -1
  "$ROOT/.venv/bin/python" -m ffexp.monitor "$ROOT/plany/$PLAN" --results "$ROOT/wyniki"
} >> "$LOG" 2>&1
