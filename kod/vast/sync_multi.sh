#!/usr/bin/env bash
# Sync from several instances (Stage C split across hosts). Usage: sync_multi.sh PLAN.json HOST:PORT [HOST:PORT ...]
set -uo pipefail
PLAN="$1"; shift
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG="$ROOT/wyniki/sync_cron.log"
exec 9>"$ROOT/wyniki/.sync.lock"; flock -n 9 || exit 0
{
  date '+%F %T'
  for hp in "$@"; do
    rsync -az --timeout=120 -e "ssh -p ${hp#*:} -o ConnectTimeout=30 -o BatchMode=yes" \
      "root@${hp%%:*}:/workspace/ff/wyniki/" "$ROOT/wyniki/" --exclude 'locks/' --exclude 'gpu_check*.json' 2>&1 | grep -v -i 'welcome\|have fun'
  done
  cd "$ROOT/kod" && "$ROOT/.venv/bin/python" -m ffexp.sync_verify "$ROOT/wyniki" | head -1
  "$ROOT/.venv/bin/python" -m ffexp.monitor "$ROOT/plany/$PLAN" --results "$ROOT/wyniki"
} >> "$LOG" 2>&1
