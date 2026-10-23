#!/usr/bin/env bash
# Pull results every INTERVAL seconds and confirm checkpoint hashes locally (PROTOKOL_v5 §8).
# Prints counts only (no accuracies). Usage: vast/synchronizuj.sh SSH_HOST SSH_PORT PLAN_NAME [INTERVAL]
set -uo pipefail
HOST="$1"; PORT="$2"; PLAN="$3"; INTERVAL="${4:-300}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/.venv/bin/python"
while true; do
  rsync -az -e "ssh -p $PORT" "root@$HOST:/workspace/ff/wyniki/" "$ROOT/wyniki/" --exclude 'locks/'
  (cd "$ROOT/kod" && "$PY" -m ffexp.sync_verify "$ROOT/wyniki" && "$PY" -m ffexp.monitor "$ROOT/plany/$PLAN" --results "$ROOT/wyniki")
  date '+%H:%M:%S'
  sleep "$INTERVAL"
done
