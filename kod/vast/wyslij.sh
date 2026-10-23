#!/usr/bin/env bash
# Copy code + data to a Vast.ai instance and verify data hashes there (PROTOKOL_v5 §4.1, §8).
# Usage: vast/wyslij.sh SSH_HOST SSH_PORT
set -euo pipefail
HOST="$1"; PORT="$2"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"   # eksperyment_ff_od_zera
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new"
$SSH "root@$HOST" "mkdir -p /workspace/ff/kod /workspace/ff/dane /workspace/ff/plany /workspace/ff/wyniki"
rsync -az --delete -e "$SSH" --exclude '__pycache__' "$ROOT/kod/" "root@$HOST:/workspace/ff/kod/"
rsync -az -e "$SSH" "$ROOT/dane/" "root@$HOST:/workspace/ff/dane/"
[ -d "$ROOT/plany" ] && rsync -az -e "$SSH" "$ROOT/plany/" "root@$HOST:/workspace/ff/plany/"
$SSH "root@$HOST" "cd /workspace/ff/kod && python -c 'from ffexp import data; [data.verify_file(r) for r in data.load_sums()]; print(\"hashe danych: OK\")' && python -c 'import torch,numpy,platform; print(platform.python_version(), torch.__version__, torch.version.cuda, numpy.__version__, torch.cuda.get_device_name(0))'"
