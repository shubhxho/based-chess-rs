#!/usr/bin/env bash
# Continuous datagen waves until interrupted. Each wave refreshes the daily page.
#
#   scripts/datagen_daemon.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
echo "datagen_daemon: 200k @ 8k nodes, 4 shards, auto index (Ctrl-C to stop)"
while true; do
  bash scripts/datagen_parallel.sh 200000 8000 4 auto || echo "wave failed; retry in 60s" >&2
  sleep 10
done
