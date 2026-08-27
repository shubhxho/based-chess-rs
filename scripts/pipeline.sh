#!/usr/bin/env bash
# Run the full lab loop: daily snapshot, datagen wave, Lichess resume, gated train.
#
#   scripts/pipeline.sh           # one wave + SP gate (foreground)
#   scripts/pipeline.sh bg        # datagen + prepare in background
#   scripts/pipeline.sh 3000      # Lichess+SP mix candidate path (push_3000)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
MODE=${1:-fg}

python3 scripts/daily_page.py
echo "=== daily page refreshed ==="

run_datagen() {
  bash scripts/datagen_parallel.sh 200000 10000 4 auto
}

run_prepare() {
  bash scripts/prepare_lichess.sh
}

run_gate() {
  bash scripts/ml_cycle.sh 35 400 25
}

if [[ "$MODE" == "3000" || "$MODE" == "push3000" ]]; then
  exec bash scripts/push_3000.sh all
fi

if [[ "$MODE" == "bg" ]]; then
  echo "background: starting the single lab supervisor"
  exec bash scripts/lab_supervisor.sh start
fi

# The supervisor owns every long-running worker and queues the gated attempt.
# This avoids data preparation and another net writer racing the gate.
if [[ "$MODE" == "all" ]]; then
  exec bash scripts/lab_supervisor.sh all --web
fi
exec bash scripts/lab_supervisor.sh all
