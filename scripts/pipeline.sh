#!/usr/bin/env bash
# Run the full lab loop: daily snapshot, datagen wave, Lichess resume, gated train.
#
#   scripts/pipeline.sh           # one wave + gate (foreground)
#   scripts/pipeline.sh bg        # datagen + prepare in background, gate when SP ready
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
MODE=${1:-fg}

python3 scripts/daily_page.py
echo "=== daily page refreshed ==="

run_datagen() {
  bash scripts/datagen_parallel.sh 200000 8000 4 auto
}

run_prepare() {
  bash scripts/prepare_lichess.sh
}

run_gate() {
  bash scripts/ml_cycle.sh 35 400 25
}

if [[ "$MODE" == "bg" || "$MODE" == "all" ]]; then
  echo "background: lab supervisor (datagen daemon + lichess + auto-restart)"
  nohup bash scripts/lab_supervisor.sh bg >> /tmp/lab_supervisor.log 2>&1 &
  echo "  supervisor: /tmp/lab_supervisor.log (pid $!)"
  echo "  UI: bash scripts/lab.sh"
  exit 0
fi

echo "=== datagen wave ==="
run_datagen
echo "=== lichess resume (500k) ==="
run_prepare
echo "=== gated SP train ==="
run_gate
echo "=== pipeline done — open http://127.0.0.1:8375/daily ==="
