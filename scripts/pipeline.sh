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
  rm -f data/lichess-sf/.prepare_hf.lock
  .venv/bin/python prepare_hf.py data/lichess-sf --max-positions 500000 --resume
}

run_gate() {
  bash scripts/ml_cycle.sh 35 400 25
}

if [[ "$MODE" == "bg" || "$MODE" == "all" ]]; then
  echo "background: datagen daemon + lichess prepare"
  nohup bash scripts/datagen_daemon.sh >> /tmp/datagen_daemon.log 2>&1 &
  echo "  datagen: /tmp/datagen_daemon.log (pid $!)"
  ( run_prepare >> /tmp/prepare_resume.log 2>&1 ) &
  echo "  prepare: /tmp/prepare_resume.log (pid $!)"
  echo "  gate when ready: bash scripts/ml_cycle.sh 35 400 25"
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
