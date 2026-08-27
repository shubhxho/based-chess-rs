#!/usr/bin/env bash
# Sable lab pipeline — datagen → (optional prepare) → SP gate / mix.
#
#   scripts/pipeline.sh                # one 12k-node datagen wave (foreground)
#   scripts/pipeline.sh datagen        # same
#   scripts/pipeline.sh datagen-bg     # continuous 12k-node daemon
#   scripts/pipeline.sh selfplay       # SP gate @ +25 (pauses datagen for RAM)
#   scripts/pipeline.sh selfplay-bg    # durable SP gate
#   scripts/pipeline.sh 3000           # Lichess+SP mix candidate path
#   scripts/pipeline.sh bg|all         # full lab supervisor
#
# Env: DATAGEN_NODES (default 12000) DATAGEN_POS DATAGEN_N
#      GATE_EPOCHS GATE_GAMES GATE_MIN_ELO SP_KEEP
#      BATCH LR PATIENCE  (MLX train.py)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
MODE=${1:-datagen}

# Stronger self-play labels for new shards; MLX stays on Apple GPU.
export DATAGEN_NODES=${DATAGEN_NODES:-12000}
export DATAGEN_POS=${DATAGEN_POS:-200000}
export DATAGEN_N=${DATAGEN_N:-4}
export GATE_EPOCHS=${GATE_EPOCHS:-50}
export GATE_GAMES=${GATE_GAMES:-400}
export GATE_MIN_ELO=${GATE_MIN_ELO:-25}
export SP_KEEP=${SP_KEEP:-8}
# MLX / train.py knobs used by ml_cycle → train_gate → train.py
# Tuned off the +20.9 near-miss (was 7 shards / 0.9 / 0.70 / 3e-3 / 10).
export BATCH=${BATCH:-16384}
export LR=${LR:-2.5e-3}
export PATIENCE=${PATIENCE:-14}
export EVAL_W=${EVAL_W:-0.88}
export OUT_SCALE=${OUT_SCALE:-0.68}
export MX_FORCE_GPU=${MX_FORCE_GPU:-1}

python3 scripts/daily_page.py >/dev/null 2>&1 || true
echo "=== pipeline mode=$MODE nodes=$DATAGEN_NODES epochs=$GATE_EPOCHS batch=$BATCH ==="

case "$MODE" in
  datagen|wave|fg)
    exec bash scripts/datagen_parallel.sh "$DATAGEN_POS" "$DATAGEN_NODES" "$DATAGEN_N" auto
    ;;
  datagen-bg|daemon)
    bash scripts/datagen_daemon.sh stop 2>/dev/null || true
    rm -f data/selfplay/.datagen_paused
    NODES="$DATAGEN_NODES" POS="$DATAGEN_POS" N="$DATAGEN_N" \
      nohup bash scripts/datagen_daemon.sh >>/tmp/datagen_daemon.log 2>&1 &
    echo "datagen daemon pid $! → /tmp/datagen_daemon.log (${DATAGEN_NODES}n × ${DATAGEN_N})"
    ;;
  selfplay|sp|gate)
    exec bash scripts/ml_cycle.sh "$GATE_EPOCHS" "$GATE_GAMES" "$GATE_MIN_ELO"
    ;;
  selfplay-bg|sp-bg|gate-bg)
    exec bash scripts/ml_cycle.sh bg "$GATE_EPOCHS" "$GATE_GAMES" "$GATE_MIN_ELO"
    ;;
  3000|push3000|mix)
    exec bash scripts/push_3000.sh "${2:-all}"
    ;;
  bg)
    echo "background: lab supervisor"
    exec bash scripts/lab_supervisor.sh start
    ;;
  all)
    exec bash scripts/lab_supervisor.sh all --web
    ;;
  status)
    python3 - <<'PY'
import json
from pathlib import Path
root = Path('.')
for p in [root/'web/gate_last.json', root/'data/selfplay/datagen_status.json']:
    if p.exists():
        d = json.loads(p.read_text())
        print(p.name, {k: d.get(k) for k in list(d)[:12]})
PY
    pgrep -fl 'datagen|ml_cycle|train_gate|arena|prepare_hf' | head -20 || true
    ;;
  *)
    echo "usage: $0 [datagen|datagen-bg|selfplay|selfplay-bg|3000|bg|all|status]" >&2
    exit 2
    ;;
esac
