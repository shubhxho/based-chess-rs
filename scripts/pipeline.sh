#!/usr/bin/env bash
# Sable lab pipeline — datagen → (optional prepare) → SP gate / mix.
#
#   scripts/pipeline.sh                # one 12k-node datagen wave (foreground)
#   scripts/pipeline.sh datagen        # same
#   scripts/pipeline.sh datagen-bg     # continuous 12k-node daemon
#   scripts/pipeline.sh selfplay       # SP gate @ +25 (pauses datagen for RAM)
#   scripts/pipeline.sh selfplay-bg    # durable SP gate
#   scripts/pipeline.sh bench          # engine bench + SP arena smoke + SF calibrate
#   scripts/pipeline.sh stress         # deeper bench + longer SP arena smoke
#   scripts/pipeline.sh arena          # candidate vs shipping arena (no retrain)
#   scripts/pipeline.sh 3000           # Lichess+SP mix candidate path
#   scripts/pipeline.sh bg|all         # full lab supervisor
#
# Env: DATAGEN_NODES (default 12000) DATAGEN_POS DATAGEN_N
#      GATE_EPOCHS GATE_GAMES GATE_MIN_ELO SP_KEEP
#      BATCH LR PATIENCE  (MLX train.py)
#      BENCH_DEPTH ARENA_NODES (bench / arena)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
MODE=${1:-datagen}

# Stronger self-play labels for new shards; MLX stays on Apple GPU.
export DATAGEN_NODES=${DATAGEN_NODES:-12000}
export DATAGEN_POS=${DATAGEN_POS:-200000}
export DATAGEN_N=${DATAGEN_N:-4}
export GATE_EPOCHS=${GATE_EPOCHS:-45}
export GATE_GAMES=${GATE_GAMES:-400}
export GATE_MIN_ELO=${GATE_MIN_ELO:-25}
export SP_KEEP=${SP_KEEP:-7}
# MLX / train.py — GPU + mem_used_mb logging; proven SP recipe (SEED=42).
export BATCH=${BATCH:-16384}
export LR=${LR:-3e-3}
export LR_FLOOR=${LR_FLOOR:-0.08}
export PATIENCE=${PATIENCE:-10}
export MIN_EPOCHS=${MIN_EPOCHS:-20}
export EVAL_EVERY=${EVAL_EVERY:-1}
export WARMUP=${WARMUP:-2}
export EVAL_W=${EVAL_W:-0.9}
export OUT_SCALE=${OUT_SCALE:-0.70}
export SEED=${SEED:-42}
export SHARD_DECAY=${SHARD_DECAY:-1.0}
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
  bench|engine|benchmark)
    # Engine throughput + short SP arena smoke + optional Stockfish calibrate.
    cargo build --release
    ENG=${ENGINE:-$ROOT/target/release/sable}
    echo "=== engine bench (depth ${BENCH_DEPTH:-12}) ==="
    printf 'bench %s\nquit\n' "${BENCH_DEPTH:-12}" | "$ENG" | tee /tmp/sable_engine_bench.log
    echo "=== SP arena smoke: shipping vs shipping (40 games) ==="
    .venv/bin/python arena.py "$ENG" "$ENG" 40 "nodes ${ARENA_NODES:-20000}" 4 \
      | tee /tmp/sable_sp_arena_smoke.log
    if command -v stockfish >/dev/null 2>&1; then
      echo "=== Stockfish calibrate (movetime 100, 20 games/level) ==="
      : > /tmp/sable_calibrate.log
      .venv/bin/python tests/calibrate.py "$ENG" "movetime 100" 20 \
        2600 2700 2800 2900 3000 | tee /tmp/sable_calibrate.log
    else
      echo "stockfish not on PATH — skip calibrate" | tee /tmp/sable_calibrate.log
    fi
    # Drop empty/aborted calibrate stubs so the lab board does not treat them as live.
    if [[ ! -s /tmp/sable_calibrate.log ]]; then
      echo "calibrate aborted (empty log)" | tee /tmp/sable_calibrate.log
    fi
    echo "bench logs: /tmp/sable_engine_bench.log /tmp/sable_sp_arena_smoke.log /tmp/sable_calibrate.log"
    ;;
  arena|sp-arena)
    # Arena an existing candidate against shipping (no retrain).
    cargo build --release
    ENG=${ENGINE:-$ROOT/target/release/sable}
    if [[ ! -f net-candidate.bin ]]; then
      echo "need net-candidate.bin" >&2
      exit 1
    fi
    cp -f net.bin.ship net.bin
    cargo build --release
    cp -f "$ENG" /tmp/sable-shipping
    cp -f net-candidate.bin net.bin
    cargo build --release
    cp -f "$ENG" /tmp/sable-candidate
    cp -f net.bin.ship net.bin
    cargo build --release
    exec .venv/bin/python arena.py /tmp/sable-candidate /tmp/sable-shipping \
      "${GATE_GAMES}" "nodes ${ARENA_NODES:-20000}" 4
    ;;
  stress|stress-test)
    # Heavier engine stress: deeper UCI bench + longer SP arena smoke.
    cargo build --release
    ENG=${ENGINE:-$ROOT/target/release/sable}
    DEPTH=${BENCH_DEPTH:-14}
    SMOKE=${STRESS_GAMES:-80}
    NODES=${ARENA_NODES:-20000}
    echo "=== stress engine bench (depth $DEPTH) ==="
    printf 'bench %s\nquit\n' "$DEPTH" | "$ENG" | tee /tmp/sable_engine_stress.log
    echo "=== stress SP arena smoke ($SMOKE games @ ${NODES}n) ==="
    .venv/bin/python arena.py "$ENG" "$ENG" "$SMOKE" "nodes $NODES" 4 \
      | tee /tmp/sable_sp_arena_stress.log
    if [[ "${STRESS_AB:-0}" == "1" ]]; then
      echo "=== presearch A/B stress ==="
      bash tests/presearch_ab.sh "${STRESS_BASE:-60ab7d3}" "${STRESS_AB_GAMES:-100}" "$NODES" 4 \
        | tee /tmp/sable_presearch_ab.log
    fi
    echo "stress logs: /tmp/sable_engine_stress.log /tmp/sable_sp_arena_stress.log"
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
    echo "usage: $0 [datagen|datagen-bg|selfplay|selfplay-bg|bench|stress|arena|3000|bg|all|status]" >&2
    exit 2
    ;;
esac
