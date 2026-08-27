#!/usr/bin/env bash
# Canonical SP-first gated train. Never ships a net that loses the arena.
#
# Best measured: +23.5 @ EVAL_W=0.9 OUT_SCALE=0.70 LR=3e-3 SP_KEEP=7.
# The 8-shard / 0.88 / 0.68 / 2.5e-3 nudge measured −20 — stay on the proven set.
# A 3.2M newest window measured −6.9; default ~1.4M (7 full 200k shards).
# Datagen engines fight train for RAM — pause them for the gate window.
#
#   scripts/ml_cycle.sh              # foreground gate @ +25
#   scripts/ml_cycle.sh bg           # durable background (survives shell teardown)
#   scripts/ml_cycle.sh 45 400 25
#   SP_KEEP=10 scripts/ml_cycle.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

if [[ "${1:-}" == "bg" ]]; then
  shift || true
  LOG=${LOG:-/tmp/ml_cycle.log}
  : >"$LOG"
  .venv/bin/python - <<PY
import subprocess
from pathlib import Path
root = Path("$ROOT")
log = Path("$LOG")
args = ["bash", str(root / "scripts" / "ml_cycle.sh")] + """$*""".split()
args = [a for a in args if a]
out = open(log, "a", buffering=1)
subprocess.Popen(
    args,
    cwd=str(root),
    stdin=subprocess.DEVNULL,
    stdout=out,
    stderr=subprocess.STDOUT,
    start_new_session=True,
    close_fds=True,
)
print(f"ml_cycle bg → {log}", flush=True)
PY
  sleep 2
  tail -15 "$LOG" || true
  exit 0
fi

EPOCHS=${1:-45}
GAMES=${2:-400}
MIN_ELO=${3:-25}

export DATA_DIR=data/selfplay
export DATA_GLOB='aug_sp_0*.txt'
export EVAL_W=${EVAL_W:-0.9}
export OUT_SCALE=${OUT_SCALE:-0.70}
export WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
export PATIENCE=${PATIENCE:-10}
export MIN_EPOCHS=${MIN_EPOCHS:-20}
export EVAL_EVERY=${EVAL_EVERY:-5}
export LR_FLOOR=${LR_FLOOR:-0.08}
export SP_BOOST=${SP_BOOST:-1.0}
export MIN_SHARD=${MIN_SHARD:-0}
export SHARD_DECAY=${SHARD_DECAY:-1.0}
export LR=${LR:-3e-3}
export BATCH=${BATCH:-16384}
export ENGINE=${ENGINE:-$ROOT/target/release/sable}
export MX_FORCE_GPU=${MX_FORCE_GPU:-1}

PAUSE_DG=${PAUSE_DG:-$ROOT/data/selfplay/.datagen_paused}
PAUSE_PREP=${PAUSE_PREP:-$ROOT/data/lichess-sf/.prepare_paused}

pause_workers() {
  mkdir -p "$(dirname "$PAUSE_DG")" "$(dirname "$PAUSE_PREP")"
  touch "$PAUSE_DG" "$PAUSE_PREP"
  echo "pausing datagen + prepare for SP gate RAM"
  bash scripts/datagen_daemon.sh stop 2>/dev/null || true
  pkill -f 'datagen_parallel.sh' 2>/dev/null || true
  pkill -9 -f 'prepare_hf.py' 2>/dev/null || true
  # Stop stray sable datagen children (not the gate arena binaries).
  pkill -f 'target/release/sable' 2>/dev/null || true
  sleep 2
}

resume_workers() {
  rm -f "$PAUSE_DG"
  # Leave prepare paused only if caller wants; default resume both.
  rm -f "$PAUSE_PREP"
  if ! pgrep -f 'datagen_daemon.sh' >/dev/null 2>&1; then
    echo "resuming datagen @ ${DATAGEN_NODES:-12000}n"
    NODES=${DATAGEN_NODES:-12000} POS=${DATAGEN_POS:-200000} N=${DATAGEN_N:-4} \
      nohup bash scripts/datagen_daemon.sh >>/tmp/datagen_daemon.log 2>&1 &
  fi
  if ! pgrep -f 'prepare_hf.py data/lichess-sf' >/dev/null 2>&1; then
    nohup bash scripts/prepare_lichess.sh >>/tmp/prepare_resume.log 2>&1 &
  fi
}

# Full finished shards only.
min_lines=${MIN_LINES:-200000}
# 7×200k ≈ 1.4M — +19.1 / +23.5 / +20.9 sweet spot (8-shard nudge → −20).
SP_KEEP=${SP_KEEP:-7}
shopt -s nullglob
all_ready=()
for f in data/selfplay/aug_sp_0*.txt; do
  lines=$(wc -l <"$f" | tr -d ' ')
  if (( lines >= min_lines )); then
    all_ready+=("$f")
  else
    echo "  skip incomplete $(basename "$f") ($lines < $min_lines lines)" >&2
  fi
done
ready=()
if (( ${#all_ready[@]} > SP_KEEP )); then
  start=$((${#all_ready[@]} - SP_KEEP))
  ready=("${all_ready[@]:$start}")
  echo "  SP_KEEP=$SP_KEEP: newest ${#ready[@]}/${#all_ready[@]} full shards (≥$min_lines)" >&2
else
  ready=("${all_ready[@]}")
fi
if (( ${#ready[@]} < 2 )); then
  echo "need ≥2 finished shards (≥$min_lines lines); have ${#ready[@]}" >&2
  exit 1
fi

n=$(cat "${ready[@]}" | wc -l | tr -d ' ')
echo "ml_cycle: $n lines in ${#ready[@]} shards → epochs=$EPOCHS games=$GAMES min-elo=$MIN_ELO"
echo "  PATIENCE=$PATIENCE MIN_EPOCHS=$MIN_EPOCHS LR=$LR BATCH=$BATCH EVAL_W=$EVAL_W OUT_SCALE=$OUT_SCALE EVAL_EVERY=$EVAL_EVERY"
if (( n < 800000 )); then
  echo "warning: under 800k SP lines; gate will likely reject" >&2
fi

tmp=$(mktemp -d)
pause_workers
trap 'resume_workers; rm -rf "$tmp"' EXIT

for f in "${ready[@]}"; do
  ln -s "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" "$tmp/$(basename "$f")"
done
export DATA_DIR="$tmp"
export DATA_GLOB='aug_sp_0*.txt'
export REPORT_DATA_DIR="$ROOT/data/selfplay"
export REPORT_DATA_GLOB='aug_sp_0*.txt'

.venv/bin/python train_gate.py --epochs "$EPOCHS" --games "$GAMES" --min-elo "$MIN_ELO"
unset DATA_DIR DATA_GLOB
python3 scripts/daily_page.py
python3 scripts/blog_page.py
