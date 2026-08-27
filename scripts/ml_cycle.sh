#!/usr/bin/env bash
# Canonical SP-first gated train. Never ships a net that loses the arena.
#
# Best measured: +23.5 on finished SP @ EVAL_W=0.9 OUT_SCALE=0.70 LR=3e-3.
# Lab all-time +19.1 was on ~1.34M lines — not every shard on disk. Newest-only
# 3.2M windows have also measured −6.9, so default to ~8 full 200k shards
# (~1.6M) with more epochs/patience.
#
#   scripts/ml_cycle.sh              # newest full shards → gate @ +25
#   scripts/ml_cycle.sh 45 400 25
#   SP_KEEP=12 scripts/ml_cycle.sh   # larger window
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
EPOCHS=${1:-45}
GAMES=${2:-400}
MIN_ELO=${3:-25}

export DATA_DIR=data/selfplay
export DATA_GLOB='aug_sp_0*.txt'
export EVAL_W=0.9
export OUT_SCALE=0.70
export WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
export PATIENCE=${PATIENCE:-10}
export SP_BOOST=${SP_BOOST:-1.0}
export MIN_SHARD=${MIN_SHARD:-0}
export SHARD_DECAY=${SHARD_DECAY:-1.0}
export LR=${LR:-3e-3}
export BATCH=${BATCH:-16384}
export ENGINE=${ENGINE:-$ROOT/target/release/sable}

# Full finished shards only (datagen target is 200k). Reject truncated / tiny.
min_lines=${MIN_LINES:-200000}
SP_KEEP=${SP_KEEP:-8}
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
echo "  PATIENCE=$PATIENCE LR=$LR BATCH=$BATCH EVAL_W=$EVAL_W OUT_SCALE=$OUT_SCALE"
if (( n < 800000 )); then
  echo "warning: under 800k SP lines; gate will likely reject" >&2
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
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
