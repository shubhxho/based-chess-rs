#!/usr/bin/env bash
# Canonical SP-first gated train. Never ships a net that loses the arena.
#
# The best measured pass so far was +19.1 Elo on ~1.34M lines with EVAL_W=0.9,
# OUT_SCALE=0.70, default LR 3e-3, no shard decay, full finished 5-digit shards.
# Keep those knobs; only raise epochs/patience a little and drop the 10k pilot.
#
#   scripts/ml_cycle.sh              # finished SP corpus → gate @ +25
#   scripts/ml_cycle.sh 25 400 25
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
EPOCHS=${1:-35}
GAMES=${2:-400}
MIN_ELO=${3:-25}

export DATA_DIR=data/selfplay
# Five-digit shards only. Skip aug_sp_10000 (tiny pilot).
export DATA_GLOB='aug_sp_0*.txt'
export EVAL_W=0.9
export OUT_SCALE=0.70
export WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
export PATIENCE=${PATIENCE:-7}
export SP_BOOST=${SP_BOOST:-1.0}
export MIN_SHARD=${MIN_SHARD:-0}
export SHARD_DECAY=${SHARD_DECAY:-1.0}
export LR=${LR:-3e-3}
export ENGINE=${ENGINE:-$ROOT/target/release/sable}

# Drop any still-writing / tiny shards so train never sees truncated games.
# Cap to newest SP_KEEP finished shards — full 12M+ OOMs a 16GB Mac and the
# best measured pass was on ~1.3M, not every shard ever written.
min_lines=${MIN_LINES:-100000}
SP_KEEP=${SP_KEEP:-16}
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
  echo "  SP_KEEP=$SP_KEEP: using newest ${#ready[@]}/${#all_ready[@]} finished shards" >&2
else
  ready=("${all_ready[@]}")
fi
if (( ${#ready[@]} < 2 )); then
  echo "need ≥2 finished shards (≥$min_lines lines); have ${#ready[@]}" >&2
  exit 1
fi

n=$(cat "${ready[@]}" | wc -l | tr -d ' ')
echo "ml_cycle: $n lines in ${#ready[@]} finished shards → epochs=$EPOCHS games=$GAMES min-elo=$MIN_ELO"
echo "  SHARD_DECAY=$SHARD_DECAY PATIENCE=$PATIENCE LR=$LR"
if (( n < 800000 )); then
  echo "warning: under 800k SP lines; gate will likely reject" >&2
fi

# Point DATA_GLOB at only the ready files via a temp dir of symlinks.
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
# Restore daily page with repo paths (not the temp DATA_DIR).
unset DATA_DIR DATA_GLOB
python3 scripts/daily_page.py
