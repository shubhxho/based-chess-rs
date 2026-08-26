#!/usr/bin/env bash
# Canonical SP-first gated train. Never ships a net that loses the arena.
#
#   scripts/ml_cycle.sh              # full self-play corpus, 20 epochs, 400-game gate
#   scripts/ml_cycle.sh 12 200       # shorter pilot
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
EPOCHS=${1:-20}
GAMES=${2:-400}
MIN_ELO=${3:-25}

export DATA_DIR=data/selfplay
export DATA_GLOB='aug_sp_*.txt'
export EVAL_W=0.9
export OUT_SCALE=0.70
export WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
export PATIENCE=${PATIENCE:-5}
export SP_BOOST=${SP_BOOST:-1.0}
export ENGINE=${ENGINE:-$ROOT/target/release/sable}

n=$(cat data/selfplay/aug_sp_*.txt 2>/dev/null | wc -l | tr -d ' ')
echo "ml_cycle: $n self-play lines → train_gate epochs=$EPOCHS games=$GAMES min-elo=$MIN_ELO"
if (( n < 200000 )); then
  echo "warning: under 200k SP lines; gate will likely reject" >&2
fi

.venv/bin/python train_gate.py --epochs "$EPOCHS" --games "$GAMES" --min-elo "$MIN_ELO"
