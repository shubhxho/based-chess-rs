#!/usr/bin/env bash
# Canonical SP-first gated train. Never ships a net that loses the arena.
#
#   scripts/ml_cycle.sh              # full SP corpus, 30 epochs, 400-game gate @ +25
#   scripts/ml_cycle.sh 25 400 25    # explicit overrides
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
EPOCHS=${1:-30}
GAMES=${2:-400}
MIN_ELO=${3:-25}

export DATA_DIR=data/selfplay
# Five-digit shards only — skip the tiny aug_sp_10000 pilot.
export DATA_GLOB='aug_sp_[0-9][0-9][0-9][0-9][0-9].txt'
export EVAL_W=0.9
export OUT_SCALE=0.70
export WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
export PATIENCE=${PATIENCE:-8}
export SP_BOOST=${SP_BOOST:-1.0}
# Keep the full corpus; decay older shards so newest 150k teachers dominate.
export MIN_SHARD=${MIN_SHARD:-0}
export SHARD_DECAY=${SHARD_DECAY:-0.95}
export LR=${LR:-2.5e-3}
export ENGINE=${ENGINE:-$ROOT/target/release/sable}

n=$(cat data/selfplay/aug_sp_[0-9][0-9][0-9][0-9][0-9].txt 2>/dev/null | wc -l | tr -d ' ')
echo "ml_cycle: $n self-play lines → train_gate epochs=$EPOCHS games=$GAMES min-elo=$MIN_ELO"
echo "  MIN_SHARD=$MIN_SHARD SHARD_DECAY=$SHARD_DECAY PATIENCE=$PATIENCE LR=$LR"
if (( n < 800000 )); then
  echo "warning: under 800k SP lines; gate will likely reject" >&2
fi

.venv/bin/python train_gate.py --epochs "$EPOCHS" --games "$GAMES" --min-elo "$MIN_ELO"
python3 scripts/daily_page.py
