#!/usr/bin/env bash
# Canonical SP-first gated train. Never ships a net that loses the arena.
#
#   scripts/ml_cycle.sh              # 1.5M+ SP, 25 epochs, 400-game gate @ +25
#   scripts/ml_cycle.sh 20 400 25    # explicit overrides
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
EPOCHS=${1:-25}
GAMES=${2:-400}
MIN_ELO=${3:-25}

export DATA_DIR=data/selfplay
export DATA_GLOB='aug_sp_*.txt'
export EVAL_W=0.9
export OUT_SCALE=0.70
export WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
export PATIENCE=${PATIENCE:-6}
export SP_BOOST=${SP_BOOST:-1.0}
# Drop tiny early pilots; weight newer 150k/200k shards more than old 80k runs.
export MIN_SHARD=${MIN_SHARD:-8}
export SHARD_DECAY=${SHARD_DECAY:-0.97}
export ENGINE=${ENGINE:-$ROOT/target/release/sable}

n=$(cat data/selfplay/aug_sp_*.txt 2>/dev/null | wc -l | tr -d ' ')
echo "ml_cycle: $n self-play lines → train_gate epochs=$EPOCHS games=$GAMES min-elo=$MIN_ELO"
echo "  MIN_SHARD=$MIN_SHARD SHARD_DECAY=$SHARD_DECAY PATIENCE=$PATIENCE"
if (( n < 500000 )); then
  echo "warning: under 500k SP lines; gate will likely reject" >&2
fi

.venv/bin/python train_gate.py --epochs "$EPOCHS" --games "$GAMES" --min-elo "$MIN_ELO"
python3 scripts/daily_page.py
