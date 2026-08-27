#!/usr/bin/env bash
# Score-only distillation on Lichess HF shards.
#
# Uses EVAL_W=1 so the neutral result=1 column is ignored — only Stockfish cp
# labels train the net. This path does NOT replace net.bin; run train_gate on
# self-play when you want a shipping candidate.
#
#   scripts/train_lichess.sh              # 15 epochs, full corpus
#   scripts/train_lichess.sh 20           # 20 epochs
#   scripts/train_lichess.sh 20 500000    # cap at 500k positions
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
EPOCHS=${1:-15}
LIMIT=${2:-0}

cargo build --release -q

export DATA_DIR=data/lichess-sf
export DATA_GLOB='aug_hf_*.txt'
export EVAL_W=1
export OUT_SCALE=0.70
export WEIGHTED=1
export SHARD_DECAY=1.0

echo "lichess train: EVAL_W=1 OUT_SCALE=0.70 epochs=$EPOCHS limit=$LIMIT"
echo "  corpus: $DATA_DIR/$DATA_GLOB"
echo "  note: score-only teacher — does not pass shipping gate by itself"

.venv/bin/python train.py "$LIMIT" "$EPOCHS"

echo ""
echo "  pilot weights → net-candidate.bin (shipping net.bin unchanged)"
echo "  to arena vs shipping: DATA_DIR=data/lichess-sf EVAL_W=1 python train_gate.py --epochs $EPOCHS --min-elo 10"
