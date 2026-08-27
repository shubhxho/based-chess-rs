#!/usr/bin/env bash
# Durable Lichess pilot arena (macOS-safe). Usage:
#   scripts/arena_lichess.sh [epochs] [games] [nodes] [min_elo]
# defaults: 20 400 20000 10
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
EPOCHS=${1:-20}
GAMES=${2:-400}
NODES=${3:-20000}
MIN_ELO=${4:-10}

if [[ -f net-lichess-pilot.bin ]]; then
  cp -f net-lichess-pilot.bin net-candidate.bin
elif [[ ! -f net-candidate.bin ]]; then
  echo "need net-lichess-pilot.bin or net-candidate.bin" >&2
  exit 1
fi
[[ -f net.bin.ship ]] || { echo "need net.bin.ship" >&2; exit 1; }
cp -f net.bin.ship net.bin

export DATA_DIR=data/lichess-sf
export DATA_GLOB='aug_hf_*.txt'
export EVAL_W=1
export OUT_SCALE=0.70
export REPORT_DATA_DIR=data/lichess-sf
export REPORT_DATA_GLOB='aug_hf_*.txt'
export PYTHONUNBUFFERED=1

echo "arena_lichess: games=$GAMES nodes=$NODES min_elo=$MIN_ELO" >&2
exec .venv/bin/python -u train_gate.py --skip-train \
  --epochs "$EPOCHS" --games "$GAMES" --nodes "$NODES" \
  --concurrency 4 --min-elo "$MIN_ELO"
