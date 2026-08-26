#!/usr/bin/env bash
# Canonical Lichess HF prepare — always --resume, never re-stream from row 0.
#
# Output rows: FEN | cp_white | 1  (result=1 draw placeholder for EVAL_W=1 training)
#   DATA_DIR=data/lichess-sf DATA_GLOB='aug_hf_*.txt' EVAL_W=1 python train.py 0 20
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
MAX=${1:-500000}
MIN_DEPTH=${MIN_DEPTH:-18}
rm -f data/lichess-sf/.prepare_hf.lock
exec .venv/bin/python prepare_hf.py data/lichess-sf \
  --max-positions "$MAX" \
  --min-depth "$MIN_DEPTH" \
  --resume
