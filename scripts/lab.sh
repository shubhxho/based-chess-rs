#!/usr/bin/env bash
# One terminal: lab UI + optional background pipelines.
#
#   scripts/lab.sh          # play + daily (default)
#   scripts/lab.sh all      # supervisor: auto-restart datagen + lichess + UI
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
MODE=${1:-}

if [[ "$MODE" == "all" || "$MODE" == "train" ]]; then
  exec bash scripts/lab_supervisor.sh all --web
fi

exec bash scripts/lab_supervisor.sh start --web
