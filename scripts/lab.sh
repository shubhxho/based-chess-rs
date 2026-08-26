#!/usr/bin/env bash
# One terminal: lab UI + optional background pipelines.
#
#   scripts/lab.sh          # play + daily (default)
#   scripts/lab.sh all      # supervisor: auto-restart datagen + lichess + UI
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
MODE=${1:-}

if [[ "$MODE" == "all" ]]; then
  exec bash scripts/lab_supervisor.sh
fi

cargo build --release -q 2>/dev/null || cargo build --release -q
python3 scripts/daily_page.py

echo ""
echo "  play   http://127.0.0.1:8375"
echo "  daily  http://127.0.0.1:8375/daily  (auto-refreshes)"
echo "  status http://127.0.0.1:8375/api/status"
echo ""
exec python3 web/server.py
