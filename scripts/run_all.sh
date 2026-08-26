#!/usr/bin/env bash
# One command: build, refresh daily, start full lab stack, optional gate.
#
#   scripts/run_all.sh           # supervisor + UI (default)
#   scripts/run_all.sh bg        # background supervisor only
#   scripts/run_all.sh refresh   # python3 daily snapshot only
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
MODE=${1:-}

refresh() {
  python3 scripts/daily_page.py
  python3 -m py_compile prepare_hf.py scripts/lab_status.py scripts/daily_page.py web/server.py
}

case "$MODE" in
  refresh)
    refresh
    echo "refreshed web/daily.html"
    exit 0
    ;;
  bg)
    refresh
    exec bash scripts/lab_supervisor.sh bg
    ;;
  *)
    refresh
    exec bash scripts/lab_supervisor.sh
    ;;
esac
