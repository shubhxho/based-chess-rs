#!/usr/bin/env bash
# One terminal: lab UI + optional background pipelines.
#
#   scripts/lab.sh          # play + daily (default)
#   scripts/lab.sh all      # also start datagen daemon + lichess resume in bg
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
MODE=${1:-}

cargo build --release -q 2>/dev/null || cargo build --release -q
python3 scripts/daily_page.py

if [[ "$MODE" == "all" ]]; then
  rm -f data/lichess-sf/.prepare_hf.lock
  nohup bash scripts/datagen_daemon.sh >> /tmp/datagen_daemon.log 2>&1 &
  echo "  datagen daemon pid $! → /tmp/datagen_daemon.log"
  nohup .venv/bin/python prepare_hf.py data/lichess-sf --max-positions 500000 --resume \
    >> /tmp/prepare_resume.log 2>&1 &
  echo "  lichess prepare pid $! → /tmp/prepare_resume.log"
fi

echo ""
echo "  play   http://127.0.0.1:8375"
echo "  daily  http://127.0.0.1:8375/daily  (auto-refreshes)"
echo "  status http://127.0.0.1:8375/api/status"
echo ""
exec python3 web/server.py
