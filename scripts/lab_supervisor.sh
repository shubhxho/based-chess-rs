#!/usr/bin/env bash
# Keep lab services alive: datagen daemon, Lichess prepare, web UI.
#
#   scripts/lab_supervisor.sh          # foreground supervisor + web UI
#   scripts/lab_supervisor.sh bg       # supervisor only (no web UI block)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
MODE=${1:-}

ensure() {
  local name=$1 pattern=$2 cmd=$3 log=$4
  # Mix train needs the RAM; prepare_hf holds multi-GB FEN digests.
  if [[ "$name" == "lichess" && -f "$ROOT/data/lichess-sf/.prepare_paused" ]]; then
    return 0
  fi
  if pgrep -f "$pattern" >/dev/null 2>&1; then
    return 0
  fi
  echo "supervisor: starting $name"
  nohup bash -c "$cmd" >>"$log" 2>&1 &
  echo "  pid $! → $log"
}

stop_dupes() {
  # One datagen daemon max.
  local pids
  pids=$(pgrep -f "datagen_daemon.sh" 2>/dev/null || true)
  local count=0
  for p in $pids; do
    count=$((count + 1))
    if (( count > 1 )); then
      kill "$p" 2>/dev/null || true
    fi
  done
}

cargo build --release -q 2>/dev/null || cargo build --release -q
python3 scripts/daily_page.py

if [[ "$MODE" == "bg" ]]; then
  while true; do
    stop_dupes
    ensure datagen "datagen_daemon.sh" "exec bash scripts/datagen_daemon.sh" /tmp/datagen_daemon.log
    ensure lichess "prepare_hf.py data/lichess-sf" "exec bash scripts/prepare_lichess.sh" /tmp/prepare_resume.log
    python3 scripts/daily_page.py 2>/dev/null || true
    sleep 60
  done
fi

stop_dupes
ensure datagen "datagen_daemon.sh" "exec bash scripts/datagen_daemon.sh" /tmp/datagen_daemon.log
ensure lichess "prepare_hf.py data/lichess-sf" "exec bash scripts/prepare_lichess.sh" /tmp/prepare_resume.log

if ! pgrep -f "web/server.py" >/dev/null 2>&1; then
  echo "supervisor: starting web UI"
else
  echo "supervisor: web UI already running"
fi

echo ""
echo "  play   http://127.0.0.1:8375"
echo "  daily  http://127.0.0.1:8375/daily"
echo "  status http://127.0.0.1:8375/api/status"
echo "  logs   /tmp/datagen_daemon.log · /tmp/prepare_resume.log"
echo ""

# Restart background services every 60s while serving the UI.
( while true; do
    sleep 60
    stop_dupes
    pgrep -f "datagen_daemon.sh" >/dev/null || nohup bash scripts/datagen_daemon.sh >>/tmp/datagen_daemon.log 2>&1 &
    if [[ ! -f "$ROOT/data/lichess-sf/.prepare_paused" ]]; then
      pgrep -f "prepare_hf.py data/lichess-sf" >/dev/null || nohup bash scripts/prepare_lichess.sh >>/tmp/prepare_resume.log 2>&1 &
    fi
    python3 scripts/daily_page.py 2>/dev/null || true
  done
) &

exec python3 web/server.py
