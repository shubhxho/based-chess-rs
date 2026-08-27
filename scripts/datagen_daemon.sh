#!/usr/bin/env bash
# Continuous datagen waves until interrupted. Auto-restarts after failures.
#
#   scripts/datagen_daemon.sh
#   scripts/datagen_daemon.sh stop
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PIDFILE="$ROOT/data/selfplay/.datagen_daemon.pid"
POS=${POS:-200000}
NODES=${NODES:-12000}
N=${N:-4}

stop_daemon() {
  if [[ -f "$PIDFILE" ]]; then
    old=$(cat "$PIDFILE" 2>/dev/null || true)
    if [[ -n "$old" ]] && kill -0 "$old" 2>/dev/null; then
      echo "stopping datagen daemon pid $old"
      kill "$old" 2>/dev/null || true
      sleep 1
      kill -9 "$old" 2>/dev/null || true
    fi
    rm -f "$PIDFILE"
  fi
}

if [[ "${1:-}" == "stop" ]]; then
  stop_daemon
  exit 0
fi

mkdir -p "$ROOT/data/selfplay"
if [[ -f "$PIDFILE" ]]; then
  old=$(cat "$PIDFILE" 2>/dev/null || true)
  if [[ -n "$old" ]] && kill -0 "$old" 2>/dev/null; then
    echo "datagen_daemon already running (pid $old)" >&2
    exit 1
  fi
  rm -f "$PIDFILE"
fi

echo $$ >"$PIDFILE"
trap 'rm -f "$PIDFILE"; pkill -TERM -P $$ 2>/dev/null || true' EXIT INT TERM

echo "datagen_daemon: ${POS} @ ${NODES}n, ${N} shards, auto index (Ctrl-C to stop)"
wave=0
backoff=10
PAUSE="$ROOT/data/selfplay/.datagen_paused"
while true; do
  # Lab gate / stress can block datagen via this flag without killing the daemon.
  if [[ -f "$PAUSE" ]]; then
    echo "datagen paused ($PAUSE); sleeping ${backoff}s" >&2
    sleep "$backoff"
    continue
  fi
  wave=$((wave + 1))
  echo "=== datagen wave $wave $(date -Iseconds 2>/dev/null || date) ===" >&2
  if bash scripts/datagen_parallel.sh "$POS" "$NODES" "$N" auto; then
    backoff=10
  else
    echo "wave $wave failed; retry in ${backoff}s" >&2
    sleep "$backoff"
    backoff=$((backoff * 2))
    (( backoff > 300 )) && backoff=300
    continue
  fi
  sleep "$backoff"
done
