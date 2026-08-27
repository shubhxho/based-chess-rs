#!/usr/bin/env bash
# The single owner of the Sable lab processes.
#
#   scripts/lab_supervisor.sh start [--web]  # detached controller
#   scripts/lab_supervisor.sh run [--web]    # controller in this terminal
#   scripts/lab_supervisor.sh train          # queue one gated SP attempt
#   scripts/lab_supervisor.sh all [--web]    # start + queue one attempt
#   scripts/lab_supervisor.sh status | stop
#
# The controller owns datagen, Hugging Face preparation, page refreshes, and
# the optional local web server.  Training is queued here so it never races an
# independent trainer or the memory-heavy corpus preparation job.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
STATE="$ROOT/data/lab"
LOCK="$STATE/supervisor.lock"
PIDFILE="$STATE/supervisor.pid"
REQUEST="$STATE/train.request"
STATUS="$STATE/status"
LOG="${LAB_SUPERVISOR_LOG:-/tmp/sable_lab_supervisor.log}"
INTERVAL="${SUPERVISOR_INTERVAL:-60}"
WEB=0
DATAGEN_PID=""
PREP_PID=""
WEB_PID=""
GATE_PID=""

mkdir -p "$STATE"

alive() {
  [[ -n "${1:-}" ]] && kill -0 "$1" 2>/dev/null
}

read_pid() {
  [[ -f "$PIDFILE" ]] || return 1
  local pid
  pid=$(cat "$PIDFILE" 2>/dev/null || true)
  [[ "$pid" =~ ^[0-9]+$ ]] && printf '%s\n' "$pid"
}

write_status() {
  local phase=$1
  {
    printf 'phase=%s\n' "$phase"
    printf 'pid=%s\n' "$$"
    printf 'datagen_pid=%s\n' "$DATAGEN_PID"
    printf 'prepare_pid=%s\n' "$PREP_PID"
    printf 'web_pid=%s\n' "$WEB_PID"
    printf 'gate_pid=%s\n' "$GATE_PID"
    printf 'updated=%s\n' "$(date -Iseconds 2>/dev/null || date)"
  } >"$STATUS.tmp"
  mv -f "$STATUS.tmp" "$STATUS"
}

release() {
  rm -f "$PIDFILE" "$STATUS"
  rmdir "$LOCK" 2>/dev/null || true
}

claim() {
  if mkdir "$LOCK" 2>/dev/null; then
    printf '%s\n' "$$" >"$PIDFILE"
    return
  fi
  local pid=""
  pid=$(read_pid 2>/dev/null || true)
  if alive "$pid"; then
    echo "lab supervisor already running (pid $pid)" >&2
    exit 1
  fi
  # A terminated controller can leave its directory behind.  Reclaim only a
  # stale lock; mkdir is still the atomic ownership decision.
  rm -f "$PIDFILE" "$STATUS"
  rmdir "$LOCK" 2>/dev/null || {
    echo "cannot reclaim supervisor lock: $LOCK" >&2
    exit 1
  }
  mkdir "$LOCK"
  printf '%s\n' "$$" >"$PIDFILE"
}

refresh() {
  python3 scripts/daily_page.py >>"$LOG" 2>&1 || true
  python3 scripts/blog_page.py >>"$LOG" 2>&1 || true
}

adopt_worker() {
  # Worker-level PID locks survive a supervisor restart.  Adopt them instead
  # of spawning a duplicate process that immediately fails its own lock.
  local file=$1
  [[ -f "$file" ]] || return 1
  local pid
  pid=$(sed -n 's/[^0-9].*//p' "$file" 2>/dev/null | head -n 1)
  alive "$pid" || return 1
  printf '%s\n' "$pid"
}

start_datagen() {
  [[ -f "$ROOT/data/selfplay/.datagen_paused" ]] && return
  alive "$DATAGEN_PID" && return
  local adopted=""
  adopted=$(adopt_worker "$ROOT/data/selfplay/.datagen_daemon.pid" || true)
  if alive "$adopted"; then
    DATAGEN_PID=$adopted
    echo "supervisor: adopted self-play datagen pid $DATAGEN_PID" >>"$LOG"
    return
  fi
  echo "supervisor: starting self-play datagen" >>"$LOG"
  NODES="${DATAGEN_NODES:-12000}" POS="${DATAGEN_POS:-200000}" N="${DATAGEN_N:-4}" \
    bash scripts/datagen_daemon.sh >>/tmp/datagen_daemon.log 2>&1 &
  DATAGEN_PID=$!
}

start_prepare() {
  [[ -f "$ROOT/data/lichess-sf/.prepare_paused" ]] && return
  alive "$PREP_PID" && return
  local adopted=""
  adopted=$(adopt_worker "$ROOT/data/lichess-sf/.prepare_hf.lock" || true)
  if alive "$adopted"; then
    PREP_PID=$adopted
    echo "supervisor: adopted Lichess preparation pid $PREP_PID" >>"$LOG"
    return
  fi
  echo "supervisor: starting Lichess preparation" >>"$LOG"
  bash scripts/prepare_lichess.sh >>/tmp/prepare_resume.log 2>&1 &
  PREP_PID=$!
}

start_web() {
  (( WEB )) || return
  alive "$WEB_PID" && return
  echo "supervisor: starting web UI" | tee -a "$LOG"
  python3 web/server.py >>/tmp/sable_web.log 2>&1 &
  WEB_PID=$!
}

stop_prepare_for_train() {
  mkdir -p "$ROOT/data/lichess-sf"
  touch "$ROOT/data/lichess-sf/.prepare_paused"
  if alive "$PREP_PID"; then
    echo "supervisor: pausing Lichess preparation for training" | tee -a "$LOG"
    kill -TERM "$PREP_PID" 2>/dev/null || true
    wait "$PREP_PID" 2>/dev/null || true
  fi
  PREP_PID=""
}

run_training() {
  rm -f "$REQUEST"
  # A different training path owns net.bin too.  Never run two of them.
  if pgrep -f 'train_gate.py|train.py|push_3000.sh' >/dev/null 2>&1; then
    echo "supervisor: training request deferred; another training job is active" | tee -a "$LOG"
    touch "$REQUEST"
    return
  fi
  stop_prepare_for_train
  write_status training
  echo "supervisor: starting gated self-play attempt → /tmp/sable_gate.log" >>"$LOG"
  bash scripts/ml_cycle.sh "${GATE_EPOCHS:-50}" "${GATE_GAMES:-400}" "${GATE_MIN_ELO:-25}" \
      >>/tmp/sable_gate.log 2>&1 &
  GATE_PID=$!
  write_status training
  # train_gate.py continuously writes gate_last.json during the arena, while
  # the owned web server refreshes the page.  Waiting here keeps net.bin under
  # this supervisor's exclusive gate window.
  if wait "$GATE_PID"; then
    echo "supervisor: gated attempt completed" >>"$LOG"
  else
    echo "supervisor: gated attempt rejected or failed; see /tmp/sable_gate.log" >>"$LOG"
  fi
  GATE_PID=""
  rm -f "$ROOT/data/lichess-sf/.prepare_paused"
  refresh
}

shutdown() {
  trap - EXIT INT TERM
  [[ -n "$DATAGEN_PID" ]] && kill -TERM "$DATAGEN_PID" 2>/dev/null || true
  [[ -n "$PREP_PID" ]] && kill -TERM "$PREP_PID" 2>/dev/null || true
  [[ -n "$WEB_PID" ]] && kill -TERM "$WEB_PID" 2>/dev/null || true
  release
}

run() {
  claim
  trap shutdown EXIT INT TERM
  cargo build --release -q 2>/dev/null || cargo build --release -q
  refresh
  while true; do
    start_datagen
    start_prepare
    start_web
    if [[ -f "$REQUEST" ]]; then
      run_training
    fi
    write_status running
    refresh
    sleep "$INTERVAL"
  done
}

start() {
  local args=(run)
  (( WEB )) && args+=(--web)
  local existing=""
  existing=$(read_pid 2>/dev/null || true)
  if alive "$existing"; then
    echo "lab supervisor already running (pid $existing)"
    return
  fi
  .venv/bin/python - "$ROOT" "$LOG" "${args[@]}" <<'PY'
import subprocess, sys
from pathlib import Path
root, log, *args = sys.argv[1:]
with Path(log).open("ab", buffering=0) as out:
    proc = subprocess.Popen(
        ["bash", str(Path(root) / "scripts" / "lab_supervisor.sh"), *args],
        cwd=root, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
        start_new_session=True, close_fds=True,
    )
print(f"started lab supervisor pid {proc.pid} → {log}")
PY
}

status() {
  local pid=""
  pid=$(read_pid 2>/dev/null || true)
  if alive "$pid"; then
    echo "supervisor: running (pid $pid)"
  else
    echo "supervisor: stopped"
  fi
  [[ -f "$STATUS" ]] && cat "$STATUS"
  echo "logs: $LOG · /tmp/datagen_daemon.log · /tmp/prepare_resume.log · /tmp/sable_gate.log"
}

stop() {
  local pid=""
  pid=$(read_pid 2>/dev/null || true)
  if ! alive "$pid"; then
    rm -f "$PIDFILE" "$STATUS"
    rmdir "$LOCK" 2>/dev/null || true
    echo "supervisor: already stopped"
    return
  fi
  echo "stopping lab supervisor pid $pid"
  # Detached starts create a separate process group, so this also reaps its
  # direct workers.  A foreground run receives its own normal signal instead.
  kill -TERM "$pid" 2>/dev/null || true
}

MODE=${1:-run}
shift || true
while (( $# )); do
  case "$1" in
    --web) WEB=1 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

case "$MODE" in
  run) run ;;
  start) start ;;
  train)
    touch "$REQUEST"
    echo "queued one gated self-play attempt"
    ;;
  all)
    start
    touch "$REQUEST"
    echo "queued one gated self-play attempt"
    ;;
  status) status ;;
  stop) stop ;;
  *)
    echo "usage: $0 {start|run|train|all|status|stop} [--web]" >&2
    exit 2
    ;;
esac
