#!/usr/bin/env bash
# Candidate path toward ~3000 Elo (anchor), via Lichess SF + self-play mix.
#
# Honest framing:
#   - Shipping trust is still self-play arena ≥ +25 Elo.
#   - Pure Lichess pilots measured ~0 vs shipping (−1.7 ± 34).
#   - Mix = newest HF (SF garnish) then finished SP (outcomes win on overlap).
#     Prior HF-last mix measured −1.7 twice; SP-last matches shipping's play signal.
#
#   scripts/push_3000.sh              # foreground: sync → train → arena
#   scripts/push_3000.sh bg           # durable background (survives terminal close)
#   scripts/push_3000.sh train        # sync + train + arena (foreground)
#   scripts/push_3000.sh arena        # arena existing net-candidate.bin
#   scripts/push_3000.sh status       # pid + last log lines
#   scripts/push_3000.sh stop         # stop background run only
#   scripts/push_3000.sh calibrate    # Stockfish UCI_Elo anchor
#   scripts/push_3000.sh prepare      # grow Lichess corpus
#   scripts/push_3000.sh sync         # refresh data/mix links only
#
# Env: EPOCHS GAMES NODES MIN_ELO LIMIT SP_BOOST EVAL_W OUT_SCALE HF_KEEP SP_KEEP
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
MODE=${1:-all}

EPOCHS=${EPOCHS:-40}
GAMES=${GAMES:-400}
NODES=${NODES:-20000}
MIN_ELO=${MIN_ELO:-10}
LIMIT=${LIMIT:-0}
# SP-last mix: lean on outcomes (shipping recipe EVAL_W=0.9) with hard SP_BOOST.
SP_BOOST=${SP_BOOST:-3.0}
EVAL_W=${EVAL_W:-0.9}
OUT_SCALE=${OUT_SCALE:-0.70}
PATIENCE=${PATIENCE:-10}
LR=${LR:-2e-3}
# Few newest HF + more finished SP — SF as garnish, play signal as main dish.
HF_KEEP=${HF_KEEP:-2}
SP_KEEP=${SP_KEEP:-10}
BATCH=${BATCH:-8192}

PIDFILE=${PIDFILE:-$ROOT/data/mix/.push_3000.pid}
LOG=${LOG:-/tmp/push_3000.log}
TRAIN_LOG=${TRAIN_LOG:-/tmp/mix_train.log}
PAUSE_PREPARE=${PAUSE_PREPARE:-$ROOT/data/lichess-sf/.prepare_paused}

pause_prepare_for_train() {
  mkdir -p "$(dirname "$PAUSE_PREPARE")"
  touch "$PAUSE_PREPARE"
  if pgrep -f "prepare_hf.py data/lichess-sf" >/dev/null 2>&1; then
    echo "pausing prepare_hf for mix train RAM (was ~2.5GB) → $PAUSE_PREPARE"
    pkill -f "prepare_hf.py data/lichess-sf" 2>/dev/null || true
    sleep 1
    pkill -9 -f "prepare_hf.py data/lichess-sf" 2>/dev/null || true
    sleep 1
  fi
}

resume_prepare_after_train() {
  rm -f "$PAUSE_PREPARE"
  if ! pgrep -f "prepare_hf.py data/lichess-sf" >/dev/null 2>&1; then
    echo "resuming Lichess prepare → /tmp/prepare_resume.log"
    nohup bash scripts/prepare_lichess.sh "${PREPARE_MAX:-2000000}" \
      >>/tmp/prepare_resume.log 2>&1 &
  fi
}

is_running() {
  [[ -f "$PIDFILE" ]] || return 1
  local old
  old=$(cat "$PIDFILE" 2>/dev/null || true)
  [[ -n "$old" ]] && kill -0 "$old" 2>/dev/null
}

claim_lock() {
  mkdir -p "$(dirname "$PIDFILE")"
  if is_running; then
    echo "push_3000 already running (pid $(cat "$PIDFILE")) — see $LOG" >&2
    echo "  status: scripts/push_3000.sh status" >&2
    echo "  stop:   scripts/push_3000.sh stop" >&2
    exit 1
  fi
  rm -f "$PIDFILE"
  echo $$ >"$PIDFILE"
  # On crash/OOM keep prepare paused so a retry still has RAM.
  trap 'rm -f "$PIDFILE"' EXIT
}

show_status() {
  if is_running; then
    echo "running pid $(cat "$PIDFILE")"
  else
    echo "not running"
  fi
  echo "--- $LOG (tail) ---"
  tail -20 "$LOG" 2>/dev/null || echo "(no log)"
  if [[ -f "$TRAIN_LOG" ]]; then
    echo "--- $TRAIN_LOG (last epochs) ---"
    grep -E '^epoch |exporting|quantised|gate:' "$TRAIN_LOG" | tail -15 || true
  fi
}

stop_bg() {
  if ! is_running; then
    rm -f "$PIDFILE"
    echo "not running"
    return 0
  fi
  local old
  old=$(cat "$PIDFILE")
  echo "stopping push_3000 pid $old"
  kill -TERM "$old" 2>/dev/null || true
  sleep 2
  kill -KILL "$old" 2>/dev/null || true
  pkill -f "train.py ${LIMIT} ${EPOCHS}" 2>/dev/null || true
  rm -f "$PIDFILE"
  resume_prepare_after_train
  echo "stopped"
}

sync_mix() {
  bash scripts/sync_mix.sh "$SP_KEEP" "$HF_KEEP"
}

train_mix() {
  export DATA_DIR=data/mix
  export DATA_GLOB='aug*.txt'
  export EVAL_W OUT_SCALE SP_BOOST PATIENCE LR
  export BATCH
  export WEIGHTED=1
  export SHARD_DECAY=1.0
  export ENGINE=${ENGINE:-$ROOT/target/release/sable}
  export REPORT_DATA_DIR=data/mix
  export REPORT_DATA_GLOB='aug*.txt'

  echo "push_3000 train: SP=$SP_KEEP HF=$HF_KEEP limit=${LIMIT:-full} epochs=$EPOCHS BATCH=$BATCH EVAL_W=$EVAL_W SP_BOOST=$SP_BOOST"
  : >"$TRAIN_LOG"
  # Immune to SIGHUP if the wrapper shell is reaped; log to file only (no tee).
  trap '' HUP
  nohup .venv/bin/python -u train.py "$LIMIT" "$EPOCHS" >"$TRAIN_LOG" 2>&1 &
  local train_pid=$!
  echo "  train.py pid $train_pid → $TRAIN_LOG"
  # Stream epochs into the durable log without a pipe that can SIGPIPE us.
  (
    trap '' HUP
    tail -n +1 -F "$TRAIN_LOG" 2>/dev/null | while IFS= read -r line; do
      echo "$line" >>"$LOG"
      case "$line" in
        exporting*|quantised*|wrote\ net.bin*) break ;;
      esac
    done
  ) &
  local tail_pid=$!
  wait "$train_pid"
  local rc=$?
  kill "$tail_pid" 2>/dev/null || true
  wait "$tail_pid" 2>/dev/null || true
  if (( rc != 0 )); then
    echo "train.py exited $rc — see $TRAIN_LOG" >&2
    exit "$rc"
  fi

  cp -f net.bin net-candidate.bin
  cp -f net.bin net-lichess-pilot.bin
  if [[ -f net.bin.ship ]]; then
    cp -f net.bin.ship net.bin
    cargo build --release -q
    echo "  restored shipping net.bin from net.bin.ship"
  fi

  .venv/bin/python - <<PY
import datetime as dt, hashlib, json, re
from pathlib import Path
log = Path("$TRAIN_LOG").read_text(errors="replace")
val = r = mae = epochs_done = None
for line in log.splitlines():
    m = re.match(r"epoch\s+(\d+)/(\d+)\s+.*val\s+([0-9.]+)", line)
    if m:
        epochs_done = int(m.group(1)); val = float(m.group(3))
    if "quantised net vs teacher:" in line:
        rm = re.search(r"r=([0-9.]+)", line)
        mm = re.search(r"mae=([0-9.]+)cp", line)
        if rm: r = float(rm.group(1))
        if mm: mae = float(mm.group(1))
    m2 = re.search(r"exporting best-val checkpoint \(val ([0-9.]+)\)", line)
    if m2: val = float(m2.group(1))
pilot = Path("net-lichess-pilot.bin")
meta = {
    "when": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    "path": "mix",
    "epochs_requested": int("$EPOCHS"),
    "epochs_done": epochs_done,
    "limit": int("$LIMIT"),
    "val": val, "r": r, "mae_cp": mae,
    "eval_w": "$EVAL_W", "out_scale": "$OUT_SCALE", "sp_boost": "$SP_BOOST",
    "bytes": pilot.stat().st_size if pilot.is_file() else None,
    "sha16": hashlib.sha256(pilot.read_bytes()).hexdigest()[:16] if pilot.is_file() else None,
    "files": ["net-lichess-pilot.bin", "net-candidate.bin"],
    "note": "Lichess SF + SP mix toward 3000-Elo candidate",
}
Path("web/pilot_last.json").write_text(json.dumps(meta, indent=2) + "\n")
print(f"  wrote web/pilot_last.json val={val} r={r} mae={mae}")
PY
}

arena_mix() {
  export DATA_DIR=data/mix
  export DATA_GLOB='aug*.txt'
  export EVAL_W OUT_SCALE
  export REPORT_DATA_DIR=data/mix
  export REPORT_DATA_GLOB='aug*.txt'
  export PYTHONUNBUFFERED=1
  [[ -f net-candidate.bin ]] || { echo "need net-candidate.bin" >&2; exit 1; }
  [[ -f net.bin.ship ]] || { echo "need net.bin.ship" >&2; exit 1; }
  cp -f net-candidate.bin net.bin
  echo "push_3000 arena: games=$GAMES nodes=$NODES min_elo=$MIN_ELO"
  set +e
  .venv/bin/python -u train_gate.py --skip-train \
    --epochs "$EPOCHS" --games "$GAMES" --nodes "$NODES" \
    --concurrency 4 --min-elo "$MIN_ELO"
  set -e
  python3 scripts/daily_page.py
  python3 scripts/blog_page.py
}

calibrate() {
  cargo build --release -q
  echo "push_3000 calibrate vs Stockfish UCI_Elo (movetime 100, 40 games/level)"
  .venv/bin/python tests/calibrate.py \
    "$ROOT/target/release/sable" "movetime 100" 40 \
    2600 2700 2800 2900 3000 3100 | tee /tmp/calibrate_3000.log
}

run_pipeline() {
  claim_lock
  cargo build --release -q
  sync_mix
  pause_prepare_for_train
  train_mix
  arena_mix
  resume_prepare_after_train
  echo ""
  echo "  next: if arena ≥ +10 → scripts/ml_cycle.sh 35 400 25"
  echo "  then: scripts/push_3000.sh calibrate"
  echo "  UI:   http://127.0.0.1:8375/daily"
}

case "$MODE" in
  status)
    show_status
    ;;
  stop)
    stop_bg
    ;;
  bg)
    if is_running; then
      echo "already running pid $(cat "$PIDFILE") — $LOG" >&2
      exit 1
    fi
    echo "starting background push_3000 → $LOG"
    # New session so Cursor/agent shell teardown cannot SIGTERM the train.
    .venv/bin/python - <<PY
import subprocess, sys
from pathlib import Path
root = Path("$ROOT")
log = Path("$LOG")
script = root / "scripts" / "push_3000.sh"
log.parent.mkdir(parents=True, exist_ok=True)
out = open(log, "a", buffering=1)
subprocess.Popen(
    ["bash", str(script), "all"],
    cwd=str(root),
    stdin=subprocess.DEVNULL,
    stdout=out,
    stderr=subprocess.STDOUT,
    start_new_session=True,
    close_fds=True,
)
print(f"  spawned session → {log}", flush=True)
PY
    sleep 3
    if is_running; then
      echo "  running pid $(cat "$PIDFILE")"
    else
      echo "  launched — check $LOG / scripts/push_3000.sh status"
    fi
    echo "  status: scripts/push_3000.sh status"
    ;;
  prepare)
    echo "growing Lichess corpus (2M batch, depth≥${MIN_DEPTH:-18})"
    exec bash scripts/prepare_lichess.sh "${PREPARE_MAX:-2000000}"
    ;;
  sync)
    sync_mix
    ;;
  train)
    run_pipeline train
    ;;
  arena)
    claim_lock
    cargo build --release -q
    arena_mix
    ;;
  calibrate)
    calibrate
    ;;
  all)
    run_pipeline all
    ;;
  *)
    echo "usage: $0 [all|bg|train|arena|sync|prepare|calibrate|status|stop]" >&2
    exit 2
    ;;
esac
