#!/usr/bin/env bash
# Candidate path toward ~3000 Elo (anchor), via Lichess SF + self-play mix.
#
# Honest framing from the lab notebook:
#   - Shipping trust is still self-play arena ≥ +25 Elo.
#   - Lichess alone has been measuring ~0 vs shipping (−1.7 ± 34).
#   - A stronger *candidate* comes from Stockfish-labelled Lichess volume
#     mixed with finished self-play (outcomes), then gated, then calibrated.
#
#   scripts/push_3000.sh              # sync → mix train → arena (min +10)
#   scripts/push_3000.sh train        # train + arena only (mix already synced)
#   scripts/push_3000.sh arena        # arena existing net-candidate.bin
#   scripts/push_3000.sh calibrate    # Stockfish UCI_Elo anchor (needs stockfish)
#   scripts/push_3000.sh prepare      # grow Lichess corpus (2M batch)
#
# Env knobs: EPOCHS GAMES NODES MIN_ELO LIMIT SP_BOOST EVAL_W OUT_SCALE
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
MODE=${1:-all}

EPOCHS=${EPOCHS:-30}
GAMES=${GAMES:-400}
NODES=${NODES:-20000}
MIN_ELO=${MIN_ELO:-10}
# Cap materialised positions — full HF+SP union can OOM the feature cache.
LIMIT=${LIMIT:-0}
SP_BOOST=${SP_BOOST:-2.0}
EVAL_W=${EVAL_W:-0.9}
OUT_SCALE=${OUT_SCALE:-0.70}
PATIENCE=${PATIENCE:-7}
LR=${LR:-3e-3}
# Newest HF shards linked into mix (see sync_mix.sh). Keep this modest so SP fits.
HF_KEEP=${HF_KEEP:-8}
SP_KEEP=${SP_KEEP:-12}

cargo build --release -q

sync_mix() {
  bash scripts/sync_mix.sh "$SP_KEEP" "$HF_KEEP"
}

train_mix() {
  export DATA_DIR=data/mix
  export DATA_GLOB='aug*.txt'
  export EVAL_W
  export OUT_SCALE
  export SP_BOOST
  export PATIENCE
  export LR
  export WEIGHTED=1
  export SHARD_DECAY=1.0
  export ENGINE=${ENGINE:-$ROOT/target/release/sable}
  export REPORT_DATA_DIR=data/mix
  export REPORT_DATA_GLOB='aug*.txt'

  echo "push_3000 train: SP_KEEP=$SP_KEEP HF_KEEP=$HF_KEEP limit=${LIMIT:-full} epochs=$EPOCHS EVAL_W=$EVAL_W SP_BOOST=$SP_BOOST"
  LOG=/tmp/mix_train.log
  : >"$LOG"
  # train.py treats 0 as "no limit" (falsy). Pass explicitly.
  .venv/bin/python -u train.py "$LIMIT" "$EPOCHS" 2>&1 | tee "$LOG"

  cp -f net.bin net-candidate.bin
  cp -f net.bin net-lichess-pilot.bin
  if [[ -f net.bin.ship ]]; then
    cp -f net.bin.ship net.bin
    cargo build --release -q
    echo "  restored shipping net.bin from net.bin.ship"
  fi

  # Pilot meta for the daily board (reuse Lichess pilot slot as mix candidate).
  .venv/bin/python - <<PY
import datetime as dt, hashlib, json, re
from pathlib import Path
log = Path("$LOG").read_text(errors="replace")
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
  export EVAL_W
  export OUT_SCALE
  export REPORT_DATA_DIR=data/mix
  export REPORT_DATA_GLOB='aug*.txt'
  export PYTHONUNBUFFERED=1
  [[ -f net-candidate.bin ]] || { echo "need net-candidate.bin" >&2; exit 1; }
  [[ -f net.bin.ship ]] || { echo "need net.bin.ship" >&2; exit 1; }
  cp -f net-candidate.bin net.bin
  echo "push_3000 arena: games=$GAMES nodes=$NODES min_elo=$MIN_ELO"
  .venv/bin/python -u train_gate.py --skip-train \
    --epochs "$EPOCHS" --games "$GAMES" --nodes "$NODES" \
    --concurrency 4 --min-elo "$MIN_ELO" || true
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

case "$MODE" in
  prepare)
    echo "growing Lichess corpus (2M batch, depth≥${MIN_DEPTH:-18})"
    exec bash scripts/prepare_lichess.sh "${PREPARE_MAX:-2000000}"
    ;;
  sync)
    sync_mix
    ;;
  train)
    sync_mix
    train_mix
    arena_mix
    ;;
  arena)
    arena_mix
    ;;
  calibrate)
    calibrate
    ;;
  all)
    sync_mix
    # Keep Lichess grow alive if supervisor isn't already holding it.
    if ! pgrep -f "prepare_hf.py data/lichess-sf" >/dev/null 2>&1; then
      echo "starting Lichess prepare in background → /tmp/prepare_resume.log"
      nohup bash scripts/prepare_lichess.sh "${PREPARE_MAX:-2000000}" \
        >>/tmp/prepare_resume.log 2>&1 &
    fi
    train_mix
    arena_mix
    echo ""
    echo "  next: if arena ≥ +10, run scripts/ml_cycle.sh for the +25 ship gate"
    echo "  then: scripts/push_3000.sh calibrate"
    echo "  UI:   http://127.0.0.1:8375/daily"
    ;;
  *)
    echo "usage: $0 [all|train|arena|sync|prepare|calibrate]" >&2
    exit 2
    ;;
esac
