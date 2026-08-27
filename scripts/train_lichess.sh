#!/usr/bin/env bash
# Score-only distillation on Lichess HF shards.
#
# Uses EVAL_W=1 so the neutral result=1 column is ignored — only Stockfish cp
# labels train the net. This path does NOT replace net.bin; run train_gate on
# self-play when you want a shipping candidate.
#
#   scripts/train_lichess.sh              # 15 epochs, full corpus
#   scripts/train_lichess.sh 20           # 20 epochs
#   scripts/train_lichess.sh 20 500000    # cap at 500k positions
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
EPOCHS=${1:-15}
LIMIT=${2:-0}

cargo build --release -q

export DATA_DIR=data/lichess-sf
export DATA_GLOB='aug_hf_*.txt'
export EVAL_W=1
export OUT_SCALE=0.70
export WEIGHTED=1
export SHARD_DECAY=1.0

echo "lichess train: EVAL_W=1 OUT_SCALE=0.70 epochs=$EPOCHS limit=$LIMIT"
echo "  corpus: $DATA_DIR/$DATA_GLOB"
echo "  note: score-only teacher — does not pass shipping gate by itself"

LOG=$(mktemp -t lichess_train.XXXXXX)
trap 'rm -f "$LOG"' EXIT
.venv/bin/python -u train.py "$LIMIT" "$EPOCHS" 2>&1 | tee "$LOG"

# train.py writes net.bin — park the pilot as candidate and restore shipping.
cp -f net.bin net-lichess-pilot.bin
cp -f net.bin net-candidate.bin
if [[ -f net.bin.ship ]]; then
  cp -f net.bin.ship net.bin
  cargo build --release -q
  echo "  restored shipping net.bin from net.bin.ship"
fi

.venv/bin/python - <<PY
import datetime as dt, hashlib, json, re
from pathlib import Path
root = Path(".")
log = Path("$LOG").read_text(errors="replace")
val = None
r = mae = None
epochs_done = None
for line in log.splitlines():
    m = re.match(r"epoch\s+(\d+)/(\d+)\s+.*val\s+([0-9.]+)", line)
    if m:
        epochs_done = int(m.group(1))
        val = float(m.group(3))
    if "quantised net vs teacher:" in line:
        rm = re.search(r"r=([0-9.]+)", line)
        mm = re.search(r"mae=([0-9.]+)cp", line)
        if rm:
            r = float(rm.group(1))
        if mm:
            mae = float(mm.group(1))
    m2 = re.search(r"exporting best-val checkpoint \(val ([0-9.]+)\)", line)
    if m2:
        val = float(m2.group(1))
pilot = root / "net-lichess-pilot.bin"
h = hashlib.sha256(pilot.read_bytes()).hexdigest()[:16] if pilot.is_file() else None
meta = {
    "when": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    "path": "lichess",
    "epochs_requested": int("$EPOCHS"),
    "epochs_done": epochs_done,
    "limit": int("$LIMIT"),
    "val": val,
    "r": r,
    "mae_cp": mae,
    "eval_w": "1",
    "out_scale": "0.70",
    "bytes": pilot.stat().st_size if pilot.is_file() else None,
    "sha16": h,
    "files": ["net-lichess-pilot.bin", "net-candidate.bin"],
}
out = root / "web" / "pilot_last.json"
out.write_text(json.dumps(meta, indent=2) + "\n")
print(f"  wrote {out}  val={val} r={r} mae={mae}")
PY

echo ""
echo "  pilot → net-candidate.bin + net-lichess-pilot.bin"
echo "  arena (no retrain):"
echo "    scripts/arena_lichess.sh $EPOCHS 400 20000 10"
