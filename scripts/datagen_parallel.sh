#!/usr/bin/env bash
# Parallel self-play shard generation for distillation.
#
# usage: scripts/datagen_parallel.sh [positions] [nodes] [n_shards] [start]
#   start=auto → resume first incomplete 5-digit shard, else next index
# defaults: 200000  8000  4  auto
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
POS=${1:-200000}
NODES=${2:-8000}
N=${3:-4}
START=${4:-auto}
OUT="$ROOT/data/selfplay"
BIN="$ROOT/target/release/sable"
LOG=/tmp/sable-datagen
STATUS="$OUT/datagen_status.json"

shard_lines() {
  local f=$1
  [[ -f "$f" ]] || { echo 0; return; }
  wc -l <"$f" | tr -d ' '
}

write_status() {
  local phase=$1
  local pid_csv=""
  if ((${#pids[@]})); then
    pid_csv=$(IFS=,; echo "${pids[*]}")
  fi
  python3 - <<PY
import json, datetime as dt
from pathlib import Path
p = Path("$STATUS")
p.parent.mkdir(parents=True, exist_ok=True)
raw = "$pid_csv"
pids = [int(x) for x in raw.split(",") if x.strip()]
p.write_text(json.dumps({
    "phase": "$phase",
    "when": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    "positions_per_shard": $POS,
    "nodes": $NODES,
    "n_shards": $N,
    "start_index": $START,
    "pids": pids,
}, indent=2) + "\\n")
PY
}

if [[ "$START" == "auto" ]]; then
  START=0
  incomplete=""
  for f in "$OUT"/aug_sp_*.txt; do
    [[ -f "$f" ]] || continue
    base=$(basename "$f" .txt)
    if [[ "$base" =~ ^aug_sp_(0[0-9]{4})$ ]]; then
      idx=$((10#${BASH_REMATCH[1]}))
      lines=$(shard_lines "$f")
      if (( lines > 0 && lines < POS )); then
        incomplete=$idx
      fi
      if (( idx + 1 > START )); then
        START=$((idx + 1))
      fi
    fi
  done
  if [[ -n "$incomplete" ]]; then
    # Only resume a partial shard if nothing is already writing it.
    if ! pgrep -f "aug_sp_$(printf '%05d' "$incomplete")" >/dev/null 2>&1; then
      START=$incomplete
      echo "auto: resume incomplete aug_sp_$(printf '%05d' "$incomplete")" >&2
    fi
  fi
fi

mkdir -p "$OUT"
cargo build --release --manifest-path "$ROOT/Cargo.toml" -q

echo "datagen_parallel: $N shards x $POS positions @ $NODES nodes (start index $START)"
pids=()
for ((k=0; k<N; k++)); do
  i=$((START + k))
  seed=$((i * 7919))
  shard=$(printf '%s/aug_sp_%05d.txt' "$OUT" "$i")
  err=$(printf '%s_%05d.err' "$LOG" "$i")
  printf 'datagen %s %s %s\nquit\n' "$POS" "$NODES" "$seed" \
    | "$BIN" >"$shard" 2>"$err" &
  pid=$!
  pids+=("$pid")
  echo "  pid $pid -> $shard (seed $seed)"
done

write_status running

fail=0
while (( ${#pids[@]} )); do
  still=()
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      still+=("$pid")
    else
      wait "$pid" || fail=1
    fi
  done
  pids=("${still[@]}")
  if (( ${#pids[@]} )); then
    total=0
    for ((k=0; k<N; k++)); do
      i=$((START + k))
      shard=$(printf '%s/aug_sp_%05d.txt' "$OUT" "$i")
      total=$((total + $(shard_lines "$shard")))
    done
    want=$((POS * N))
    echo "  progress ${total}/${want} lines ($(date +%H:%M:%S))" >&2
    sleep 30
  fi
done

wc -l "$OUT"/aug_sp_0*.txt 2>/dev/null | tail -5
write_status done
python3 "$ROOT/scripts/daily_page.py" 2>/dev/null || true
if (( fail )); then
  echo "datagen_parallel: worker failed" >&2
  exit 1
fi
echo "datagen_parallel done"
