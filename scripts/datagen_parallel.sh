#!/usr/bin/env bash
# Parallel self-play shard generation for distillation.
#
# usage: scripts/datagen_parallel.sh [positions] [nodes] [n_shards] [start]
#   start=auto → fill lowest incomplete shards, then append new indices
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
  local idx_csv=""
  if ((${#active_indices[@]})); then
    idx_csv=$(IFS=,; echo "${active_indices[*]}")
  fi
  python3 - <<PY
import json, datetime as dt
from pathlib import Path
p = Path("$STATUS")
p.parent.mkdir(parents=True, exist_ok=True)
raw = "$pid_csv"
pids = [int(x) for x in raw.split(",") if x.strip()]
raw_idx = "$idx_csv"
indices = [int(x) for x in raw_idx.split(",") if x.strip()]
shard_progress = []
pos = $POS
out = Path("$OUT")
for i in indices:
    f = out / f"aug_sp_{i:05d}.txt"
    lines = sum(1 for _ in f.open("rb")) if f.exists() else 0
    shard_progress.append({
        "index": i,
        "name": f.name,
        "lines": lines,
        "target": pos,
        "pct": min(100, int(100 * lines / pos)) if pos else 0,
        "done": lines >= pos,
    })
total = sum(s["lines"] for s in shard_progress)
want = pos * len(indices)
p.write_text(json.dumps({
    "phase": "$phase",
    "when": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    "positions_per_shard": pos,
    "nodes": $NODES,
    "n_shards": $N,
    "start_index": $START,
    "active_indices": indices,
    "active_shards": shard_progress,
    "wave_lines": total,
    "wave_target": want,
    "wave_pct": min(100, int(100 * total / want)) if want else 100,
    "pids": pids,
}, indent=2) + "\\n")
PY
}

pick_auto_indices() {
  python3 - <<PY
import json, re
from pathlib import Path

pos = $POS
n = $N
out = Path("$OUT")
states = {}
for f in sorted(out.glob("aug_sp_0*.txt")):
    m = re.fullmatch(r"aug_sp_(0\d{4})", f.stem)
    if not m:
        continue
    idx = int(m.group(1))
    lines = sum(1 for _ in f.open("rb"))
    states[idx] = lines

todo = []
for idx in sorted(states):
    if 0 < states[idx] < pos:
        todo.append(idx)
        if len(todo) >= n:
            break

max_idx = max(states) if states else -1
next_idx = max_idx + 1
while len(todo) < n:
    if next_idx not in states or states[next_idx] < pos:
        todo.append(next_idx)
    next_idx += 1

print(json.dumps(todo[:n]))
PY
}

mkdir -p "$OUT"
active_indices=()

if [[ "$START" == "auto" ]]; then
  active_indices=()
  while IFS= read -r line; do
    [[ -n "$line" ]] && active_indices+=("$line")
  done < <(pick_auto_indices | python3 -c "import json,sys; [print(x) for x in json.load(sys.stdin)]")
  if ((${#active_indices[@]} == 0)); then
    START=0
    echo "datagen_parallel: nothing to do (all shards complete)" >&2
    write_status idle
    exit 0
  fi
  START=${active_indices[0]}
  echo "auto: wave indices ${active_indices[*]}" >&2
else
  for ((k=0; k<N; k++)); do
    active_indices+=($((START + k)))
  done
fi

cargo build --release --manifest-path "$ROOT/Cargo.toml" -q

echo "datagen_parallel: ${#active_indices[@]} shards x $POS positions @ $NODES nodes (from index $START)"
pids=()
for i in "${active_indices[@]}"; do
  seed=$((i * 7919))
  shard=$(printf '%s/aug_sp_%05d.txt' "$OUT" "$i")
  lines=$(shard_lines "$shard")
  if (( lines >= POS )); then
    echo "  skip $shard ($lines/$POS lines already complete)"
    continue
  fi
  if pgrep -f "aug_sp_$(printf '%05d' "$i")" >/dev/null 2>&1; then
    echo "  skip $shard (already running)" >&2
    continue
  fi
  err=$(printf '%s_%05d.err' "$LOG" "$i")
  printf 'datagen %s %s %s\nquit\n' "$POS" "$NODES" "$seed" \
    | "$BIN" >"$shard" 2>"$err" &
  pid=$!
  pids+=("$pid")
  echo "  pid $pid -> $shard ($lines/$POS lines, seed $seed)"
done

if ((${#pids[@]} == 0)); then
  echo "datagen_parallel: no workers started (shards complete or busy)" >&2
  write_status idle
  exit 0
fi

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
    write_status running
    total=0
    want=0
    for i in "${active_indices[@]}"; do
      shard=$(printf '%s/aug_sp_%05d.txt' "$OUT" "$i")
      lines=$(shard_lines "$shard")
      total=$((total + lines))
      want=$((want + POS))
    done
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
