#!/usr/bin/env bash
# Parallel self-play shard generation for distillation.
#
# usage: scripts/datagen_parallel.sh [positions] [nodes] [n_shards] [start]
#   start=auto → fill highest-progress partial shards first, then new indices
# defaults: 200000  10000  4  auto
#
# Workers write to *.txt.tmp and atomically replace the shard only on success,
# so a killed run never truncates an on-disk partial shard.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
POS=${1:-200000}
NODES=${2:-10000}
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

shard_effective_lines() {
  local shard=$1
  local tmp="${shard}.tmp"
  local n=$(shard_lines "$shard")
  local t=0
  [[ -f "$tmp" ]] && t=$(shard_lines "$tmp")
  (( t > n )) && echo "$t" || echo "$n"
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
wave_lines = 0
wave_target = 0
for i in indices:
    f = out / f"aug_sp_{i:05d}.txt"
    tmp = Path(str(f) + ".tmp")
    lines = sum(1 for _ in f.open("rb")) if f.exists() else 0
    tmp_lines = sum(1 for _ in tmp.open("rb")) if tmp.exists() else 0
    effective = max(lines, tmp_lines)
    remaining = max(0, pos - effective)
    wave_lines += effective
    wave_target += pos
    shard_progress.append({
        "index": i,
        "name": f.name,
        "lines": effective,
        "on_disk": lines,
        "tmp_lines": tmp_lines,
        "target": pos,
        "remaining": remaining,
        "pct": min(100, int(100 * effective / pos)) if pos else 0,
        "done": effective >= pos,
    })
p.write_text(json.dumps({
    "phase": "$phase",
    "when": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    "positions_per_shard": pos,
    "nodes": $NODES,
    "n_shards": $N,
    "start_index": $START,
    "active_indices": indices,
    "active_shards": shard_progress,
    "wave_lines": wave_lines,
    "wave_target": wave_target,
    "wave_pct": min(100, int(100 * wave_lines / wave_target)) if wave_target else 100,
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
    tmp = Path(str(f) + ".tmp")
    if tmp.exists():
        lines = max(lines, sum(1 for _ in tmp.open("rb")))
    states[idx] = lines

# Finish shards closest to target first (e.g. legacy 150k before empty).
partial = sorted(
    (idx for idx, lines in states.items() if lines < pos),
    key=lambda i: states[i],
    reverse=True,
)
todo = partial[:n]

max_idx = max(states) if states else -1
next_idx = max_idx + 1
while len(todo) < n:
    if next_idx not in states or states[next_idx] < pos:
        todo.append(next_idx)
    next_idx += 1

print(json.dumps(todo[:n]))
PY
}

finalize_shard() {
  local shard=$1
  local tmp="${shard}.tmp"
  local shard_id=$(basename "$shard" .txt | sed 's/aug_sp_//')
  # Bash printf treats leading-zero shard names as octal; shards 00080+ are
  # decimal names, so force base 10 before formatting the log filename.
  local err=$(printf '%s_%05d.err' "$LOG" "$((10#$shard_id))")
  if [[ ! -f "$tmp" ]]; then
    return 0
  fi
  local n=$(shard_lines "$tmp")
  if (( n >= POS )); then
    mv -f "$tmp" "$shard"
    echo "  committed $shard ($n lines)" >&2
  elif (( n > $(shard_lines "$shard") )); then
    echo "  warn: $tmp only $n/$POS lines — keeping best on disk" >&2
    if (( n > $(shard_lines "$shard") )); then
      mv -f "$tmp" "$shard"
    else
      rm -f "$tmp"
    fi
  else
    rm -f "$tmp"
    if [[ -f "$err" ]] && [[ -s "$err" ]]; then
      echo "  worker failed $(basename "$shard"); see $err" >&2
      tail -3 "$err" >&2
    fi
    return 1
  fi
  return 0
}

mkdir -p "$OUT"
# Drop orphan tmps from dead workers when the shard is already complete.
for f in "$OUT"/aug_sp_*.txt.tmp; do
  [[ -f "$f" ]] || continue
  base="${f%.tmp}"
  if [[ -f "$base" ]] && (( $(shard_lines "$base") >= POS )); then
    echo "removing orphan ${f##*/}" >&2
    rm -f "$f"
  fi
done

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
pid_shard=()
for i in "${active_indices[@]}"; do
  seed=$((i * 7919))
  shard=$(printf '%s/aug_sp_%05d.txt' "$OUT" "$i")
  tmp="${shard}.tmp"
  lines=$(shard_effective_lines "$shard")
  if (( lines >= POS )); then
    echo "  skip $shard ($lines/$POS lines already complete)"
    rm -f "$tmp"
    continue
  fi
  if pgrep -f "aug_sp_$(printf '%05d' "$i")" >/dev/null 2>&1; then
    echo "  skip $shard (already running)" >&2
    continue
  fi
  rm -f "$tmp"
  err=$(printf '%s_%05d.err' "$LOG" "$i")
  printf 'datagen %s %s %s\nquit\n' "$POS" "$NODES" "$seed" \
    | "$BIN" >"$tmp" 2>"$err" &
  pid=$!
  pids+=("$pid")
  pid_shard+=("$shard")
  echo "  pid $pid -> $shard via .tmp ($lines/$POS on disk, seed $seed)"
done

if ((${#pids[@]} == 0)); then
  echo "datagen_parallel: no workers started (shards complete or busy)" >&2
  write_status idle
  exit 0
fi

write_status running

fail=0
while (( ${#pids[@]} )); do
  still_pids=()
  still_shards=()
  for j in "${!pids[@]}"; do
    pid=${pids[$j]}
    shard=${pid_shard[$j]}
    if kill -0 "$pid" 2>/dev/null; then
      still_pids+=("$pid")
      still_shards+=("$shard")
    else
      if wait "$pid"; then
        finalize_shard "$shard" || fail=1
      else
        fail=1
        rm -f "${shard}.tmp"
        echo "  worker pid $pid failed for $shard" >&2
      fi
    fi
  done
  # Bash with nounset treats an empty array expansion as unset on some
  # versions.  Keep both arrays explicitly empty when the last worker exits.
  if (( ${#still_pids[@]} )); then
    pids=("${still_pids[@]}")
    pid_shard=("${still_shards[@]}")
  else
    pids=()
    pid_shard=()
  fi
  if (( ${#pids[@]} )); then
    write_status running
    total=0
    want=0
    for i in "${active_indices[@]}"; do
      shard=$(printf '%s/aug_sp_%05d.txt' "$OUT" "$i")
      lines=$(shard_effective_lines "$shard")
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
