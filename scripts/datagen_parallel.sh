#!/usr/bin/env bash
# Parallel self-play shard generation for distillation.
#
# Strength here is bottlenecked by teacher-labelled self-play volume, not by
# another Stockfish dump. Run several independent seeds; train_gate.py decides
# whether a net trained on them is allowed to replace shipping weights.
#
# usage: scripts/datagen_parallel.sh [positions_per_shard] [nodes] [n_shards] [start_index]
# defaults: 200000  6000  4  2
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
POS=${1:-200000}
NODES=${2:-6000}
N=${3:-4}
START=${4:-2}
OUT="$ROOT/data/selfplay"
BIN="$ROOT/target/release/sable"
LOG=/tmp/sable-datagen

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
  pids+=($!)
  echo "  pid ${pids[-1]} -> $shard (seed $seed)"
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "datagen worker $pid failed" >&2
    fail=1
  fi
done

wc -l "$OUT"/aug_sp_*.txt
if (( fail )); then
  exit 1
fi
echo "datagen_parallel done"
