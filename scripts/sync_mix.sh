#!/usr/bin/env bash
# Refresh data/mix for the 3000-Elo candidate path.
#
# Load order matters in train.py (sorted paths; later shard wins on FEN overlap):
#   1. Newest Lichess HF   → Stockfish cp on quiet positions
#   2. Finished self-play  → game outcomes win on shared FENs + SP_BOOST
#
# Prior mix (SP first, HF last) measured −1.7 ±34 twice: SF overwrote the
# outcome labels that shipping was trained on. Flip so play signal wins.
#
#   scripts/sync_mix.sh           # 2 newest HF + 10 SP
#   scripts/sync_mix.sh 12 3
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
SP_KEEP=${1:-10}
HF_KEEP=${2:-2}
MIX="$ROOT/data/mix"
MIN_SP_LINES=${MIN_SP_LINES:-100000}

mkdir -p "$MIX"
find "$MIX" -maxdepth 1 -type l -delete

shopt -s nullglob

hf_all=("$ROOT"/data/lichess-sf/aug_hf_*.txt)
hf=0
if ((${#hf_all[@]} > 0)); then
  start=0
  if ((${#hf_all[@]} > HF_KEEP)); then
    start=$((${#hf_all[@]} - HF_KEEP))
  fi
  i=$start
  while (( i < ${#hf_all[@]} )); do
    f=${hf_all[$i]}
    base=$(basename "$f")
    # Plain aug_hf_* sorts before aug_z_sp_* .
    ln -s "../lichess-sf/$base" "$MIX/$base"
    hf=$((hf + 1))
    i=$((i + 1))
  done
fi

sp_list=$(
  for f in "$ROOT"/data/selfplay/aug_sp_0*.txt; do
    lines=$(wc -l <"$f" | tr -d ' ')
    if (( lines >= MIN_SP_LINES )); then
      printf '%s\n' "$(basename "$f")"
    fi
  done | sort | tail -n "$SP_KEEP"
)

sp=0
if [[ -n "$sp_list" ]]; then
  while IFS= read -r base; do
    [[ -z "$base" ]] && continue
    # Prefix aug_z_ so SP sorts after HF → outcomes overwrite SF on overlap.
    # Name still contains aug_sp_ so SP_BOOST matches.
    ln -s "../selfplay/$base" "$MIX/aug_z_$base"
    sp=$((sp + 1))
  done <<<"$sp_list"
fi

hf_lines=0
sp_lines=0
for f in "$MIX"/aug_hf_*.txt; do
  [[ -e "$f" ]] || continue
  n=$(wc -l <"$f" | tr -d ' ')
  hf_lines=$((hf_lines + n))
done
if [[ -n "$sp_list" ]]; then
  while IFS= read -r base; do
    [[ -z "$base" ]] && continue
    n=$(wc -l <"$MIX/aug_z_$base" | tr -d ' ')
    sp_lines=$((sp_lines + n))
  done <<<"$sp_list"
fi
echo "mix synced: $hf newest HF (~$hf_lines lines) then $sp SP (~$sp_lines lines) → $MIX"
echo "  order: HF first, SP last (outcomes overwrite SF on overlap). SP_BOOST applies to aug_sp_*."
echo "  DATA_DIR=data/mix DATA_GLOB='aug*.txt' EVAL_W=0.9 SP_BOOST=3.0"
