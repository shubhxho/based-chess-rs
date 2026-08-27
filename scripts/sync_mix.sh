#!/usr/bin/env bash
# Refresh data/mix for the 3000-Elo candidate path.
#
# Load order matters in train.py (sorted paths; later shard wins on FEN overlap):
#   1. Finished self-play  → game outcomes (EVAL_W blend) + SP_BOOST
#   2. Newest Lichess HF   → Stockfish cp labels overwrite shared FENs
#
# Also: a LIMIT that fills on early shards never reaches later ones. Keep HF_KEEP
# modest so the full mix fits in memory without a LIMIT that skips provenance.
#
#   scripts/sync_mix.sh           # 5 SP + 3 newest HF (~1.6M, fits busy 16GB Mac)
#   scripts/sync_mix.sh 8 5
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
SP_KEEP=${1:-5}
HF_KEEP=${2:-3}
MIX="$ROOT/data/mix"
MIN_SP_LINES=${MIN_SP_LINES:-100000}

mkdir -p "$MIX"
find "$MIX" -maxdepth 1 -type l -delete

shopt -s nullglob

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
    # Keep aug_sp_ name so SP_BOOST matches and these sort before aug_z_hf_*.
    ln -s "../selfplay/$base" "$MIX/$base"
    sp=$((sp + 1))
  done <<<"$sp_list"
fi

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
    # Prefix aug_z_ so HF sorts after aug_sp_* → SF labels win on overlap.
    ln -s "../lichess-sf/$base" "$MIX/aug_z_$base"
    hf=$((hf + 1))
    i=$((i + 1))
  done
fi

sp_lines=0
hf_lines=0
# Per-file wc only — never `cat mix/* | wc` (multi-GB hang).
if [[ -n "$sp_list" ]]; then
  while IFS= read -r base; do
    [[ -z "$base" ]] && continue
    n=$(wc -l <"$MIX/$base" | tr -d ' ')
    sp_lines=$((sp_lines + n))
  done <<<"$sp_list"
fi
for f in "$MIX"/aug_z_aug_hf_*.txt; do
  [[ -e "$f" ]] || continue
  n=$(wc -l <"$f" | tr -d ' ')
  hf_lines=$((hf_lines + n))
done
echo "mix synced: $sp SP (~$sp_lines lines) then $hf newest HF (~$hf_lines lines) → $MIX"
echo "  order: SP first, HF last (SF overwrites shared FENs). SP_BOOST applies to aug_sp_*."
echo "  DATA_DIR=data/mix DATA_GLOB='aug*.txt' EVAL_W=0.9 SP_BOOST=2.0"
