#!/usr/bin/env bash
# Head-to-head Elo of HEAD against a fixed pre-search-hardening baseline.
#
# The search work after 60ab7d3 is what the stress arena called "presearch".
# Rebuilding that baseline by hand (worktree → binary → arena → remove) is easy
# to get wrong and impossible to rerun identically. This script owns the whole
# loop and keeps openings fixed via SEED_BASE.
#
# usage: tests/presearch_ab.sh [BASE_SHA] [games] [nodes] [concurrency]
# defaults: 60ab7d3  200  25000  4
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BASE=${1:-60ab7d3}
GAMES=${2:-200}
NODES=${3:-25000}
CONC=${4:-4}
WT=/tmp/sable-presearch-wt
HEAD_BIN=/tmp/sable-head
BASE_BIN=/tmp/sable-presearch-bin
PY=${PYTHON:-"$ROOT/.venv/bin/python"}
[[ -x $PY ]] || PY=python3

cleanup() {
  if git -C "$ROOT" worktree list 2>/dev/null | grep -q "$WT"; then
    git -C "$ROOT" worktree remove "$WT" --force >/dev/null 2>&1 || true
  fi
  rm -rf "$WT"
}
trap cleanup EXIT

echo "building HEAD -> $HEAD_BIN"
cargo build --release --manifest-path "$ROOT/Cargo.toml" -q
cp "$ROOT/target/release/sable" "$HEAD_BIN"

echo "building baseline $BASE -> $BASE_BIN"
cleanup
git -C "$ROOT" worktree add --detach "$WT" "$BASE" >/dev/null
cargo build --release --manifest-path "$WT/Cargo.toml" -q
cp "$WT/target/release/sable" "$BASE_BIN"

echo "arena: HEAD vs $BASE  ($GAMES games, nodes $NODES, conc $CONC)"
SEED_BASE="${SEED_BASE:-9000}" "$PY" "$ROOT/arena.py" \
  "$HEAD_BIN" "$BASE_BIN" "$GAMES" "nodes $NODES" "$CONC"
