#!/usr/bin/env bash
# Everything that has to pass before a change is trusted.
#
# The python-chess passes dominate the runtime -- that library is the reference,
# not the fast path. They are also the only thing standing between a movegen bug
# and a plausible-looking wrong answer, so they run by default at depth 4.
# DEPTH=5 for a deeper (much slower) sweep.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}

cargo build --release || exit 1
echo "== classic perft"        && bash tests/perft_suite.sh
echo "== oracle perft"         && $PY tests/perft_oracle.py | tail -1
echo "== randomised perft"     && $PY tests/perft_fuzz.py   | tail -1
echo "== network inference"    && $PY tests/verify_net.py   | tail -1
echo "== bench (must be bit-identical run to run)"
printf 'bench 13\n' | ./target/release/sable | tail -3
