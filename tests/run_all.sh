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

# The same three gates CI runs first, in the same form. They used to be absent
# here, so a green local run could still fail the build on formatting -- and the
# clippy line in particular has to keep `--release`, because the dev profile
# builds the no_std binary with unwinding and dies before linting anything.
echo "== fmt"                  && cargo fmt --check && echo "clean"
echo "== clippy"               && cargo clippy --release --all-targets -- -D warnings 2>&1 | tail -1
echo "== unit tests"           && cargo test --release 2>&1 | grep "^test result" | tail -1

echo "== classic perft"        && bash tests/perft_suite.sh
echo "== oracle perft"         && $PY tests/perft_oracle.py | tail -1
echo "== randomised perft"     && $PY tests/perft_fuzz.py   | tail -1
echo "== network inference"    && $PY tests/verify_net.py   | tail -1

echo "== bench (must be bit-identical run to run, with or without a trailing quit)"
printf 'bench 13\n' | ./target/release/sable | tail -3
printf 'bench 13\nquit\n' | ./target/release/sable | grep "Total nodes"
