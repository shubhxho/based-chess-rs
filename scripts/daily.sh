#!/usr/bin/env bash
# One-shot: tests, daily snapshot, print URLs.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
cargo test --release -q
bash scripts/run_all.sh refresh
echo "start full lab: scripts/run_all.sh"
