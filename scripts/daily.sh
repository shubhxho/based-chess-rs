#!/usr/bin/env bash
# One-shot daily health: tests + status page.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
cargo test --release -q
python3 scripts/daily_page.py
echo "open http://127.0.0.1:8375/daily after: python3 web/server.py"
