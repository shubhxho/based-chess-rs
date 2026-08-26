#!/usr/bin/env bash
# One terminal command: refresh daily board + serve chess UI.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
cargo build --release -q 2>/dev/null || cargo build --release -q
python3 scripts/daily_page.py
echo ""
echo "  play   http://127.0.0.1:8375"
echo "  daily  http://127.0.0.1:8375/daily"
echo "  gate   http://127.0.0.1:8375/gate_last.json"
echo ""
exec python3 web/server.py
