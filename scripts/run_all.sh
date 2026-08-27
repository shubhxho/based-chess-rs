# One command: build, refresh daily, start full lab stack.
#
#   scripts/run_all.sh           # supervisor + play UI (default)
#   scripts/run_all.sh bg        # background supervisor only
#   scripts/run_all.sh refresh   # python3 daily snapshot only
#   scripts/run_all.sh 3000      # Lichess+SP mix candidate path
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
MODE=${1:-}

banner() {
  echo ""
  echo "  SABLE LAB"
  echo "  play   http://127.0.0.1:8375"
  echo "  daily  http://127.0.0.1:8375/daily"
  echo "  blog   http://127.0.0.1:8375/blog"
  echo "  status http://127.0.0.1:8375/api/status"
  echo "  3000   scripts/push_3000.sh"
  echo ""
}

refresh() {
  python3 scripts/daily_page.py
  python3 scripts/blog_page.py
  python3 -m py_compile prepare_hf.py scripts/lab_status.py scripts/daily_page.py scripts/blog_page.py web/server.py
}

case "$MODE" in
  refresh)
    refresh
    banner
    echo "  refreshed web/daily.html"
    exit 0
    ;;
  3000|push3000)
    refresh
    banner
    exec bash scripts/push_3000.sh all
    ;;
  bg)
    refresh
    banner
    exec bash scripts/lab_supervisor.sh bg
    ;;
  *)
    refresh
    banner
    exec bash scripts/lab_supervisor.sh
    ;;
esac
