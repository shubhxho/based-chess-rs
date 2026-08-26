#!/usr/bin/env python3
"""Build web/daily.html — a lab status page for corpus, git, and pipeline health.

    .venv/bin/python scripts/daily_page.py
    # then: python web/server.py  →  http://127.0.0.1:8375/daily
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "daily.html"


def sh(*args: str) -> str:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def count_lines(pattern: str) -> tuple[int, int]:
    files = sorted(ROOT.glob(pattern))
    total = 0
    for f in files:
        try:
            total += sum(1 for _ in f.open("rb"))
        except OSError:
            pass
    return len(files), total


def main() -> None:
    now = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    branch = sh("git", "rev-parse", "--abbrev-ref", "HEAD") or "?"
    commit = sh("git", "log", "-1", "--oneline") or "?"
    status = sh("git", "status", "-sb") or "?"
    ahead = "origin" in status and "ahead" in status
    dirty = any(line[:1] in " MADRCU?" for line in status.splitlines()[1:])

    sp_n, sp_lines = count_lines("data/selfplay/aug_sp_*.txt")
    hf_n, hf_lines = count_lines("data/lichess-sf/aug_hf_*.txt")

    resume = None
    manifest = ROOT / "data" / "lichess-sf" / "hf_source.json"
    if manifest.exists():
        try:
            m = json.loads(manifest.read_text())
            base = int(m.get("absolute_skip_rows", m.get("skip_rows", 0)) or 0)
            scanned = int(m.get("scanned_rows", 0) or 0)
            resume = base + scanned
        except (json.JSONDecodeError, TypeError, ValueError):
            resume = None

    procs = sh("pgrep", "-lf", "prepare_hf|train_gate|datagen_parallel|target/release/sable")
    alive = []
    for line in procs.splitlines():
        if "Helper" in line or "pgrep" in line:
            continue
        if any(k in line for k in ("prepare_hf", "train_gate", "datagen", "release/sable")):
            alive.append(line[:120])

    net_ok = (ROOT / "net.bin").exists()
    gate = ROOT / "train_gate.py"
    presearch = ROOT / "tests" / "presearch_ab.sh"

    rows = [
        ("Generated", now),
        ("Branch", branch),
        ("HEAD", commit),
        ("Tree", "dirty" if dirty else ("ahead of origin" if ahead else "clean / synced")),
        ("Self-play shards", f"{sp_n} files · {sp_lines:,} positions"),
        ("Lichess shards", f"{hf_n} files · {hf_lines:,} positions"),
        ("Lichess --resume skip", f"{resume:,}" if resume is not None else "unknown"),
        ("net.bin", "present" if net_ok else "missing"),
        ("train_gate", "yes" if gate.exists() else "no"),
        ("presearch_ab", "yes" if presearch.exists() else "no"),
    ]

    proc_html = (
        "<ul>" + "".join(f"<li><code>{p}</code></li>" for p in alive) + "</ul>"
        if alive
        else "<p class='dim'>No prepare / datagen / train_gate workers visible.</p>"
    )
    table = "\n".join(
        f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sable — daily</title>
<style>
  :root {{
    --bg: #131110; --panel: #1c1917; --line: #2c2724;
    --ink: #e8e0d2; --amber: #e0a458; --dim: #9b8f83;
    --mono: "SF Mono", ui-monospace, Menlo, monospace;
  }}
  @media (prefers-color-scheme: light) {{
    :root {{
      --bg: #f4f1ea; --panel: #fffdf8; --line: #ddd5c7;
      --ink: #2a2420; --dim: #6f6459; --amber: #a86a1c;
    }}
  }}
  * {{ box-sizing: border-box; margin: 0; }}
  body {{
    background: var(--bg); color: var(--ink);
    font-family: var(--mono); font-size: 14px;
    min-height: 100vh; padding: 32px 20px;
  }}
  main {{ max-width: 720px; margin: 0 auto; }}
  h1 {{ font-size: 15px; letter-spacing: .16em; font-weight: 600; }}
  h1 span {{ color: var(--amber); }}
  .sub {{ color: var(--dim); font-size: 12px; margin: 8px 0 28px; line-height: 1.5; }}
  .panel {{
    background: var(--panel); border: 1px solid var(--line);
    padding: 16px 18px; margin-bottom: 16px;
  }}
  .label {{
    color: var(--dim); font-size: 10px; letter-spacing: .22em;
    text-transform: uppercase; margin-bottom: 12px;
  }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ text-align: left; padding: 8px 0; border-bottom: 1px solid var(--line); vertical-align: top; }}
  th {{ color: var(--dim); font-weight: 500; width: 38%; }}
  code, .dim {{ color: var(--dim); font-size: 12px; }}
  ul {{ padding-left: 18px; }}
  li {{ margin: 6px 0; }}
  a {{ color: var(--amber); }}
  .cmds code {{
    display: block; padding: 10px 12px; margin: 8px 0;
    border: 1px solid var(--line); color: var(--ink); white-space: pre-wrap;
  }}
</style>
</head>
<body>
<main>
  <h1>SABLE <span>DAILY</span></h1>
  <p class="sub">Lab board for corpus size, git HEAD, and live pipeline workers.
  Regenerated by <code>scripts/daily_page.py</code>. Not a rating claim.</p>

  <section class="panel">
    <div class="label">Snapshot</div>
    <table>{table}</table>
  </section>

  <section class="panel">
    <div class="label">Live workers</div>
    {proc_html}
  </section>

  <section class="panel cmds">
    <div class="label">Next commands</div>
    <code>.venv/bin/python prepare_hf.py data/lichess-sf --max-positions 300000 --resume</code>
    <code>scripts/datagen_parallel.sh 150000 6000 2 6</code>
    <code>DATA_DIR=data/selfplay DATA_GLOB='aug_sp_*.txt' EVAL_W=0.9 OUT_SCALE=0.70 \\
  .venv/bin/python train_gate.py --epochs 15 --games 400 --min-elo 25</code>
    <code>tests/presearch_ab.sh 60ab7d3 200 25000 4</code>
  </section>

  <p class="sub"><a href="/">← play</a></p>
</main>
</body>
</html>
"""
    OUT.write_text(html)
    print(f"wrote {OUT.relative_to(ROOT)} ({sp_lines:,} sp · {hf_lines:,} lichess)")


if __name__ == "__main__":
    main()
