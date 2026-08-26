#!/usr/bin/env python3
"""Build web/daily.html — lab board with live datagen progress."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from lab_status import collect, sh  # noqa: E402

OUT = ROOT / "web" / "daily.html"


def count_lines(pattern: str) -> tuple[int, int]:
    files = sorted(ROOT.glob(pattern))
    total = 0
    for f in files:
        try:
            total += sum(1 for _ in f.open("rb"))
        except OSError:
            pass
    return len(files), total


def progress_bars(active: list[dict]) -> str:
    if not active:
        return "<p class='dim'>No active datagen shards.</p>"
    rows = []
    for s in active:
        rows.append(
            f"<div class='bar-row'><span>{s['name']}</span>"
            f"<div class='bar'><i style='width:{s['pct']}%'></i></div>"
            f"<span class='dim'>{s['lines']:,}/{s['target']:,}</span></div>"
        )
    return "\n".join(rows)


def main() -> None:
    now = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    snap = collect()
    branch = sh("git", "rev-parse", "--abbrev-ref", "HEAD") or "?"
    commit = sh("git", "log", "-1", "--oneline") or "?"
    status = sh("git", "status", "-sb") or "?"
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
            pass

    g = snap.get("gate")
    gate_row = "none yet"
    gate_panel = "<p class='dim'>No gate yet. <code>scripts/ml_cycle.sh</code></p>"
    gate_bar = ""
    if g:
        shipped = "SHIPPED" if g.get("shipped") else "REJECTED"
        elo = float(g.get("elo", 0))
        need = float(g.get("min_elo", 25))
        pct = min(100, max(0, int(100 * elo / need))) if need > 0 and elo > 0 else 0
        gate_row = f"{shipped}  Elo {elo:+.1f} (need {need:+.0f}) · {g.get('when', '')}"
        gate_panel = (
            "<table>"
            + "".join(
                f"<tr><th>{k}</th><td>{v}</td></tr>"
                for k, v in [
                    ("Result", shipped),
                    ("Elo", f"{elo:+.1f} ± arena"),
                    ("Threshold", f"{need:+.1f}"),
                    ("Best so far", "+19.1 (1.34M SP)"),
                    ("Games / nodes", f"{g.get('games')} / {g.get('nodes')}"),
                    ("Epochs", g.get("epochs")),
                    ("When", g.get("when", "")),
                ]
            )
            + "</table>"
        )
        gate_bar = (
            f"<div class='gate-bar'><div class='gate-fill' style='width:{pct}%'></div></div>"
            f"<p class='dim'>Gate progress toward +{need:.0f} Elo (not shipped until bar clears threshold)</p>"
        )

    rows = [
        ("Generated", now),
        ("Branch", branch),
        ("HEAD", commit),
        ("Tree", "dirty" if dirty else "clean / synced"),
        ("Self-play", f"{sp_n} shards · {sp_lines:,} lines"),
        ("Lichess", f"{hf_n} shards · {hf_lines:,} lines"),
        ("Lichess skip", f"{resume:,}" if resume is not None else "?"),
        ("Last gate", gate_row),
    ]

    proc_html = (
        "<ul>" + "".join(f"<li><code>{p}</code></li>" for p in snap["workers"]) + "</ul>"
        if snap["workers"]
        else "<p class='dim'>No workers.</p>"
    )
    datagen_html = progress_bars(snap.get("active_shards") or [])
    wave = snap.get("datagen")
    if wave:
        datagen_html = (
            f"<p class='dim'>wave {wave.get('phase')} · start {wave.get('start_index')} "
            f"· {wave.get('positions_per_shard')}@{wave.get('nodes')}n</p>"
            + datagen_html
        )

    table = "\n".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)

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
    :root {{ --bg: #f4f1ea; --panel: #fffdf8; --line: #ddd5c7;
      --ink: #2a2420; --dim: #6f6459; --amber: #a86a1c; }}
  }}
  * {{ box-sizing: border-box; margin: 0; }}
  body {{ background: var(--bg); color: var(--ink); font-family: var(--mono);
    font-size: 14px; min-height: 100vh; padding: 32px 20px; }}
  main {{ max-width: 720px; margin: 0 auto; }}
  h1 {{ font-size: 15px; letter-spacing: .16em; font-weight: 600; }}
  h1 span {{ color: var(--amber); }}
  .sub {{ color: var(--dim); font-size: 12px; margin: 8px 0 28px; line-height: 1.5; }}
  .panel {{ background: var(--panel); border: 1px solid var(--line);
    padding: 16px 18px; margin-bottom: 16px; }}
  .label {{ color: var(--dim); font-size: 10px; letter-spacing: .22em;
    text-transform: uppercase; margin-bottom: 12px; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ text-align: left; padding: 8px 0; border-bottom: 1px solid var(--line); }}
  th {{ color: var(--dim); font-weight: 500; width: 38%; }}
  code, .dim {{ color: var(--dim); font-size: 12px; }}
  ul {{ padding-left: 18px; }} li {{ margin: 6px 0; }}
  a {{ color: var(--amber); }}
  .cmds code {{ display: block; padding: 10px 12px; margin: 8px 0;
    border: 1px solid var(--line); white-space: pre-wrap; }}
  .bar-row {{ display: grid; grid-template-columns: 88px 1fr 100px; gap: 10px;
    align-items: center; margin: 8px 0; font-size: 11px; }}
  .bar {{ height: 8px; background: var(--line); overflow: hidden; }}
  .bar i {{ display: block; height: 100%; background: var(--amber); }}
  .gate-bar {{ height: 10px; background: var(--line); margin: 12px 0 6px; }}
  .gate-fill {{ height: 100%; background: var(--amber); max-width: 100%; }}
</style>
</head>
<body>
<main>
  <h1>SABLE <span>LAB</span></h1>
  <p class="sub">Auto-refreshes every 30s. <code>scripts/lab.sh all</code> = UI + datagen + Lichess.</p>

  <section class="panel">
    <div class="label">Snapshot</div>
    <table>{table}</table>
  </section>

  <section class="panel">
    <div class="label">Gate (+25 to ship)</div>
    {gate_bar}
    {gate_panel}
  </section>

  <section class="panel">
    <div class="label">Datagen progress</div>
    {datagen_html}
  </section>

  <section class="panel">
    <div class="label">Workers</div>
    {proc_html}
  </section>

  <section class="panel cmds">
    <div class="label">Commands</div>
    <code>scripts/lab.sh all</code>
    <code>scripts/datagen_daemon.sh</code>
    <code>scripts/ml_cycle.sh 35 400 25</code>
  </section>

  <p class="sub"><a href="/">← play chess</a></p>
</main>
<script>
setInterval(() => location.reload(), 30000);
</script>
</body>
</html>
"""
    OUT.write_text(html)
    print(f"wrote {OUT.relative_to(ROOT)} ({sp_lines:,} sp · {hf_lines:,} lichess)")


if __name__ == "__main__":
    main()
