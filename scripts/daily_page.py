#!/usr/bin/env python3
"""Build web/daily.html — lab board with live datagen + Lichess progress."""

from __future__ import annotations

import datetime as dt
import html
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from lab_status import collect, sh  # noqa: E402

OUT = ROOT / "web" / "daily.html"


def esc(s: object) -> str:
    return html.escape(str(s))


def fmt(n: int | float | None) -> str:
    if n is None:
        return "?"
    if isinstance(n, float):
        return f"{n:,.2f}"
    return f"{int(n):,}"


def progress_bars(active: list[dict], el_id: str = "") -> str:
    if not active:
        return "<p class='dim'>No active datagen shards.</p>"
    rows = []
    for s in active:
        sid = esc(s.get("name") or f"aug_sp_{s.get('index', 0):05d}.txt")
        tmp_note = ""
        if s.get("tmp_lines"):
            tmp_note = f" (+{s['tmp_lines']:,} tmp)"
        rows.append(
            f"<div class='bar-row' data-shard='{sid}'>"
            f"<span class='bar-label'>{sid}</span>"
            f"<div class='bar'><i style='width:{s['pct']}%'></i></div>"
            f"<span class='bar-num'>{s['lines']:,}/{s['target']:,}{tmp_note}</span></div>"
        )
    return "\n".join(rows)


def dropped_table(dropped: dict | None) -> str:
    if not dropped:
        return ""
    total = sum(int(v) for v in dropped.values())
    if not total:
        return ""
    rows = "".join(
        f"<tr><th>{esc(k)}</th><td>{fmt(v)}</td>"
        f"<td class='dim'>{100 * int(v) / total:.1f}%</td></tr>"
        for k, v in sorted(dropped.items(), key=lambda kv: -int(kv[1]))
    )
    return (
        f"<table class='mini'><caption class='dim'>filtered out ({fmt(total):,} rows)</caption>"
        f"{rows}</table>"
    )


def lichess_panel(lf: dict) -> str:
    phase = lf.get("phase", "idle")
    lines = lf.get("lines", 0)
    shards = lf.get("shards", 0)
    skip_pct = lf.get("skip_pct", 0)
    keep_pct = lf.get("keep_pct", 0)
    kept = lf.get("kept", lf.get("emitted", 0))
    mx = lf.get("max_positions", 0)
    skip_row = lf.get("stream_row", 0)
    skip_tgt = lf.get("skip_target", 0)
    elapsed = lf.get("elapsed_s", 0)
    keep_rate = lf.get("keep_rate_pct")
    cursor = lf.get("stream_cursor", lf.get("absolute_skip"))
    row_fmt = lf.get("row_format", "FEN | cp_white | result")
    result_code = lf.get("result_code", 1)
    train_hint = lf.get("train_hint", "EVAL_W=1")

    bars = ""
    if phase == "skipping" and skip_tgt:
        bars = (
            f"<div class='bar-row'><span class='bar-label'>skip</span>"
            f"<div class='bar bar-skip'><i style='width:{skip_pct}%'></i></div>"
            f"<span class='bar-num'>{fmt(skip_row)}/{fmt(skip_tgt)}</span></div>"
        )
    elif phase in ("filtering", "running") and mx:
        bars = (
            f"<div class='bar-row'><span class='bar-label'>batch</span>"
            f"<div class='bar bar-keep'><i style='width:{keep_pct}%'></i></div>"
            f"<span class='bar-num'>{fmt(kept)}/{fmt(mx)}</span></div>"
        )

    meta = [
        ("Phase", f"<b id='lf-phase'>{esc(phase)}</b>"),
        ("Corpus", f"<span id='lf-corpus'>{shards} shards · {fmt(lines)} positions</span>"),
        ("Stream cursor", f"<span id='lf-cursor'>{fmt(cursor) if cursor else '—'}</span>"),
        ("Keep rate", f"<span id='lf-keep-rate'>{keep_rate}%" if keep_rate else "—"),
        ("Elapsed", f"<span id='lf-elapsed'>{fmt(elapsed)}s</span>" if elapsed else "—"),
    ]
    meta_rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in meta if v != "—")

    format_box = (
        f"<div class='callout' id='lf-format'>"
        f"<div class='callout-title'>Row format</div>"
        f"<code>{esc(row_fmt)}</code>"
        f"<p class='dim'>result=<b>{result_code}</b> draw placeholder (wdl=0.5) — no game was played. "
        f"Train with <code>{esc(train_hint)}</code> so the Stockfish cp label is the only teacher.</p>"
        f"</div>"
    )

    return (
        f"<table class='meta'>{meta_rows}</table>"
        + bars
        + format_box
        + dropped_table(lf.get("dropped"))
    )


def gate_section(g: dict | None, need: float) -> tuple[str, str, str]:
    if not g:
        return (
            "none yet",
            "<p class='dim'>No gate yet. <code>scripts/ml_cycle.sh 35 400 25</code></p>",
            "",
        )
    shipped = "SHIPPED" if g.get("shipped") else "REJECTED"
    elo = float(g.get("elo", 0))
    pct = min(100, max(0, int(100 * elo / need))) if need > 0 and elo > 0 else 0
    gate_row = f"{shipped}  Elo {elo:+.1f} (need {need:+.0f}) · {g.get('when', '')}"
    panel = (
        "<table>"
        + "".join(
            f"<tr><th>{k}</th><td>{v}</td></tr>"
            for k, v in [
                ("Result", f"<span id='gate-result'>{shipped}</span>"),
                ("Elo", f"<span id='gate-elo'>{elo:+.1f}</span> ± arena"),
                ("Threshold", f"+{need:.0f}"),
                ("Best so far", "+19.1 (1.34M SP, EVAL_W=0.9)"),
                ("Games / nodes", f"{g.get('games')} / {g.get('nodes')}"),
                ("Epochs", g.get("epochs")),
                ("When", g.get("when", "")),
            ]
        )
        + "</table>"
    )
    bar = (
        f"<div class='gate-bar'><div id='gate-fill' class='gate-fill' style='width:{pct}%'></div></div>"
        f"<p class='dim'>Arena must clear +{need:.0f} Elo before <code>net.bin</code> ships.</p>"
    )
    return gate_row, panel, bar


def main() -> None:
    now = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    snap = collect()
    branch = sh("git", "rev-parse", "--abbrev-ref", "HEAD") or "?"
    commit = sh("git", "log", "-1", "--oneline") or "?"
    status = sh("git", "status", "-sb") or "?"
    dirty = any(line[:1] in " MADRCU?" for line in status.splitlines()[1:])

    lf = snap.get("lichess") or {}
    sp_lines = snap.get("selfplay_lines", 0)
    sp_shards = snap.get("selfplay_shards", 0)
    hf_lines = lf.get("lines", 0)
    hf_shards = lf.get("shards", 0)

    g = snap.get("gate")
    need = float(snap.get("gate_need", 25))
    gate_row, gate_panel, gate_bar = gate_section(g, need)

    rows = [
        ("Generated", now),
        ("Branch", branch),
        ("HEAD", commit),
        ("Tree", "dirty" if dirty else "clean / synced"),
        ("Self-play", f"{sp_shards} shards · {sp_lines:,} lines"),
        ("Lichess HF", f"{hf_shards} shards · {hf_lines:,} lines"),
        ("Lichess cursor", fmt(lf.get("absolute_skip")) if lf.get("absolute_skip") else "?"),
        ("Last gate", gate_row),
    ]

    proc_html = (
        "<ul id='workers'>" + "".join(f"<li><code>{esc(p)}</code></li>" for p in snap["workers"]) + "</ul>"
        if snap["workers"]
        else "<p class='dim' id='workers'>No workers.</p>"
    )
    datagen_html = progress_bars(snap.get("active_shards") or [])
    wave = snap.get("datagen")
    sp_done = snap.get("selfplay_done", 0)
    sp_partial = snap.get("selfplay_partial", 0)
    if wave:
        wave_bar = ""
        if wave.get("wave_target"):
            wp = wave.get("wave_pct", 0)
            wave_bar = (
                f"<div class='bar-row'><span class='bar-label'>wave</span>"
                f"<div class='bar bar-keep'><i id='dg-wave-fill' style='width:{wp}%'></i></div>"
                f"<span class='bar-num' id='dg-wave-num'>{wave.get('wave_lines', 0):,}/{wave.get('wave_target', 0):,}</span></div>"
            )
        indices = wave.get("active_indices") or []
        idx_txt = ",".join(str(i) for i in indices) if indices else str(wave.get("start_index", "?"))
        datagen_html = (
            f"<p class='dim' id='dg-wave'>phase <b>{esc(wave.get('phase', '?'))}</b> · "
            f"shards [{idx_txt}] · {wave.get('positions_per_shard')}@{wave.get('nodes')}n · "
            f"{sp_done} done · {sp_partial} partial</p>"
            + wave_bar
            + datagen_html
        )
    lichess_html = lichess_panel(lf)

    table = "\n".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sable — lab daily</title>
<style>
  :root {{
    --bg: #0f0e0c; --panel: #1a1816; --panel2: #141210;
    --line: #2e2a26; --ink: #ece4d6; --amber: #e4a855;
    --green: #7cb87c; --red: #d47272; --dim: #9a8f82;
    --mono: "IBM Plex Mono", "SF Mono", ui-monospace, Menlo, monospace;
    --sans: "IBM Plex Sans", system-ui, sans-serif;
  }}
  @media (prefers-color-scheme: light) {{
    :root {{ --bg: #f6f2ea; --panel: #fffdf9; --panel2: #f0ebe3;
      --line: #ddd4c8; --ink: #2a2420; --dim: #6f6459; --amber: #a86a1c; }}
  }}
  * {{ box-sizing: border-box; margin: 0; }}
  body {{ background: var(--bg); color: var(--ink);
    font-family: var(--mono); font-size: 13px; min-height: 100vh; }}
  .top {{ border-bottom: 1px solid var(--line); padding: 20px 24px;
    display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 8px; }}
  h1 {{ font-size: 13px; letter-spacing: .2em; font-weight: 600; }}
  h1 span {{ color: var(--amber); }}
  .live {{ font-size: 11px; color: var(--dim); }}
  .live b {{ color: var(--green); font-weight: 500; }}
  main {{ max-width: 960px; margin: 0 auto; padding: 24px 20px 48px; }}
  .grid {{ display: grid; gap: 16px; }}
  @media (min-width: 720px) {{ .grid-2 {{ grid-template-columns: 1fr 1fr; }} }}
  .tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-bottom: 20px; }}
  .tile {{ background: var(--panel2); border: 1px solid var(--line); padding: 14px 16px; }}
  .tile .n {{ font-size: 22px; font-weight: 600; color: var(--amber); line-height: 1.1; }}
  .tile .l {{ font-size: 10px; letter-spacing: .14em; text-transform: uppercase;
    color: var(--dim); margin-top: 6px; }}
  .panel {{ background: var(--panel); border: 1px solid var(--line); padding: 18px 20px; }}
  .label {{ color: var(--dim); font-size: 10px; letter-spacing: .22em;
    text-transform: uppercase; margin-bottom: 14px; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ text-align: left; padding: 7px 0; border-bottom: 1px solid var(--line); vertical-align: top; }}
  th {{ color: var(--dim); font-weight: 500; width: 36%; font-size: 12px; }}
  table.mini th {{ width: 28%; font-size: 11px; }}
  table.mini caption {{ text-align: left; margin-bottom: 8px; font-size: 11px; }}
  code, .dim {{ color: var(--dim); font-size: 12px; }}
  code {{ font-family: var(--mono); }}
  ul {{ padding-left: 18px; }} li {{ margin: 5px 0; word-break: break-all; }}
  a {{ color: var(--amber); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .cmds code {{ display: block; padding: 10px 12px; margin: 6px 0;
    border: 1px solid var(--line); background: var(--panel2); white-space: pre-wrap; }}
  .bar-row {{ display: grid; grid-template-columns: 72px 1fr 110px; gap: 10px;
    align-items: center; margin: 10px 0; font-size: 11px; }}
  .bar-label {{ color: var(--dim); }}
  .bar {{ height: 9px; background: var(--line); overflow: hidden; border-radius: 1px; }}
  .bar i {{ display: block; height: 100%; background: var(--amber); transition: width .4s ease; }}
  .bar-skip i {{ background: #6a8fad; }}
  .bar-keep i {{ background: var(--amber); }}
  .bar-num {{ color: var(--dim); text-align: right; font-size: 10px; }}
  .gate-bar {{ height: 11px; background: var(--line); margin: 12px 0 8px; border-radius: 1px; }}
  .gate-fill {{ height: 100%; background: linear-gradient(90deg, var(--amber), var(--green));
    max-width: 100%; transition: width .4s ease; }}
  .callout {{ margin-top: 14px; padding: 12px 14px; background: var(--panel2);
    border-left: 3px solid var(--amber); }}
  .callout-title {{ font-size: 10px; letter-spacing: .16em; text-transform: uppercase;
    color: var(--dim); margin-bottom: 8px; }}
  .callout code {{ display: block; margin: 6px 0; color: var(--ink); }}
  .foot {{ margin-top: 28px; font-size: 12px; color: var(--dim); }}
</style>
</head>
<body>
<header class="top">
  <h1>SABLE <span>LAB</span></h1>
  <p class="live"><b id="pulse">●</b> live · <span id="api-ts">{esc(snap.get('generated_at', ''))}</span></p>
</header>
<main>
  <div class="tiles">
    <div class="tile"><div class="n" id="t-sp">{sp_lines:,}</div><div class="l">Self-play lines</div></div>
    <div class="tile"><div class="n" id="t-hf">{hf_lines:,}</div><div class="l">Lichess positions</div></div>
    <div class="tile"><div class="n" id="t-gate">{f"{float(g.get('elo', 0)):+.1f}" if g else "—"}</div><div class="l">Last gate Elo</div></div>
    <div class="tile"><div class="n">+{need:.0f}</div><div class="l">Ship threshold</div></div>
  </div>

  <div class="grid grid-2">
    <section class="panel">
      <div class="label">Gate</div>
      {gate_bar}
      <div id="gate-panel">{gate_panel}</div>
    </section>
    <section class="panel">
      <div class="label">Snapshot</div>
      <table>{table}</table>
    </section>
  </div>

  <section class="panel" style="margin-top:16px">
    <div class="label">Lichess HF prepare</div>
    <div id="lichess">{lichess_html}</div>
  </section>

  <section class="panel" style="margin-top:16px">
    <div class="label">Self-play datagen</div>
    <div id="datagen">{datagen_html}</div>
  </section>

  <div class="grid grid-2" style="margin-top:16px">
    <section class="panel">
      <div class="label">Workers</div>
      {proc_html}
    </section>
    <section class="panel cmds">
      <div class="label">Commands</div>
      <code>scripts/lab.sh all</code>
      <code>scripts/lab_supervisor.sh</code>
      <code>scripts/datagen_daemon.sh</code>
      <code>DATA_DIR=data/lichess-sf EVAL_W=1 python train.py …</code>
      <code>scripts/ml_cycle.sh 35 400 25</code>
    </section>
  </div>

  <p class="foot"><a href="/">← play chess</a></p>
</main>
<script>
function fmt(n) {{ return n == null ? '—' : Number(n).toLocaleString(); }}

function barRow(label, pct, left, right, cls='') {{
  return `<div class="bar-row"><span class="bar-label">${{label}}</span>`
    + `<div class="bar ${{cls}}"><i style="width:${{pct}}%"></i></div>`
    + `<span class="bar-num">${{left}}/${{right}}</span></div>`;
}}

function datagenHtml(active, wave) {{
  let h = '';
  if (wave) {{
    const idx = (wave.active_indices || []).join(',') || wave.start_index;
    h += `<p class="dim" id="dg-wave">phase <b>${{wave.phase}}</b> · shards [${{idx}}] · ${{wave.positions_per_shard}}@${{wave.nodes}}n</p>`;
    if (wave.wave_target) {{
      h += barRow('wave', wave.wave_pct || 0, fmt(wave.wave_lines), fmt(wave.wave_target), 'bar-keep');
    }}
  }}
  if (!active?.length) return h + `<p class="dim">No active datagen shards.</p>`;
  for (const s of active) {{
    const name = s.name || `aug_sp_${{String(s.index).padStart(5,'0')}}.txt`;
    const tmp = s.tmp_lines ? ` (+${{Number(s.tmp_lines).toLocaleString()}} tmp)` : '';
    h += `<div class="bar-row" data-shard="${{name}}">`
      + `<span class="bar-label">${{name}}</span>`
      + `<div class="bar"><i style="width:${{s.pct}}%"></i></div>`
      + `<span class="bar-num">${{fmt(s.lines)}}/${{fmt(s.target)}}${{tmp}}</span></div>`;
  }}
  return h;
}}

function workersHtml(list) {{
  if (!list?.length) return `<p class="dim" id="workers">No workers.</p>`;
  return `<ul id="workers">${{list.map(p => `<li><code>${{p}}</code></li>`).join('')}}</ul>`;
}}

async function refresh() {{
  try {{
    const r = await fetch('/api/status');
    if (!r.ok) return;
    const d = await r.json();
    document.getElementById('api-ts').textContent = d.generated_at || '—';
    document.getElementById('t-sp').textContent = fmt(d.selfplay_lines);
    document.getElementById('t-hf').textContent = fmt(d.lichess?.lines);
    if (d.gate_elo != null) {{
      document.getElementById('t-gate').textContent = `${{d.gate_elo >= 0 ? '+' : ''}}${{d.gate_elo.toFixed(1)}}`;
      const need = d.gate_need || 25;
      const pct = Math.min(100, Math.max(0, Math.round(100 * d.gate_elo / need)));
      const fill = document.getElementById('gate-fill');
      if (fill) fill.style.width = pct + '%';
      const ge = document.getElementById('gate-elo');
      if (ge) ge.textContent = `${{d.gate_elo >= 0 ? '+' : ''}}${{d.gate_elo.toFixed(1)}}`;
      const gr = document.getElementById('gate-result');
      if (gr && d.gate) gr.textContent = d.gate.shipped ? 'SHIPPED' : 'REJECTED';
    }}
    const lf = d.lichess;
    if (lf) {{
      const phase = document.getElementById('lf-phase');
      if (phase) phase.textContent = lf.phase;
      const corp = document.getElementById('lf-corpus');
      if (corp) corp.textContent = `${{lf.shards}} shards · ${{fmt(lf.lines)}} positions`;
      const cur = document.getElementById('lf-cursor');
      if (cur) cur.textContent = fmt(lf.stream_cursor || lf.absolute_skip);
      const kr = document.getElementById('lf-keep-rate');
      if (kr && lf.keep_rate_pct != null) kr.textContent = lf.keep_rate_pct + '%';
      const el = document.getElementById('lf-elapsed');
      if (el && lf.elapsed_s) el.textContent = lf.elapsed_s + 's';
      let bars = '';
      if (lf.phase === 'skipping' && lf.skip_target) {{
        bars = barRow('skip', lf.skip_pct, fmt(lf.stream_row), fmt(lf.skip_target), 'bar-skip');
      }} else if (lf.max_positions && lf.phase !== 'done' && lf.phase !== 'idle') {{
        bars = barRow('batch', lf.keep_pct, fmt(lf.kept), fmt(lf.max_positions), 'bar-keep');
      }}
      const box = document.getElementById('lichess');
      if (box && bars) {{
        const callout = box.querySelector('.callout');
        const table = box.querySelector('table.meta');
        const dropped = box.querySelector('table.mini');
        box.innerHTML = (table?.outerHTML || '') + bars + (callout?.outerHTML || '') + (dropped?.outerHTML || '');
      }}
    }}
    const dg = document.getElementById('datagen');
    if (dg) dg.innerHTML = datagenHtml(d.active_shards, d.datagen);
    const wk = document.getElementById('workers');
    if (wk) {{
      const parent = wk.parentElement;
      if (parent) parent.innerHTML = '<div class="label">Workers</div>' + workersHtml(d.workers);
    }}
  }} catch (_) {{}}
}}
setInterval(refresh, 12000);
refresh();
</script>
</body>
</html>
"""
    OUT.write_text(html_doc)
    print(f"wrote {OUT.relative_to(ROOT)} ({sp_lines:,} sp · {hf_lines:,} lichess)")


if __name__ == "__main__":
    main()
