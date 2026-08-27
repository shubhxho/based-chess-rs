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
        f"<table class='mini'><caption class='dim'>filtered out ({fmt(total)} rows)</caption>"
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
        f"<div class='callout callout-gold' id='lf-format'>"
        f"<div class='callout-title'>Lichess trainer</div>"
        f"<code>{esc(row_fmt)}</code>"
        f"<p>result=<b>{result_code}</b> is a neutral draw placeholder — with "
        f"<code>{esc(train_hint)}</code> only the Stockfish cp label trains the net. "
        f"The dummy result cannot pull the target toward win/loss.</p>"
        f"<p class='dim trust'>Ship bar is still <b>self-play +25 Elo</b>. "
        f"Toward ~3000: grow Lichess → <code>scripts/push_3000.sh</code> "
        f"(SF+SP mix) → arena → SP gate → Stockfish calibrate.</p>"
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
    status = g.get("status") or ("SHIPPED" if g.get("shipped") else "REJECTED")
    status_u = str(status).upper()
    elo = g.get("elo")
    elo_f = float(elo) if elo is not None else None
    err = g.get("elo_err")
    data_l = str(g.get("data_dir", "")).lower()
    if g.get("path"):
        path = g["path"]
    elif "lichess" in data_l:
        path = "lichess"
    elif "mix" in data_l:
        path = "mix"
    else:
        path = "selfplay"
    ship_need = 25.0
    pct = min(100, max(0, int(100 * elo_f / ship_need))) if ship_need > 0 and elo_f and elo_f > 0 else 0
    err_txt = f" ± {float(err):.0f}" if err is not None else ""
    elo_txt = f"{elo_f:+.1f}{err_txt}" if elo_f is not None else "…"
    band = ""
    if g.get("elo_lo") is not None and g.get("elo_hi") is not None:
        band = f" · band [{float(g['elo_lo']):+.1f}, {float(g['elo_hi']):+.1f}]"
    live = " · LIVE" if g.get("live") or status == "running" else ""
    pid = g.get("pid")
    pid_txt = f" · pid {pid}" if pid else ""
    played = g.get("played")
    play_txt = f" · {played}/{g.get('games')} games" if played else ""
    gate_row = (
        f"{status_u}  Elo {elo_txt}{band} (need {float(g.get('min_elo', need)):+.0f}) "
        f"· {path}{live}{pid_txt}{play_txt} · {g.get('when', '')}"
    )
    wdl = ""
    if g.get("wins") is not None:
        wdl = f"+{g.get('wins')} ={g.get('draws')} -{g.get('losses')}"
        if g.get("score") is not None:
            wdl += f" · score {float(g['score']):.3f}"
    val_bits = []
    if g.get("val") is not None:
        val_bits.append(f"val {g['val']}")
    if g.get("val_r") is not None:
        val_bits.append(f"r={g['val_r']}")
    if g.get("val_mae_cp") is not None:
        val_bits.append(f"mae={g['val_mae_cp']}cp")
    val_txt = " · ".join(val_bits) if val_bits else "—"
    cand = g.get("candidate") or {}
    panel = (
        "<table>"
        + "".join(
            f"<tr><th>{k}</th><td>{v}</td></tr>"
            for k, v in [
                ("Result", f"<span id='gate-result'>{status_u}</span>"),
                ("Elo estimate", f"<span id='gate-elo'>{elo_txt if elo_f is not None else '…'}</span> (95%)"),
                ("CI band", f"[{g.get('elo_lo', '—')}, {g.get('elo_hi', '—')}]"),
                ("Path", f"<span id='gate-path'>{esc(path)}</span> · EVAL_W={esc(g.get('eval_w', '?'))}"),
                ("PID", esc(pid) if pid else "—"),
                ("Match", f"{esc(wdl) if wdl else '—'} · {g.get('games')}@{g.get('nodes')}n ×{g.get('concurrency', '?')}"),
                ("Pilot fit", esc(val_txt)),
                ("Candidate", f"{esc(cand.get('sha16', '—'))} · {esc(cand.get('bytes', '—'))}B"),
                ("This gate need", f"+{float(g.get('min_elo', need)):.0f}"),
                ("Ship need (SP)", f"+{ship_need:.0f}"),
                ("Best SP gate", "+23.5 (1.4M SP) · last SP −8.7"),
                ("Gate recipe", "EPOCHS=45 · GAMES=400 · MIN_ELO=+25 · SHARD_DECAY=1.0 · MX_FORCE_GPU=1"),
                ("Epochs", g.get("epochs")),
                ("When", g.get("when", "")),
            ]
        )
        + "</table>"
        + "<p class='dim' style='margin-top:10px'>Shipping trust is self-play only. "
        "Lichess score-only nets measure teacher fit; arena Elo is the estimate that matters.</p>"
    )
    bar = (
        f"<div class='gate-bar'><div id='gate-fill' class='gate-fill' style='width:{pct}%'></div></div>"
        f"<p class='dim'>Progress toward <b>+{ship_need:.0f} Elo</b> ship bar "
        f"(current estimate <span id='gate-elo-inline'>{elo_txt if elo_f is not None else '…'}</span>).</p>"
    )
    return gate_row, panel, bar


def tmp_pipeline_panel() -> str:
    """Surface /tmp stress · smoke · nps · calibrate · lab-run tails on the board."""
    files = [
        ("nps", Path("/tmp/sable_nps.log")),
        ("smoke", Path("/tmp/sable_sp_arena_smoke.log")),
        ("stress", Path("/tmp/sable_sp_arena_stress.log")),
        ("calibrate", Path("/tmp/sable_calibrate.log")),
        ("lab_run", Path("/tmp/sable_lab_run.log")),
        ("ml_cycle", Path("/tmp/ml_cycle.log")),
    ]
    rows = []
    for label, path in files:
        if not path.is_file() or path.stat().st_size == 0:
            rows.append(
                f"<tr><th>{esc(label)}</th><td class='dim'>{esc(path.name)} · empty/missing</td></tr>"
            )
            continue
        try:
            lines = path.read_text(errors="replace").strip().splitlines()
        except OSError:
            rows.append(f"<tr><th>{esc(label)}</th><td class='dim'>unreadable</td></tr>")
            continue
        # Prefer a summary line (Elo / Nodes/second / GATE) over raw UCI spam.
        pick = ""
        for key in ("Nodes/second", "Elo ", "GATE_EPOCHS", "games ", "calibrate", "ml_cycle:", "lab run"):
            for line in reversed(lines):
                if key in line and not line.startswith("info depth"):
                    pick = line.strip()
                    break
            if pick:
                break
        if not pick:
            pick = lines[-1].strip() if lines else ""
        if len(pick) > 140:
            pick = pick[:137] + "…"
        rows.append(
            f"<tr><th>{esc(label)}</th><td><code>{esc(pick)}</code> "
            f"<span class='dim'>({len(lines)} lines)</span></td></tr>"
        )
    return (
        "<table class='mini' id='tmp-pipeline'>"
        "<caption class='dim'>Pipeline /tmp logs (stress · smoke · nps · gate)</caption>"
        + "".join(rows)
        + "</table>"
    )


def elo_tiles(g: dict | None, need: float) -> str:
    elo_raw = g.get("elo") if g else None
    elo = float(elo_raw) if elo_raw is not None else None
    err = g.get("elo_err") if g else None
    path = (g or {}).get("path", "—")
    status = (g or {}).get("status")
    elo_n = f"{elo:+.1f}" if elo is not None else ("…" if status == "running" else "—")
    err_n = f"±{float(err):.0f}" if err is not None else ("live" if status == "running" else "±?")
    return f"""
    <div class="tile"><div class="n" id="t-elo">{elo_n}</div><div class="l">Last Elo estimate</div></div>
    <div class="tile"><div class="n" id="t-err">{err_n}</div><div class="l">95% CI</div></div>
    <div class="tile"><div class="n" id="t-path">{esc(path)}</div><div class="l">Gate path</div></div>
    <div class="tile"><div class="n">+{need:.0f}</div><div class="l">This-gate need</div></div>
    <div class="tile"><div class="n">+25</div><div class="l">Ship threshold</div></div>
    <div class="tile"><div class="n">+23.5</div><div class="l">Best SP Elo</div></div>
    """


def elo_history_panel() -> str:
    path = ROOT / "web" / "elo_history.json"
    if not path.is_file():
        return "<p class='dim'>No Elo history yet.</p>"
    try:
        hist = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return "<p class='dim'>Elo history unreadable.</p>"
    if not isinstance(hist, list) or not hist:
        return "<p class='dim'>No Elo history yet.</p>"
    rows = []
    for e in reversed(hist[-12:]):
        elo = float(e.get("elo", 0))
        err = e.get("elo_err")
        err_s = f"±{float(err):.0f}" if err is not None else "±?"
        ship = "SHIP" if e.get("shipped") else "reject"
        cls = "ok" if elo >= 25 else ("warn" if elo >= 0 else "bad")
        rows.append(
            "<tr>"
            f"<td class='dim'>{esc(str(e.get('when', ''))[:19])}</td>"
            f"<td><span class='tag {esc(e.get('path', '?'))}'>{esc(e.get('path', '?'))}</span></td>"
            f"<td class='{cls}'>{elo:+.1f} {err_s}</td>"
            f"<td class='dim'>{esc(e.get('games', '?'))}@{esc(e.get('nodes', '?'))}n</td>"
            f"<td class='dim'>EVAL_W={esc(e.get('eval_w', '?'))}</td>"
            f"<td>{ship}</td>"
            "</tr>"
        )
    return (
        "<table class='mini' id='elo-hist'>"
        "<caption class='dim'>Elo estimate history (arena vs shipping · 95% CI)</caption>"
        "<thead><tr><th>when</th><th>path</th><th>Elo ± CI</th><th>match</th><th>blend</th><th></th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def health_panel(h: dict | None) -> str:
    if not h:
        return "<p class='dim'>No health snapshot.</p>"
    score = esc(h.get("score", "?"))
    rows = []
    for f in h.get("flags") or []:
        cls = "ok" if f.get("ok") else "bad"
        rows.append(
            f"<tr><th>{esc(f.get('key'))}</th>"
            f"<td class='{cls}'>{esc(f.get('msg'))}</td></tr>"
        )
    return (
        f"<p class='dim' id='health-score'>health <b>{score}</b></p>"
        f"<table id='health-table'>{''.join(rows)}</table>"
    )


def repo_panel(gt: dict) -> str:
    state = "clean · synced" if gt.get("clean") else f"{len(gt.get('changed') or [])} change(s)"
    rows = [
        ("Branch", f"<code id='git-branch'>{esc(gt.get('branch_line', '?'))}</code>"),
        ("HEAD", f"<span id='git-head'>{esc(gt.get('head', '?'))}</span>"),
        ("Author", esc(gt.get("author", "?"))),
        ("Tree", f"<span id='git-tree'>{esc(state)}</span>"),
    ]
    meta = "<table>" + "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows) + "</table>"
    files = gt.get("changed") or []
    file_rows = ""
    if files:
        file_rows = (
            "<table class='mini' style='margin-top:12px'><caption class='dim'>working tree diff</caption>"
            + "".join(
                f"<tr><th><code>{esc(c['code'])}</code></th><td><code>{esc(c['path'])}</code></td></tr>"
                for c in files[:24]
            )
            + ("<tr><td colspan='2' class='dim'>…</td></tr>" if len(files) > 24 else "")
            + "</table>"
        )
    diff = gt.get("diff_short") or gt.get("staged_short") or ""
    diff_block = f"<p class='dim' id='git-diff'>{esc(diff)}</p>" if diff else ""
    stat = gt.get("diff_stat") or ""
    stat_block = (
        f"<pre class='diff-stat' id='git-stat'>{esc(stat)}</pre>" if stat else ""
    )
    return meta + file_rows + diff_block + stat_block


def main() -> None:
    now = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    snap = collect()
    gt = snap.get("git") or {}
    branch = sh("git", "rev-parse", "--abbrev-ref", "HEAD") or "?"
    commit = gt.get("head") or sh("git", "log", "-1", "--oneline") or "?"
    dirty = not gt.get("clean", True)
    repo_html = repo_panel(gt)

    lf = snap.get("lichess") or {}
    sp_lines = snap.get("selfplay_lines", 0)
    sp_shards = snap.get("selfplay_shards", 0)
    hf_lines = lf.get("lines", 0)
    hf_shards = lf.get("shards", 0)

    g = snap.get("gate")
    need = float(snap.get("gate_need", 25))
    if g and g.get("min_elo") is not None:
        need = float(g["min_elo"])
    gate_row, gate_panel, gate_bar = gate_section(g, need)
    tiles_html = elo_tiles(g, need)
    hist_html = elo_history_panel()
    tmp_html = tmp_pipeline_panel()
    health_html = health_panel(snap.get("health"))
    wsum = snap.get("workers_summary") or {}
    counts = wsum.get("counts") or {}
    count_txt = " · ".join(f"{k} {v}" for k, v in counts.items()) if counts else "none"

    rows = [
        ("Generated", now),
        ("Engine", f"{snap.get('engine', {}).get('full_name', snap.get('engine', {}).get('name', 'Sable'))}"),
        ("Author", snap.get("engine", {}).get("author", "?")),
        ("Branch", branch),
        ("HEAD", commit),
        ("Tree", "dirty" if dirty else "clean / synced"),
        ("Self-play", f"{sp_shards} shards · {sp_lines:,} lines"),
        ("Lichess HF", f"{hf_shards} shards · {hf_lines:,} lines"),
        ("Lichess cursor", fmt(lf.get("absolute_skip")) if lf.get("absolute_skip") else "?"),
        ("Last gate", gate_row),
    ]

    proc_html = (
        f"<p class='dim' id='worker-counts'>{esc(count_txt)} · total {wsum.get('total', 0)}</p>"
        + (
            "<ul id='workers'>"
            + "".join(f"<li><code>{esc(p)}</code></li>" for p in snap["workers"])
            + "</ul>"
            if snap["workers"]
            else "<p class='dim' id='workers'>No workers.</p>"
        )
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
    --line: #2e2a26; --ink: #ece4d6;
    --amber: #ffc933; --gold: #ffe066; --gold-deep: #e6a800;
    --green: #7cb87c; --red: #d47272; --dim: #9a8f82;
    --mono: "IBM Plex Mono", "SF Mono", ui-monospace, Menlo, monospace;
    --sans: "IBM Plex Sans", system-ui, sans-serif;
  }}
  @media (prefers-color-scheme: light) {{
    :root {{ --bg: #f6f2ea; --panel: #fffdf9; --panel2: #f0ebe3;
      --line: #ddd4c8; --ink: #2a2420; --dim: #6f6459;
      --amber: #c17f00; --gold: #e6a800; --gold-deep: #a86a1c; }}
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
  .tile .n {{ font-size: 22px; font-weight: 600; color: var(--gold); line-height: 1.1;
    text-shadow: 0 0 24px rgba(255, 201, 51, 0.25); }}
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
  .bar i {{ display: block; height: 100%;
    background: linear-gradient(90deg, var(--gold-deep), var(--gold));
    box-shadow: 0 0 12px rgba(255, 201, 51, 0.35);
    transition: width .4s ease; }}
  .bar-skip i {{ background: linear-gradient(90deg, #4a7a9a, #6a8fad); box-shadow: none; }}
  .bar-keep i {{ background: linear-gradient(90deg, var(--gold-deep), var(--gold));
    box-shadow: 0 0 12px rgba(255, 201, 51, 0.35); }}
  .bar-num {{ color: var(--dim); text-align: right; font-size: 10px; }}
  .gate-bar {{ height: 11px; background: var(--line); margin: 12px 0 8px; border-radius: 1px; }}
  .gate-fill {{ height: 100%;
    background: linear-gradient(90deg, var(--gold-deep), var(--gold), var(--green));
    box-shadow: 0 0 16px rgba(255, 201, 51, 0.4);
    max-width: 100%; transition: width .4s ease; }}
  .callout {{ margin-top: 14px; padding: 12px 14px; background: var(--panel2);
    border-left: 3px solid var(--amber); }}
  .callout-gold {{ border-left-color: var(--gold); background: linear-gradient(90deg, rgba(255,201,51,.08), transparent); }}
  .callout p.trust {{ margin-top: 10px; font-size: 11px; color: var(--dim); border-top: 1px solid var(--line); padding-top: 8px; }}
  .callout-title {{ font-size: 10px; letter-spacing: .16em; text-transform: uppercase;
    color: var(--dim); margin-bottom: 8px; }}
  .callout code {{ display: block; margin: 6px 0; color: var(--ink); }}
  .foot {{ margin-top: 28px; font-size: 12px; color: var(--dim); }}
  .nav {{ display: flex; gap: 16px; font-size: 12px; }}
  .nav a {{ color: var(--amber); }}
  pre.diff-stat {{ margin-top: 10px; padding: 10px 12px; background: var(--panel2);
    border: 1px solid var(--line); font-size: 11px; line-height: 1.5; overflow-x: auto; white-space: pre-wrap; }}
  .tag {{ display: inline-block; padding: 2px 6px; border: 1px solid var(--line);
    font-size: 10px; letter-spacing: .08em; }}
  .tag.clean {{ color: var(--green); border-color: var(--green); }}
  .tag.dirty {{ color: var(--amber); border-color: var(--amber); }}
  .tag.lichess {{ color: var(--gold); border-color: var(--gold-deep); }}
  .tag.mix {{ color: #7db7ff; border-color: #4a7ab0; }}
  .tag.selfplay {{ color: var(--green); border-color: var(--green); }}
  td.ok {{ color: var(--green); }}
  td.warn {{ color: var(--amber); }}
  td.bad {{ color: var(--red); }}
  #elo-hist th {{ width: auto; font-size: 10px; letter-spacing: .08em; text-transform: uppercase; }}
  #elo-hist td {{ font-size: 12px; white-space: nowrap; }}
</style>
</head>
<body>
<header class="top">
  <h1>SABLE <span>LAB</span></h1>
  <nav class="nav">
    <a href="/">play</a>
    <a href="/daily">daily</a>
    <a href="/blog">blog</a>
    <a href="/api/status">status</a>
  </nav>
  <p class="live"><b id="pulse">●</b> live · <span id="api-ts">{esc(snap.get('generated_at', ''))}</span>
    · <span class="tag {'clean' if gt.get('clean') else 'dirty'}" id="git-tag">{'clean' if gt.get('clean') else 'dirty'}</span></p>
</header>
<main>
  <div class="tiles">
    <div class="tile"><div class="n" id="t-sp">{sp_lines:,}</div><div class="l">Self-play lines</div></div>
    <div class="tile"><div class="n" id="t-hf">{hf_lines:,}</div><div class="l">Lichess positions</div></div>
    {tiles_html}
  </div>

  <div class="grid grid-2">
    <section class="panel">
      <div class="label">Gate</div>
      {gate_bar}
      <div id="gate-panel">{gate_panel}</div>
    </section>
    <section class="panel">
      <div class="label">Repository</div>
      <div id="repo">{repo_html}</div>
    </section>
    <section class="panel">
      <div class="label">Lab snapshot</div>
      <table>{table}</table>
    </section>
  </div>

  <section class="panel" style="margin-top:16px">
    <div class="label">Lab health</div>
    <div id="health">{health_html}</div>
  </section>

  <section class="panel" style="margin-top:16px">
    <div class="label">Elo estimate</div>
    {hist_html}
  </section>

  <section class="panel" style="margin-top:16px">
    <div class="label">Pipeline /tmp</div>
    {tmp_html}
  </section>

  <section class="panel" style="margin-top:16px">
    <div class="label">Lichess trainer</div>
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
      <code>scripts/pipeline.sh run</code>
      <code>scripts/pipeline.sh stress</code>
      <code>scripts/pipeline.sh selfplay-bg</code>
      <code>scripts/pipeline.sh datagen-bg</code>
      <code>scripts/ml_cycle.sh 45 400 25</code>
      <code>scripts/prepare_lichess.sh</code>
    </section>
  </div>

  <p class="foot"><a href="/">← play chess</a> · <a href="/blog">lab notes</a> · <a href="/api/status">raw status</a></p>
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
      const te = document.getElementById('t-elo');
      if (te) te.textContent = `${{d.gate_elo >= 0 ? '+' : ''}}${{d.gate_elo.toFixed(1)}}`;
      const need = (d.gate && d.gate.min_elo) || d.gate_need || 25;
      const pct = Math.min(100, Math.max(0, Math.round(100 * d.gate_elo / 25)));
      const fill = document.getElementById('gate-fill');
      if (fill) fill.style.width = pct + '%';
      const ge = document.getElementById('gate-elo');
      if (ge) ge.textContent = `${{d.gate_elo >= 0 ? '+' : ''}}${{d.gate_elo.toFixed(1)}}`;
      const gi = document.getElementById('gate-elo-inline');
      if (gi) gi.textContent = `${{d.gate_elo >= 0 ? '+' : ''}}${{d.gate_elo.toFixed(1)}}`;
      const gr = document.getElementById('gate-result');
      if (gr && d.gate) gr.textContent = d.gate.shipped ? 'SHIPPED' : 'REJECTED';
      const terr = document.getElementById('t-err');
      if (terr && d.gate?.elo_err != null) terr.textContent = '±' + Number(d.gate.elo_err).toFixed(0);
      const tp = document.getElementById('t-path');
      if (tp && d.gate?.path) tp.textContent = d.gate.path;
      const gp = document.getElementById('gate-path');
      if (gp && d.gate) gp.textContent = `${{d.gate.path || '?'}} · EVAL_W=${{d.gate.eval_w || '?'}}`;
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
