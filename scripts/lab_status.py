#!/usr/bin/env python3
"""Shared lab snapshot for daily.html and web /api/status."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SP = ROOT / "data" / "selfplay"
HF = ROOT / "data" / "lichess-sf"
DEFAULT_TARGET = 200_000

ROW_FORMAT = "FEN | cp_white | result"
ROW_FORMAT_NOTE = (
    "Lichess shards use result=1 (draw placeholder, wdl=0.5). "
    "Train with EVAL_W=1 so only the Stockfish cp label matters."
)


def sh(*args: str) -> str:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def shard_index(name: str) -> int | None:
    m = re.fullmatch(r"aug_sp_(0\d{4})\.txt", name)
    return int(m.group(1)) if m else None


def count_lines(path: Path) -> int:
    try:
        with path.open("rb") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def shard_effective_lines(path: Path) -> tuple[int, int, int]:
    """Return (effective, on_disk, tmp_lines) for a shard file."""
    on_disk = count_lines(path)
    tmp = Path(str(path) + ".tmp")
    tmp_lines = count_lines(tmp) if tmp.exists() else 0
    return max(on_disk, tmp_lines), on_disk, tmp_lines


def list_sp_shards() -> list[dict]:
    wave = datagen_wave()
    wave_target = int(wave.get("positions_per_shard", DEFAULT_TARGET)) if wave else DEFAULT_TARGET
    wave_indices: set[int] = set()
    if wave and wave.get("active_indices"):
        wave_indices = set(int(x) for x in wave["active_indices"])
    names: set[str] = {p.name for p in SP.glob("aug_sp_*.txt")}
    for tmp in SP.glob("aug_sp_*.txt.tmp"):
        names.add(tmp.name[: -len(".tmp")])
    shards = []
    for name in sorted(names):
        idx = shard_index(name)
        if idx is None:
            continue
        path = SP / name
        lines, on_disk, tmp_lines = shard_effective_lines(path)
        target = wave_target if idx in wave_indices or not wave_indices else DEFAULT_TARGET
        if wave and wave.get("active_shards"):
            for ws in wave["active_shards"]:
                if ws.get("index") == idx:
                    target = int(ws.get("target", target))
                    break
        shards.append({
            "name": name,
            "index": idx,
            "lines": lines,
            "on_disk": on_disk,
            "tmp_lines": tmp_lines,
            "target": target,
            "pct": min(100, int(100 * lines / target)) if target else 0,
            "done": on_disk >= target,
            "running": tmp_lines > 0 and on_disk < target,
        })
    return shards


def probe_engine() -> dict:
    """Live UCI ident from the release binary."""
    bin_path = ROOT / "target" / "release" / "sable"
    if not bin_path.exists():
        return {"name": "Sable", "version": "?", "author": "not built", "raw": []}
    try:
        p = subprocess.Popen(
            [str(bin_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            cwd=ROOT,
        )
        assert p.stdin and p.stdout
        p.stdin.write("uci\n")
        p.stdin.flush()
        raw: list[str] = []
        for _ in range(32):
            line = p.stdout.readline().strip()
            if not line or "uciok" in line:
                break
            if line.startswith("id "):
                raw.append(line[3:])
        p.stdin.write("quit\n")
        p.stdin.flush()
        p.terminate()
        info = {"name": "Sable", "version": "?", "author": "?", "raw": raw}
        for line in raw:
            if line.startswith("name "):
                full = line[5:].strip()
                info["full_name"] = full
                parts = full.split(None, 1)
                info["name"] = parts[0]
                if len(parts) > 1:
                    info["version"] = parts[1]
            elif line.startswith("author "):
                info["author"] = line[7:].strip()
        return info
    except (OSError, subprocess.SubprocessError):
        return {"name": "Sable", "version": "?", "author": "probe failed", "raw": []}


def git_tree() -> dict:
    """Working tree snapshot for the daily board and /api/status."""
    status_sb = sh("git", "status", "-sb")
    lines = [ln for ln in status_sb.splitlines() if ln.strip()]
    branch_line = lines[0] if lines else "?"
    changed: list[dict] = []
    for line in lines[1:]:
        code = line[:2].strip() or "?"
        path = line[3:].strip()
        if path:
            changed.append({"code": code, "path": path})
    untracked = [p for p in sh("git", "ls-files", "--others", "--exclude-standard").splitlines() if p.strip()]
    for path in untracked:
        changed.append({"code": "?", "path": path})
    diff_stat = sh("git", "diff", "--stat")
    diff_short = sh("git", "diff", "--shortstat")
    staged_stat = sh("git", "diff", "--cached", "--shortstat")
    return {
        "branch_line": branch_line,
        "head": sh("git", "log", "-1", "--oneline") or "?",
        "author": sh("git", "log", "-1", "--format=%an") or "?",
        "when": sh("git", "log", "-1", "--format=%ci") or "?",
        "subject": sh("git", "log", "-1", "--format=%s") or "?",
        "short_hash": sh("git", "rev-parse", "--short", "HEAD") or "?",
        "clean": not changed,
        "changed": changed,
        "diff_stat": diff_stat,
        "diff_short": diff_short,
        "staged_short": staged_stat,
    }


def gate() -> dict | None:
    path = ROOT / "web" / "gate_last.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def file_age_s(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return max(0, int(time.time() - path.stat().st_mtime))
    except OSError:
        return None


def sha16(path: Path) -> str | None:
    if not path.is_file():
        return None
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def net_slot(name: str) -> dict | None:
    path = ROOT / name
    if not path.is_file():
        return None
    return {
        "path": name,
        "bytes": path.stat().st_size,
        "sha16": sha16(path),
        "age_s": file_age_s(path),
    }


def nets() -> dict:
    ship = net_slot("net.bin.ship")
    cur = net_slot("net.bin")
    cand = net_slot("net-candidate.bin")
    pilot = net_slot("net-lichess-pilot.bin")
    ship_match = bool(
        ship and cur and ship.get("sha16") and ship["sha16"] == cur.get("sha16")
    )
    return {
        "shipping": ship,
        "current": cur,
        "candidate": cand,
        "pilot": pilot,
        "current_is_shipping": ship_match,
        "pilot_is_candidate": bool(
            pilot and cand and pilot.get("sha16") == cand.get("sha16")
        ),
    }


def pilot() -> dict | None:
    return load_json(ROOT / "web" / "pilot_last.json")


def elo_history(limit: int = 8) -> list:
    hist = load_json(ROOT / "web" / "elo_history.json")
    if not isinstance(hist, list):
        return []
    return hist[-limit:]


def classify_worker(line: str) -> str:
    low = line.lower()
    if "prepare_hf" in low:
        return "prepare"
    if "train_gate" in low or "arena.py" in low:
        return "gate"
    if "train.py" in low or "train_lichess" in low:
        return "train"
    if "datagen" in low:
        return "datagen"
    if "ml_cycle" in low:
        return "cycle"
    if "web/server.py" in low:
        return "server"
    if "target/release/sable" in low or "/sable" in low:
        return "engine"
    return "other"


def workers() -> list[str]:
    out = sh(
        "pgrep",
        "-lf",
        "prepare_hf|train_gate|train\\.py|arena\\.py|datagen_parallel|datagen_daemon|ml_cycle|web/server\\.py|based-chess-rs/target/release/sable",
    )
    alive = []
    for line in out.splitlines():
        if "Helper" in line or "pgrep" in line:
            continue
        if any(
            k in line
            for k in (
                "prepare_hf",
                "train_gate",
                "train.py",
                "arena.py",
                "datagen",
                "ml_cycle",
                "web/server.py",
                "target/release/sable",
            )
        ):
            alive.append(line[:160])
    return alive


def worker_summary(lines: list[str]) -> dict:
    by: dict[str, list[str]] = {}
    for line in lines:
        kind = classify_worker(line)
        by.setdefault(kind, []).append(line)
    return {
        "total": len(lines),
        "counts": {k: len(v) for k, v in sorted(by.items())},
        "by_kind": by,
    }


def datagen_wave() -> dict | None:
    path = SP / "datagen_status.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if isinstance(data, dict):
        data["status_age_s"] = file_age_s(path)
    return data


def health(snap_bits: dict) -> dict:
    """Compact OK/warn flags for /api/status and the daily board."""
    flags = []
    g = snap_bits.get("gate") or {}
    lf = snap_bits.get("lichess") or {}
    wave = snap_bits.get("datagen") or {}
    nets_info = snap_bits.get("nets") or {}
    wsum = snap_bits.get("workers_summary") or {}
    counts = wsum.get("counts") or {}

    if g.get("status") == "running" or g.get("live"):
        flags.append({"ok": True, "key": "gate", "msg": f"arena live pid {g.get('pid')} · {g.get('played', 0)}/{g.get('games', '?')}"})
    elif g.get("shipped"):
        flags.append({"ok": True, "key": "gate", "msg": f"last SHIPPED {g.get('elo'):+}"})
    elif g.get("elo") is not None:
        flags.append({"ok": False, "key": "gate", "msg": f"last {str(g.get('status','reject')).upper()} {float(g['elo']):+.1f} ±{g.get('elo_err', '?')}"})
    else:
        flags.append({"ok": False, "key": "gate", "msg": "no gate yet"})

    if counts.get("datagen"):
        wp = wave.get("wave_pct", 0)
        flags.append({"ok": True, "key": "datagen", "msg": f"{counts['datagen']} workers · wave {wp}%"})
    else:
        flags.append({"ok": False, "key": "datagen", "msg": "no datagen workers"})

    phase = lf.get("phase", "idle")
    if counts.get("prepare") or phase in ("filtering", "skipping", "running"):
        flags.append({
            "ok": True,
            "key": "prepare",
            "msg": f"{phase} · kept {lf.get('kept', lf.get('emitted', 0))}/{lf.get('max_positions', '?')}",
        })
    else:
        flags.append({"ok": True, "key": "prepare", "msg": f"idle · corpus {lf.get('lines', 0):,}"})

    if nets_info.get("current_is_shipping"):
        flags.append({"ok": True, "key": "net", "msg": "net.bin == shipping"})
    else:
        flags.append({"ok": False, "key": "net", "msg": "net.bin differs from ship (candidate loaded?)"})

    gt = snap_bits.get("git") or {}
    if gt.get("clean"):
        flags.append({"ok": True, "key": "git", "msg": gt.get("head", "clean")})
    else:
        n = len(gt.get("changed") or [])
        flags.append({"ok": False, "key": "git", "msg": f"{n} local change(s) · {gt.get('head', '?')}"})

    ok_n = sum(1 for f in flags if f["ok"])
    return {
        "ok": ok_n == len(flags),
        "score": f"{ok_n}/{len(flags)}",
        "flags": flags,
    }


def lichess() -> dict:
    lines = sum(count_lines(p) for p in HF.glob("aug_hf_*.txt"))
    shards = len(list(HF.glob("aug_hf_*.txt")))
    info: dict = {
        "lines": lines,
        "shards": shards,
        "phase": "idle",
        "skip_pct": 0,
        "keep_pct": 0,
        "row_format": ROW_FORMAT,
        "row_format_note": ROW_FORMAT_NOTE,
        "result_code": 1,
        "train_hint": "EVAL_W=1",
    }
    manifest = HF / "hf_source.json"
    if manifest.exists():
        try:
            m = json.loads(manifest.read_text())
            base = int(m.get("absolute_skip_rows", m.get("skip_rows", 0)) or 0)
            scanned = int(m.get("scanned_rows", 0) or 0)
            info["absolute_skip"] = base + scanned
            info["stream_cursor"] = int(m.get("stream_row", base + scanned) or 0)
            info["max_positions"] = int(m.get("max_positions", 0) or 0)
            info["emitted"] = int(m.get("emitted_positions", 0) or 0)
            info["dropped"] = m.get("dropped")
            info["min_depth"] = m.get("min_depth")
            info["filters"] = m.get("filters")
            info["train_hint"] = m.get("train_hint", info["train_hint"])
            info["row_format"] = m.get("row_format", ROW_FORMAT)
            info["row_format_note"] = m.get("row_format_note", ROW_FORMAT_NOTE)
            info["result_code"] = m.get("result_code", 1)
            if m.get("phase") == "done" and info["phase"] == "idle":
                info["phase"] = "done"
            if scanned and info.get("emitted"):
                info["keep_rate_pct"] = round(100 * info["emitted"] / scanned, 2)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    status = HF / "prepare_status.json"
    if status.exists():
        try:
            s = json.loads(status.read_text())
            info["phase"] = s.get("phase", "idle")
            info["stream_row"] = int(s.get("stream_row", 0) or 0)
            info["skip_target"] = int(s.get("skip_target", 0) or 0)
            info["kept"] = int(s.get("kept", 0) or 0)
            info["scanned"] = int(s.get("scanned", 0) or 0)
            info["keep_rate_pct"] = s.get("keep_rate_pct")
            info["dropped"] = s.get("dropped", info.get("dropped"))
            info["max_positions"] = int(s.get("max_positions", info.get("max_positions", 0)) or 0)
            info["elapsed_s"] = int(s.get("elapsed_s", 0) or 0)
            info["when"] = s.get("when")
            info["status_age_s"] = file_age_s(status)
            if info.get("skip_target"):
                info["skip_pct"] = min(100, int(100 * info["stream_row"] / info["skip_target"]))
            if info.get("max_positions"):
                info["keep_pct"] = min(100, int(100 * info["kept"] / info["max_positions"]))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return info


def collect() -> dict:
    shards = list_sp_shards()
    g = gate()
    wave = datagen_wave()
    lf = lichess()
    wlines = workers()
    wsum = worker_summary(wlines)
    nets_info = nets()
    pilot_info = pilot()
    hist = elo_history()
    gt = git_tree()
    active = []
    if wave and wave.get("active_shards") and wave.get("phase") == "running":
        # Prefer live line counts from disk/tmp over stale status snapshot.
        live = []
        for ws in wave["active_shards"]:
            name = ws.get("name") or f"aug_sp_{int(ws.get('index', 0)):05d}.txt"
            path = SP / name
            lines, on_disk, tmp_lines = shard_effective_lines(path)
            target = int(ws.get("target", DEFAULT_TARGET))
            live.append({
                **ws,
                "name": name,
                "lines": lines,
                "on_disk": on_disk,
                "tmp_lines": tmp_lines,
                "target": target,
                "remaining": max(0, target - lines),
                "pct": min(100, int(100 * lines / target)) if target else 0,
                "done": on_disk >= target,
            })
        active = live
        wave = {
            **wave,
            "active_shards": live,
            "wave_lines": sum(s["lines"] for s in live),
            "wave_pct": min(
                100,
                int(100 * sum(s["lines"] for s in live) / max(1, int(wave.get("wave_target") or 1))),
            ),
        }
    else:
        active = [s for s in shards if s.get("running") or (not s["done"] and s["lines"] > 0)][-6:]
        if not active:
            active = [s for s in shards if not s["done"]][-4:]
    done_n = sum(1 for s in shards if s["done"])
    partial_n = sum(1 for s in shards if 0 < s["lines"] < s["target"])
    snap_bits = {
        "gate": g,
        "lichess": lf,
        "datagen": wave,
        "nets": nets_info,
        "workers_summary": wsum,
        "git": gt,
    }
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "selfplay_lines": sum(s["lines"] for s in shards),
        "selfplay_shards": len(shards),
        "selfplay_done": done_n,
        "selfplay_partial": partial_n,
        "shards": shards[-8:],
        "active_shards": active,
        "lichess": lf,
        "gate": g,
        "gate_need": g.get("min_elo", 25) if g else 25,
        "gate_elo": g.get("elo") if g else None,
        "gate_err": g.get("elo_err") if g else None,
        "gate_status": g.get("status") if g else None,
        "workers": wlines,
        "workers_summary": wsum,
        "datagen": wave,
        "nets": nets_info,
        "pilot": pilot_info,
        "elo_history": hist,
        "health": health(snap_bits),
        "git": gt,
        "engine": probe_engine(),
    }


if __name__ == "__main__":
    import pprint

    data = collect()
    if "--json" in sys.argv:
        print(json.dumps(data, indent=2))
    else:
        h = data.get("health") or {}
        print(f"Sable lab status  {data.get('generated_at')}  health {h.get('score')}")
        for f in h.get("flags") or []:
            mark = "ok" if f.get("ok") else "!!"
            print(f"  [{mark}] {f.get('key')}: {f.get('msg')}")
        print(f"  SP {data.get('selfplay_lines'):,} lines · {data.get('selfplay_done')}/{data.get('selfplay_shards')} done")
        lf = data.get("lichess") or {}
        print(f"  HF {lf.get('lines'):,} · phase {lf.get('phase')} · kept {lf.get('kept', '?')}/{lf.get('max_positions', '?')}")
        g = data.get("gate") or {}
        if g:
            print(
                f"  gate {g.get('status')} Elo {g.get('elo')} ±{g.get('elo_err')} "
                f"path={g.get('path')} pid={g.get('pid')}"
            )
        ws = data.get("workers_summary") or {}
        print(f"  workers {ws.get('total')} {ws.get('counts')}")
        n = data.get("nets") or {}
        print(f"  net ship_match={n.get('current_is_shipping')} pilot=cand {n.get('pilot_is_candidate')}")
        if "--full" in sys.argv:
            pprint.pp(data)

