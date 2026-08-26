#!/usr/bin/env python3
"""Shared lab snapshot for daily.html and web /api/status."""

from __future__ import annotations

import json
import re
import subprocess
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
    shards = []
    for path in sorted(SP.glob("aug_sp_*.txt")):
        idx = shard_index(path.name)
        if idx is None:
            continue
        lines, on_disk, tmp_lines = shard_effective_lines(path)
        target = wave_target if idx in wave_indices or not wave_indices else DEFAULT_TARGET
        if wave and wave.get("active_shards"):
            for ws in wave["active_shards"]:
                if ws.get("index") == idx:
                    target = int(ws.get("target", target))
                    break
        shards.append({
            "name": path.name,
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


def workers() -> list[str]:
    out = sh("pgrep", "-lf", "prepare_hf|train_gate|datagen_parallel|datagen_daemon|ml_cycle|release/sable")
    alive = []
    for line in out.splitlines():
        if "Helper" in line or "pgrep" in line:
            continue
        if any(k in line for k in ("prepare_hf", "train_gate", "datagen", "ml_cycle", "release/sable")):
            alive.append(line[:140])
    return alive


def gate() -> dict | None:
    path = ROOT / "web" / "gate_last.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def datagen_wave() -> dict | None:
    path = SP / "datagen_status.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


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
    active = []
    if wave and wave.get("active_shards") and wave.get("phase") == "running":
        active = wave["active_shards"]
    else:
        active = [s for s in shards if s.get("running") or (not s["done"] and s["lines"] > 0)][-6:]
        if not active:
            active = [s for s in shards if not s["done"]][-4:]
    done_n = sum(1 for s in shards if s["done"])
    partial_n = sum(1 for s in shards if 0 < s["lines"] < s["target"])
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "selfplay_lines": sum(s["lines"] for s in shards),
        "selfplay_shards": len(shards),
        "selfplay_done": done_n,
        "selfplay_partial": partial_n,
        "shards": shards[-8:],
        "active_shards": active,
        "lichess": lichess(),
        "gate": g,
        "gate_need": g.get("min_elo", 25) if g else 25,
        "gate_elo": g.get("elo") if g else None,
        "workers": workers(),
        "datagen": wave,
    }
