#!/usr/bin/env python3
"""Shared lab snapshot for daily.html and web /api/status."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SP = ROOT / "data" / "selfplay"
DEFAULT_TARGET = 200_000


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


def list_sp_shards() -> list[dict]:
    shards = []
    for path in sorted(SP.glob("aug_sp_*.txt")):
        idx = shard_index(path.name)
        if idx is None:
            continue
        lines = count_lines(path)
        target = DEFAULT_TARGET
        status_path = SP / "datagen_status.json"
        if status_path.exists():
            try:
                wave = json.loads(status_path.read_text())
                if wave.get("start_index", 0) <= idx < wave.get("start_index", 0) + wave.get("n_shards", 0):
                    target = int(wave.get("positions_per_shard", DEFAULT_TARGET))
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        shards.append({
            "name": path.name,
            "index": idx,
            "lines": lines,
            "target": target,
            "pct": min(100, int(100 * lines / target)) if target else 0,
            "done": lines >= target,
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


def collect() -> dict:
    shards = list_sp_shards()
    g = gate()
    return {
        "selfplay_lines": sum(s["lines"] for s in shards),
        "selfplay_shards": len(shards),
        "shards": shards[-8:],  # latest wave tail
        "active_shards": [s for s in shards if not s["done"]][-4:],
        "gate": g,
        "gate_need": g.get("min_elo", 25) if g else 25,
        "gate_elo": g.get("elo") if g else None,
        "workers": workers(),
        "datagen": datagen_wave(),
    }
