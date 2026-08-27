#!/usr/bin/env python3
"""Train a candidate network, then ship it only if it beats the current binary.

The last Lichess-mix and tiny self-play nets both *lost* hundreds of Elo to the
shipping weights. Fit loss is not a release criterion. This gate:

  1. Backs up the shipping net.bin
  2. Runs train.py with the caller's DATA_DIR / EVAL_W / OUT_SCALE / epochs
  3. Rebuilds the release binary around the candidate
  4. Arenas candidate vs the backed-up shipping binary at fixed nodes
  5. Restores shipping unless the measured Elo gain clears --min-elo

Example:
    DATA_DIR=data/mix DATA_GLOB='aug*.txt' EVAL_W=1 OUT_SCALE=0.70 \\
      .venv/bin/python train_gate.py --epochs 15 --games 200 --min-elo 10
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENGINE = Path(os.environ.get("ENGINE", ROOT / "target/release/sable"))
NET = ROOT / "net.bin"
SHIP_BAK = ROOT / "net.bin.ship"
CAND_BAK = ROOT / "net-candidate.bin"
PILOT_BAK = ROOT / "net-lichess-pilot.bin"
PILOT_META = ROOT / "web" / "pilot_last.json"
REPORT_PATH = ROOT / "web" / "gate_last.json"
HIST_PATH = ROOT / "web" / "elo_history.json"


def run(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def sha16(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def net_info(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return {"path": path.name, "bytes": path.stat().st_size, "sha16": sha16(path)}


def load_pilot_meta() -> dict:
    if not PILOT_META.is_file():
        return {}
    try:
        data = json.loads(PILOT_META.read_text())
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def write_gate(report: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {REPORT_PATH.relative_to(ROOT)}", flush=True)


def append_history(report: dict) -> None:
    hist: list = []
    if HIST_PATH.exists():
        try:
            hist = json.loads(HIST_PATH.read_text())
            if not isinstance(hist, list):
                hist = []
        except (json.JSONDecodeError, OSError):
            hist = []
    # Drop prior unfinished row for this pid if any.
    pid = report.get("pid")
    if pid is not None:
        hist = [h for h in hist if not (h.get("status") == "running" and h.get("pid") == pid)]
    if report.get("status") != "running":
        hist.append({k: v for k, v in report.items() if k != "live"})
    HIST_PATH.write_text(json.dumps(hist[-24:], indent=2) + "\n")
    print(f"wrote {HIST_PATH.relative_to(ROOT)} ({len(hist[-24:])} entries)", flush=True)


def parse_arena_elo(text: str) -> tuple[float, float | None]:
    """Return (elo, ±95% err) from arena.py's final summary line."""
    elo = None
    err = None
    for line in text.splitlines():
        if line.startswith("Elo "):
            parts = line.split()
            elo = float(parts[1].replace("+", ""))
            if len(parts) >= 4 and parts[2] == "+/-":
                try:
                    err = float(parts[3])
                except ValueError:
                    err = None
    if elo is None:
        raise SystemExit("arena produced no Elo line:\n" + text[-500:])
    return elo, err


def parse_score_line(text: str) -> dict:
    """Parse 'games N  +W =D -L  score S' from arena summary."""
    out: dict = {}
    m = re.search(
        r"games\s+(\d+)\s+\+(\d+)\s+=(\d+)\s+-(\d+)\s+score\s+([0-9.]+)",
        text,
    )
    if m:
        out["played"] = int(m.group(1))
        out["wins"] = int(m.group(2))
        out["draws"] = int(m.group(3))
        out["losses"] = int(m.group(4))
        out["score"] = float(m.group(5))
    return out


def parse_progress_line(line: str) -> dict | None:
    # "  40 games  +12 =10 -18  Elo -35 +/- 68"
    m = re.match(
        r"\s*(\d+)\s+games\s+\+(\d+)\s+=(\d+)\s+-(\d+)\s+Elo\s+([+-]?\d+(?:\.\d+)?)\s+\+/-\s+(\d+(?:\.\d+)?)",
        line,
    )
    if not m:
        return None
    return {
        "played": int(m.group(1)),
        "wins": int(m.group(2)),
        "draws": int(m.group(3)),
        "losses": int(m.group(4)),
        "elo": float(m.group(5)),
        "elo_err": float(m.group(6)),
    }


def base_report(args, *, status: str) -> dict:
    data_dir = os.environ.get("REPORT_DATA_DIR", os.environ.get("DATA_DIR", "data"))
    path_tag = "lichess" if "lichess" in str(data_dir).lower() else "selfplay"
    pilot = load_pilot_meta()
    report = {
        "when": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": status,
        "pid": os.getpid(),
        "elo": None,
        "elo_err": None,
        "elo_lo": None,
        "elo_hi": None,
        "min_elo": args.min_elo,
        "games": args.games,
        "nodes": args.nodes,
        "epochs": args.epochs,
        "concurrency": args.concurrency,
        "path": path_tag,
        "data_dir": data_dir,
        "data_glob": os.environ.get("REPORT_DATA_GLOB", os.environ.get("DATA_GLOB", "aug*.txt")),
        "eval_w": os.environ.get("EVAL_W", "0.9"),
        "out_scale": os.environ.get("OUT_SCALE", "0.70"),
        "shipped": False,
        "skip_train": bool(args.skip_train),
        "candidate": net_info(CAND_BAK),
        "shipping": net_info(SHIP_BAK),
        "pilot": net_info(PILOT_BAK) if path_tag == "lichess" else None,
        "val": pilot.get("val"),
        "val_r": pilot.get("r"),
        "val_mae_cp": pilot.get("mae_cp"),
        "train_epochs_done": pilot.get("epochs_done"),
        "played": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "score": None,
    }
    return report


def finalize_elo(report: dict, elo: float, elo_err: float | None) -> None:
    report["elo"] = elo
    report["elo_err"] = elo_err
    if elo_err is not None:
        report["elo_lo"] = round(elo - elo_err, 1)
        report["elo_hi"] = round(elo + elo_err, 1)
    report["shipped"] = elo >= float(report["min_elo"])
    report["status"] = "shipped" if report["shipped"] else "rejected"
    report["when"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")


def run_arena_streaming(args, cand_bin: Path, ship_bin: Path, report: dict) -> str:
    """Run arena, stream progress, refresh gate_last.json live."""
    py = sys.executable
    cmd = [
        py,
        str(ROOT / "arena.py"),
        str(cand_bin),
        str(ship_bin),
        str(args.games),
        f"nodes {args.nodes}",
        str(args.concurrency),
    ]
    print("+", " ".join(cmd), flush=True)
    write_gate(report)
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    chunks: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        chunks.append(line)
        prog = parse_progress_line(line.rstrip("\n"))
        if prog:
            report["played"] = prog["played"]
            report["wins"] = prog["wins"]
            report["draws"] = prog["draws"]
            report["losses"] = prog["losses"]
            report["elo"] = prog["elo"]
            report["elo_err"] = prog["elo_err"]
            report["elo_lo"] = round(prog["elo"] - prog["elo_err"], 1)
            report["elo_hi"] = round(prog["elo"] + prog["elo_err"], 1)
            report["live"] = True
            report["when"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
            write_gate(report)
    rc = proc.wait()
    text = "".join(chunks)
    if rc != 0:
        raise SystemExit(f"arena exited {rc}\n{text[-500:]}")
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--nodes", type=int, default=20_000)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--min-elo", type=float, default=25.0,
                   help="ship only if arena Elo >= this (default 25; short matches are noisy below ~20)")
    p.add_argument("--skip-train", action="store_true", help="arena an already-written net.bin")
    args = p.parse_args()

    py = sys.executable
    if not ENGINE.exists():
        run(["cargo", "build", "--release"], cwd=ROOT)

    if not args.skip_train:
        # A crashed prior gate can leave net.bin as a rejected candidate while
        # net.bin.ship still holds the real shipping weights. Never overwrite
        # that backup with whatever happens to sit in net.bin now.
        if SHIP_BAK.exists():
            shutil.copy2(SHIP_BAK, NET)
            print(f"restored shipping net from {SHIP_BAK}", flush=True)
        else:
            shutil.copy2(NET, SHIP_BAK)
            print(f"backed up shipping net -> {SHIP_BAK}", flush=True)
        env = os.environ.copy()
        env.setdefault("OUT_SCALE", "0.70")
        env.setdefault("ENGINE", str(ENGINE))
        run([py, str(ROOT / "train.py"), "0", str(args.epochs)], cwd=ROOT, env=env)
        shutil.copy2(NET, CAND_BAK)
        print(f"candidate saved -> {CAND_BAK}", flush=True)
    else:
        if not SHIP_BAK.exists():
            raise SystemExit(f"--skip-train needs an existing {SHIP_BAK}")
        # After a reject, net.bin is shipping again while net-candidate.bin still
        # holds the trainee. Copying NET→CAND would wipe that trainee and arena
        # shipping against itself.
        if CAND_BAK.exists():
            shutil.copy2(CAND_BAK, NET)
            print(f"reloaded candidate from {CAND_BAK}", flush=True)
        else:
            shutil.copy2(NET, CAND_BAK)
            print(f"no prior candidate; treating current net.bin as candidate", flush=True)

    # Candidate binary (current net.bin) vs shipping binary (restored weights).
    run(["cargo", "build", "--release"], cwd=ROOT)
    cand_bin = Path("/tmp/sable-candidate")
    ship_bin = Path("/tmp/sable-shipping")
    shutil.copy2(ENGINE, cand_bin)
    shutil.copy2(SHIP_BAK, NET)
    run(["cargo", "build", "--release"], cwd=ROOT)
    shutil.copy2(ENGINE, ship_bin)
    # Put the candidate back so a successful gate leaves it ready to commit.
    shutil.copy2(CAND_BAK, NET)

    report = base_report(args, status="running")
    report["live"] = True
    print(f"gate pid={report['pid']} path={report['path']} games={args.games}", flush=True)

    text = run_arena_streaming(args, cand_bin, ship_bin, report)
    elo, elo_err = parse_arena_elo(text)
    report.update(parse_score_line(text))
    report.pop("live", None)
    finalize_elo(report, elo, elo_err)
    print(
        f"\ngate: Elo {elo:+.1f}"
        + (f" ± {elo_err:.0f}" if elo_err is not None else "")
        + f" vs shipping (threshold {args.min_elo:+.1f}) [{report['status']}]",
        flush=True,
    )
    write_gate(report)
    append_history(report)

    if elo >= args.min_elo:
        shutil.copy2(CAND_BAK, NET)
        shutil.copy2(CAND_BAK, SHIP_BAK)
        run(["cargo", "build", "--release"], cwd=ROOT)
        print("SHIPPED candidate net.bin — rebuild done. Commit only after a second opening set.", flush=True)
        return 0

    shutil.copy2(SHIP_BAK, NET)
    run(["cargo", "build", "--release"], cwd=ROOT)
    print("REJECTED candidate — shipping net restored.", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
