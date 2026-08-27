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
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENGINE = Path(os.environ.get("ENGINE", ROOT / "target/release/sable"))
NET = ROOT / "net.bin"
SHIP_BAK = ROOT / "net.bin.ship"
CAND_BAK = ROOT / "net-candidate.bin"


def run(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def parse_arena_elo(text: str) -> tuple[float, float | None]:
    """Return (elo, ±95% err) from arena.py's final summary line."""
    elo = None
    err = None
    for line in text.splitlines():
        if line.startswith("Elo "):
            parts = line.split()
            elo = float(parts[1].replace("+", ""))
            # "Elo -1.7 +/- 34.0 (95%)"
            if len(parts) >= 4 and parts[2] == "+/-":
                try:
                    err = float(parts[3])
                except ValueError:
                    err = None
    if elo is None:
        raise SystemExit("arena produced no Elo line:\n" + text[-500:])
    return elo, err


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

    arena = run(
        [
            py,
            str(ROOT / "arena.py"),
            str(cand_bin),
            str(ship_bin),
            str(args.games),
            f"nodes {args.nodes}",
            str(args.concurrency),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    sys.stdout.write(arena.stdout)
    if arena.stderr:
        sys.stderr.write(arena.stderr)
    elo, elo_err = parse_arena_elo(arena.stdout)
    print(f"\ngate: Elo {elo:+.1f} vs shipping (threshold {args.min_elo:+.1f})", flush=True)

    data_dir = os.environ.get("REPORT_DATA_DIR", os.environ.get("DATA_DIR", "data"))
    path_tag = "lichess" if "lichess" in str(data_dir).lower() else "selfplay"
    report = {
        "when": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "elo": elo,
        "elo_err": elo_err,
        "min_elo": args.min_elo,
        "games": args.games,
        "nodes": args.nodes,
        "epochs": args.epochs,
        "path": path_tag,
        "data_dir": data_dir,
        "data_glob": os.environ.get("REPORT_DATA_GLOB", os.environ.get("DATA_GLOB", "aug*.txt")),
        "eval_w": os.environ.get("EVAL_W", "0.9"),
        "out_scale": os.environ.get("OUT_SCALE", "0.70"),
        "shipped": elo >= args.min_elo,
    }
    report_path = ROOT / "web" / "gate_last.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {report_path.relative_to(ROOT)}", flush=True)
    # Keep a short history for the daily Elo board.
    hist_path = ROOT / "web" / "elo_history.json"
    hist: list = []
    if hist_path.exists():
        try:
            hist = json.loads(hist_path.read_text())
            if not isinstance(hist, list):
                hist = []
        except (json.JSONDecodeError, OSError):
            hist = []
    hist.append(report)
    hist_path.write_text(json.dumps(hist[-24:], indent=2) + "\n")
    print(f"wrote {hist_path.relative_to(ROOT)} ({len(hist[-24:])} entries)", flush=True)

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
