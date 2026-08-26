#!/usr/bin/env python3
"""Stream high-quality labelled chess positions from Hugging Face into Sable shards.

Default source: Lichess/chess-position-evaluations (CC0-1.0).  It contains
Stockfish evaluations, depth, node count, FENs, and a PV.  Unlike raw human PGN
corpora, every retained position already has a position-level teacher label.
The PV lets this converter discard tactical best moves, matching the static-net
filter in src/datagen.rs.  Rows are streamed; the 42 GB source is not downloaded
up front.

A safe pilot:
    uv pip install -r requirements-training.txt
    python prepare_hf.py data/lichess-sf --max-positions 200000
    # later runs:
    python prepare_hf.py data/lichess-sf --max-positions 300000 --resume
    DATA_DIR=data/lichess-sf DATA_GLOB='aug_hf_*.txt' EVAL_W=1 python train.py 0 20

For a full run omit --max-positions, but materialise a bounded, deduplicated
corpus first: train.py holds all FENs and feature arrays in memory.  Pin the
source revision in --revision and retain hf_source.json.  A model is only a
candidate until it passes opening-balanced arena matches and calibration.
"""

import argparse
import atexit
import fcntl
import hashlib
import json
import os
import signal
from pathlib import Path

DEFAULT_DATASET = "Lichess/chess-position-evaluations"
DEFAULT_REVISION = "abb8f0b1251f89295a35b5ac801cb08a873812de"


def first_pv_move(line):
    """Return the first UCI move in a Lichess analysis PV, if any."""
    if not isinstance(line, str):
        return None
    move = line.split(maxsplit=1)[0]
    return move if 4 <= len(move) <= 5 else None


def canonical_fen(fen):
    """Validate and canonicalise a standard chess FEN to six fields."""
    import chess

    try:
        board = chess.Board(fen)
    except ValueError:
        # Malformed or empty FENs show up in the stream; the caller already
        # counts a None return as a fen drop rather than aborting the run.
        return None, None
    if board.is_valid() and board.chess960 is False:
        return board, board.fen(en_passant="fen")
    return None, None


def sample_key(fen, seed):
    """Stable key for deterministic sampling independent of stream ordering."""
    return int.from_bytes(hashlib.blake2b((seed + "\0" + fen).encode(), digest_size=8).digest(), "big")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out_dir", help="directory for aug_hf_*.txt shards")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--revision", default=DEFAULT_REVISION, help="HF commit SHA; use main only deliberately")
    p.add_argument("--split", default="train")
    p.add_argument("--min-depth", type=int, default=18)
    p.add_argument("--min-knodes", type=int, default=0)
    p.add_argument("--max-cp", type=int, default=2000)
    p.add_argument("--max-positions", type=int, default=10_000_000, help="0 means no cap")
    p.add_argument("--skip-rows", type=int, default=0, help="restart offset in the streamed source")
    p.add_argument(
        "--resume",
        action="store_true",
        help="set --skip-rows from hf_source.json (absolute_skip_rows + scanned_rows)",
    )
    p.add_argument("--shard-size", type=int, default=250_000)
    p.add_argument("--dedupe", choices=("none", "memory"), default="memory")
    p.add_argument("--sample-mod", type=int, default=1, help="keep rows whose stable hash mod N is zero")
    p.add_argument("--seed", default="sable-lichess-sf-v1")
    p.add_argument(
        "--allow-restart",
        action="store_true",
        help="permit --skip-rows 0 when aug_hf_*.txt already exist (usually a mistake)",
    )
    args = p.parse_args()
    if args.min_depth < 1 or args.max_cp < 1 or args.shard_size < 1 or args.sample_mod < 1:
        p.error("limits must be positive")

    try:
        from datasets import load_dataset
        import chess  # noqa: F401 -- import now gives a useful dependency error
    except ImportError as exc:
        raise SystemExit("install requirements-training.txt before preparing data") from exc

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("aug_hf_*.txt"))
    manifest_path = out_dir / "hf_source.json"

    if args.resume:
        if not manifest_path.exists():
            raise SystemExit(f"--resume needs {manifest_path}")
        prev = json.loads(manifest_path.read_text())
        base = int(prev.get("absolute_skip_rows", prev.get("skip_rows", 0)) or 0)
        scanned = int(prev.get("scanned_rows", 0) or 0)
        args.skip_rows = base + scanned
        print(f"resume: skip-rows={args.skip_rows:,} (base {base:,} + scanned {scanned:,})", flush=True)

    if existing and args.skip_rows <= 0 and not args.allow_restart:
        raise SystemExit(
            f"{len(existing)} shards already in {out_dir}; refusing --skip-rows 0 "
            f"(re-streaming from the start mostly emits duplicates). "
            f"Use --resume, or pass an explicit --skip-rows, or --allow-restart."
        )
    # Concurrent prepares race on shard numbers and rewrite identical FENs into
    # parallel files. An exclusive lock makes the second process fail fast.
    lock_path = out_dir / ".prepare_hf.lock"
    lock_fh = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit(f"another prepare_hf is already writing {out_dir}") from exc
    lock_fh.seek(0)
    lock_fh.truncate()
    lock_fh.write(f"pid {os.getpid()}\n")
    lock_fh.flush()

    def release_lock():
        try:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            lock_fh.close()
        except Exception:
            pass
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass

    atexit.register(release_lock)
    # SIGKILL cannot be caught (the "zsh: killed" case); TERM/INT still clear
    # the lock so the next prepare is not blocked by a dead pid.
    def _die(signum, _frame):
        release_lock()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _die)
    signal.signal(signal.SIGINT, _die)

    # The manifest makes the otherwise enormous dataset selection reproducible.
    manifest = vars(args).copy()
    manifest.update({
        "license": "CC0-1.0 (verify the current dataset card before redistribution)",
        "score_pov": "white",
        "result": "neutral dummy draw; train score-only data with EVAL_W=1",
        "filters": "valid standard FEN, no mate, quiet non-promotion PV move, not in check",
        "dedupe_rule": "first retained canonical FEN wins in stream order",
    })
    (out_dir / "hf_source.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    stream = load_dataset(args.dataset, revision=args.revision, split=args.split, streaming=True)
    # Skipping in the stream API avoids materialising millions of discarded
    # rows in the Python loop (which is what made long --skip-rows runs look
    # wedged and then get OOM-killed on the way out).
    if args.skip_rows:
        print(f"skipping {args.skip_rows:,} source rows via stream.skip …", flush=True)
        stream = stream.skip(args.skip_rows)
        print("skip complete; filtering positions", flush=True)
    seen = set() if args.dedupe == "memory" else None
    shard_no, rows_in_shard, kept, scanned = 0, 0, 0, 0
    dropped = {"depth": 0, "score": 0, "fen": 0, "tactical": 0, "duplicate": 0, "sample": 0}
    out = None
    final = None

    def open_shard():
        nonlocal shard_no, rows_in_shard, out, final
        candidate = out_dir / f"aug_hf_{shard_no:05d}.txt"
        while candidate.exists() or Path(str(candidate) + ".tmp").exists():
            shard_no += 1
            candidate = out_dir / f"aug_hf_{shard_no:05d}.txt"
        out = open(str(candidate) + ".tmp", "wb")
        rows_in_shard = 0
        final = candidate
        return candidate

    open_shard()

    def emit(fen, cp):
        nonlocal shard_no, rows_in_shard, kept, out
        # cp is White POV, precisely the convention read_labels expects on disk.
        out.write(fen.encode() + b" | " + str(cp).encode() + b" | 1\n")
        kept += 1
        rows_in_shard += 1
        if rows_in_shard >= args.shard_size:
            out.close()
            os.replace(str(final) + ".tmp", final)
            shard_no += 1
            open_shard()

    try:
        for row in stream:
            scanned += 1
            if args.max_positions and kept >= args.max_positions:
                break
            depth, knodes, cp, mate = row.get("depth"), row.get("knodes"), row.get("cp"), row.get("mate")
            if depth is None or int(depth) < args.min_depth or (args.min_knodes and int(knodes or 0) < args.min_knodes):
                dropped["depth"] += 1
                continue
            if cp is None or mate is not None or abs(int(cp)) >= args.max_cp:
                dropped["score"] += 1
                continue
            board, fen = canonical_fen(row.get("fen", ""))
            if board is None or board.is_check():
                dropped["fen"] += 1
                continue
            pv = first_pv_move(row.get("line"))
            try:
                move = board.parse_uci(pv) if pv else None
            except ValueError:
                move = None
            if move is None or board.is_capture(move) or move.promotion:
                dropped["tactical"] += 1
                continue
            if sample_key(fen, args.seed) % args.sample_mod:
                dropped["sample"] += 1
                continue
            if seen is not None:
                if fen in seen:
                    dropped["duplicate"] += 1
                    continue
                seen.add(fen)
            emit(fen, int(cp))
            if kept and kept % 100_000 == 0:
                print(f"rows {scanned:,}; kept {kept:,}; dropped {dropped}", flush=True)
    finally:
        if out is not None and not out.closed:
            out.close()
        if final is not None:
            tmp = Path(str(final) + ".tmp")
            if tmp.exists():
                if rows_in_shard:
                    os.replace(tmp, final)
                else:
                    tmp.unlink()
        # Drop the FEN set and stream before process teardown. Keeping millions of
        # strings alive into interpreter shutdown is what triggers the post-write
        # "zsh: killed" OOM after an otherwise successful run.
        if seen is not None:
            seen.clear()
        del stream
        release_lock()

    manifest.update({
        "scanned_rows": scanned,
        "emitted_positions": kept,
        "dropped": dropped,
        "absolute_skip_rows": args.skip_rows,
    })
    (out_dir / "hf_source.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {kept:,} positions from {scanned:,} streamed rows; dropped {dropped}", flush=True)
    # Hard-exit after a successful write: atexit GC of HF/dataset internals has
    # been OOM-killing this process even though every shard is already on disk.
    release_lock()
    os._exit(0)


if __name__ == "__main__":
    main()
