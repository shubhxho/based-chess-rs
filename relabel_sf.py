#!/usr/bin/env python3
"""Relabel the self-play shards with Stockfish scores.

The shards are `FEN | score | result` triples, where `score` came from this
engine's own search. That is the ceiling the README talks about: a student
distilled from a 2800-rated teacher cannot pass 2800 by much, however well it
fits. This rewrites the middle field with a stronger teacher's opinion and
leaves everything else — file names, line order, the game result — alone, so
the feature cache keyed on that order stays valid and `DATA_DIR=... train.py`
picks the new labels up with no other change.

Two details are load-bearing.

*Scale.* The obvious thing is to take Stockfish's `wdl` output, which is a win
probability and therefore scale-free, and invert this project's own sigmoid to
get a centipawn label: `cp = 400 * logit(p)`. That is wrong, and measurably so.
Stockfish's win model at depth 9 is far sharper than `sigmoid(cp / 400)` — the
first three positions in the corpus come back at 94.8%, 95.9% and 87% losing,
which invert to labels past the clamp — so most of the decisive corpus collapses
onto a single saturated value and the student loses every gradation it is
supposed to learn. `score cp` is already on a conventional pawns-times-hundred
scale, which is the scale the shards and the trainer's sigmoid were written for,
so it is used directly. Mate scores become a large finite value rather than
infinity, matching what `datagen` did.

*Throughput.* Two things were costing most of the machine.

A synchronous `write; read` per position spends its time in the pipe rather than
in the search: at depth 6 Stockfish answers in 0.5 ms and the round trip costs
1.9. Commands for a whole chunk are queued by a writer thread instead, so the
engine never waits for Python and Python never waits for the engine.

And the unit of work is a slice of a shard, not a shard. One job per file on
nine workers ran at 194% CPU of a possible 900%: the pool has nothing left to
hand out once the short files are done, so eight cores sit idle waiting for
whichever shard happened to be full of hard middlegames. Slices are small enough
that there is always more work than workers.
"""

import multiprocessing as mp
import os
import select
import subprocess
import sys
import time

# `datagen` recorded nothing past 2000, treating anything beyond it as decided,
# so labels stay inside the same box the rest of the corpus lives in.
CP_CLAMP = 2000
MATE_CP = 2000

# Fixed nodes make labels reproducible across machines and let a full corpus
# use a teacher materially stronger than the old depth-8 default.  Leave this
# unset to preserve the historical depth mode.
DEPTH = os.environ.get("SF_DEPTH", "8")
NODES = os.environ.get("SF_NODES")
SF = os.environ.get("SF", "stockfish")
HASH = os.environ.get("SF_HASH", "64")
GO = b"go nodes " + NODES.encode() if NODES else b"go depth " + DEPTH.encode()
LABEL_LIMIT = f"nodes {NODES}" if NODES else f"depth {DEPTH}"
UCI_TIMEOUT = float(os.environ.get("SF_TIMEOUT", "120"))


def read_uci_line(proc, context):
    """Read one engine line with a bounded wait and useful crash diagnostics."""
    ready, _, _ = select.select([proc.stdout], [], [], UCI_TIMEOUT)
    if not ready:
        proc.kill()
        raise SystemExit(f"Stockfish timeout while {context} ({UCI_TIMEOUT:g}s)")
    raw = proc.stdout.readline()
    if not raw:
        raise SystemExit(f"Stockfish exited while {context} (status {proc.poll()})")
    return raw


def wait_for(proc, token, context):
    deadline = time.monotonic() + UCI_TIMEOUT
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            proc.kill()
            raise SystemExit(f"Stockfish timeout waiting for {token!r} while {context}")
        ready, _, _ = select.select([proc.stdout], [], [], remaining)
        if not ready:
            continue
        raw = proc.stdout.readline()
        if not raw:
            raise SystemExit(f"Stockfish exited while {context} (status {proc.poll()})")
        if raw.startswith(token):
            return raw


def relabel_chunk(job):
    """Relabel `lines[lo:hi]` of one shard and write the slice to its own file.

    Slices are reassembled in order afterwards, so a run can be interrupted and
    resumed at slice granularity rather than losing a whole shard.
    """
    src, part_path, lo, hi = job
    if os.path.exists(part_path):
        return part_path, 0, 0

    lines = []
    with open(src, "rb") as fh:
        for i, line in enumerate(fh):
            if i >= hi:
                break
            if i >= lo:
                lines.append(line)

    # Keep the original line verbatim when it is malformed, so a bad line in
    # means the same bad line out and the file stays the same length.
    parsed = []
    for line in lines:
        parts = line.split(b"|")
        if len(parts) < 3:
            parsed.append(None)
            continue
        fen = parts[0].strip()
        if len(fen.split()) < 2:
            parsed.append(None)
            continue
        parsed.append((fen, parts[2].strip()))

    todo = [i for i, p in enumerate(parsed) if p is not None]

    p = subprocess.Popen(
        [SF], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1 << 20
    )
    try:
        # UCI `go` is not pipeline-safe: engines poll stdin while searching,
        # so a later position or quit command may stop the active search.
        # Send exactly one request and consume its bestmove before the next.
        p.stdin.write(b"uci\n")
        p.stdin.flush()
        wait_for(p, b"uciok", "initialising UCI")
        p.stdin.write(
            b"setoption name Threads value 1\n"
            b"setoption name Hash value " + HASH.encode() + b"\n"
            b"setoption name MultiPV value 1\n"
            b"isready\n"
        )
        p.stdin.flush()
        wait_for(p, b"readyok", "configuring engine")

        # One exact score per valid input is a data-integrity requirement. A
        # missing/aspiration-bound score must fail the slice, never preserve a
        # stale weak label silently.
        out = []
        for i in todo:
            p.stdin.write(b"position fen " + parsed[i][0] + b"\n" + GO + b"\n")
            p.stdin.flush()
            last = None
            while True:
                raw = read_uci_line(p, f"labelling position {i}")
                if raw.startswith(b"info ") and b" score " in raw and b"bound" not in raw:
                    last = raw
                elif raw.startswith(b"bestmove"):
                    if last is None:
                        raise SystemExit(f"{src}[{lo}:{hi}]: no exact score for input line {lo + i}")
                    out.append(last)
                    break
        p.stdin.write(b"quit\n")
        p.stdin.flush()
        p.wait(timeout=UCI_TIMEOUT)
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()

    if len(out) != len(todo):
        raise SystemExit(f"{src}[{lo}:{hi}]: {len(out)} answers for {len(todo)} positions")

    written = 0
    for i, info in zip(todo, out):
        if info is None:
            continue
        tok = info.split()
        k = tok.index(b"score")
        if tok[k + 1] == b"mate":
            cp = MATE_CP if int(tok[k + 2]) > 0 else -MATE_CP
        else:
            cp = int(tok[k + 2])
        fen, result = parsed[i]
        # Stockfish scores from the side to move; the shard format is white-relative.
        if fen.split()[1] != b"w":
            cp = -cp
        cp = max(-CP_CLAMP, min(CP_CLAMP, cp))
        lines[i] = fen + b" | " + str(cp).encode() + b" | " + result + b"\n"
        written += 1

    tmp = part_path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.writelines(lines)
    os.replace(tmp, part_path)
    return part_path, written, len(lines)


def line_count(path):
    n = 0
    with open(path, "rb") as fh:
        for _ in fh:
            n += 1
    return n


def main():
    src_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    dst_dir = sys.argv[2] if len(sys.argv) > 2 else "data_sf"
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 9
    chunk = int(os.environ.get("CHUNK", "25000"))
    parts_dir = os.path.join(dst_dir, "parts")
    os.makedirs(parts_dir, exist_ok=True)

    names = sorted(f for f in os.listdir(src_dir) if f.startswith("aug") and f.endswith(".txt"))
    jobs, by_shard = [], {}
    for f in names:
        if os.path.exists(os.path.join(dst_dir, f)):
            continue                      # shard already assembled
        src = os.path.join(src_dir, f)
        n = line_count(src)
        by_shard[f] = []
        for lo in range(0, n, chunk):
            part = os.path.join(parts_dir, f"{f}.{lo:09d}")
            jobs.append((src, part, lo, min(lo + chunk, n)))
            by_shard[f].append(part)

    total_lines = sum(hi - lo for _, _, lo, hi in jobs)
    todo = sum(1 for j in jobs if not os.path.exists(j[1]))
    print(
        f"{len(names)} shards, {len(jobs)} slices ({todo} outstanding), "
        f"{total_lines} positions, {LABEL_LIMIT}, {procs} processes",
        flush=True,
    )

    done, relabelled = 0, 0
    with mp.Pool(procs) as pool:
        for part, written, _ in pool.imap_unordered(relabel_chunk, jobs, chunksize=1):
            done += 1
            relabelled += written
            if done % 10 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} slices, {relabelled} positions", flush=True)

    # Reassemble. Slice files are named by their starting line, so sorting them
    # restores the original order -- which the feature cache depends on.
    for f, parts in by_shard.items():
        out = os.path.join(dst_dir, f)
        with open(out + ".tmp", "wb") as fh:
            for p in sorted(parts):
                with open(p, "rb") as pf:
                    fh.write(pf.read())
        os.replace(out + ".tmp", out)
        for p in parts:
            os.remove(p)
        print(f"  assembled {out}", flush=True)


if __name__ == "__main__":
    main()
