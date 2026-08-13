#!/usr/bin/env python3
"""Head-to-head UCI match runner.

Plays two engine binaries against each other from randomised openings, with
colours swapped on every pair so an opening advantage cannot bias the result,
and reports the score with a likelihood-ratio confidence interval on Elo.

Usage: arena.py ENGINE_A ENGINE_B [games] ["nodes 20000"|"movetime 100"] [concurrency]
"""

import math
import os
import multiprocessing as mp
import random
import subprocess
import sys

import chess


class Engine:
    def __init__(self, path):
        self.p = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.send("uci")
        self.wait("uciok")
        self.send("setoption name Hash value 16")
        self.isready()

    def send(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def wait(self, token):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine died")
            if line.startswith(token):
                return line.strip()

    def isready(self):
        self.send("isready")
        self.wait("readyok")

    def bestmove(self, board, limit):
        """`limit` is either `movetime N` or `nodes N`. Node limits make the
        result independent of how loaded the machine is, which matters when
        the comparison itself is the point."""
        self.send("position fen " + board.fen())
        self.send("go " + limit)
        line = self.wait("bestmove")
        return line.split()[1]

    def quit(self):
        try:
            self.send("quit")
            self.p.wait(timeout=2)
        except Exception:
            self.p.kill()


def random_opening(rng, plies=8):
    """A shallow random opening, rejected if it is already lopsided in
    material -- the point is variety, not chaos."""
    while True:
        b = chess.Board()
        for _ in range(plies):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(rng.choice(moves))
        if b.is_game_over():
            continue
        vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
        bal = sum(
            vals.get(p.piece_type, 0) * (1 if p.color else -1)
            for p in b.piece_map().values()
        )
        if abs(bal) <= 2:
            return b


def play(args):
    """One opening played twice, once with each engine as white."""
    path_a, path_b, seed, limit = args
    rng = random.Random(seed)
    start = random_opening(rng)
    ea, eb = Engine(path_a), Engine(path_b)
    results = []
    try:
        for swap in (False, True):
            board = start.copy()
            white, black = (eb, ea) if swap else (ea, eb)
            while not board.is_game_over(claim_draw=True) and board.fullmove_number < 200:
                eng = white if board.turn == chess.WHITE else black
                mv = eng.bestmove(board, limit)
                if mv == "0000":
                    break
                try:
                    board.push_uci(mv)
                except ValueError:
                    # An illegal move loses the game outright.
                    board = None
                    results.append(0.0 if (eng is ea) else 1.0)
                    break
            if board is None:
                continue
            r = board.result(claim_draw=True)
            if r == "1-0":
                results.append(0.0 if swap else 1.0)
            elif r == "0-1":
                results.append(1.0 if swap else 0.0)
            else:
                results.append(0.5)
    finally:
        ea.quit()
        eb.quit()
    return results


def elo(score, n):
    if n == 0 or score <= 0:
        return float("-inf"), 0.0
    if score >= n:
        return float("inf"), 0.0
    p = score / n
    e = -400 * math.log10(1 / p - 1)
    # Standard error on the score rate, propagated through the Elo curve.
    var = p * (1 - p) / n
    se = 400 / math.log(10) * math.sqrt(var) / (p * (1 - p))
    return e, 1.96 * se


def main():
    a, b = sys.argv[1], sys.argv[2]
    pairs = int(sys.argv[3]) // 2 if len(sys.argv) > 3 else 100
    limit = sys.argv[4] if len(sys.argv) > 4 else "movetime 100"
    conc = int(sys.argv[5]) if len(sys.argv) > 5 else 8

    # Openings are seeded per pair. The base is settable so a result can be
    # confirmed against a genuinely different set of openings rather than a
    # replay of the same ones -- a single 1000-game match is one sample.
    base = int(os.environ.get("SEED_BASE", "9000"))
    jobs = [(a, b, base + i, limit) for i in range(pairs)]
    score, n, w, d, l = 0.0, 0, 0, 0, 0
    with mp.Pool(conc) as pool:
        for res in pool.imap_unordered(play, jobs):
            for r in res:
                score += r
                n += 1
                w += r == 1.0
                d += r == 0.5
                l += r == 0.0
            if n % 20 == 0:
                e, err = elo(score, n)
                print(f"  {n} games  +{w} ={d} -{l}  Elo {e:+.0f} +/- {err:.0f}", flush=True)
    e, err = elo(score, n)
    print(f"\n{a} vs {b}")
    print(f"games {n}  +{w} ={d} -{l}  score {score/n:.3f}")
    print(f"Elo {e:+.1f} +/- {err:.1f} (95%)")


if __name__ == "__main__":
    main()
