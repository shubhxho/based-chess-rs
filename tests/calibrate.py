#!/usr/bin/env python3
"""Place an engine on Stockfish's UCI_Elo scale.

Stockfish with UCI_LimitStrength plays to a nominal rating. Playing our engine
against a few of those settings and finding where the score crosses 50% gives an
absolute number that is comparable to something outside this repo -- which no
match between two Sable binaries can provide.

The number is only as good as Stockfish's own calibration, which is approximate
and was fitted at longer time controls than this uses. It is an anchor, not a
measurement.

Usage: calib.py ENGINE "movetime 100" GAMES ELO [ELO ...]
"""
import math
import multiprocessing as mp
import random
import subprocess
import sys

import chess


class Engine:
    def __init__(self, path, options=None):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, bufsize=1)
        self.send("uci")
        self.wait("uciok")
        for name, value in (options or {}).items():
            self.send(f"setoption name {name} value {value}")
        self.send("setoption name Hash value 16")
        self.send("isready")
        self.wait("readyok")

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

    def bestmove(self, board, limit):
        self.send("position fen " + board.fen())
        self.send("go " + limit)
        return self.wait("bestmove").split()[1]

    def quit(self):
        try:
            self.send("quit")
            self.p.wait(timeout=2)
        except Exception:
            self.p.kill()


def random_opening(rng, plies=8):
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
        bal = sum(vals.get(p.piece_type, 0) * (1 if p.color else -1)
                  for p in b.piece_map().values())
        if abs(bal) <= 2:
            return b


def play(args):
    ours, elo, seed, limit = args
    rng = random.Random(seed)
    start = random_opening(rng)
    ea = Engine(ours)
    eb = Engine("stockfish", {"UCI_LimitStrength": "true", "UCI_Elo": str(elo)})
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
                    results.append(0.0 if eng is ea else 1.0)
                    board = None
                    break
            if board is None:
                continue
            r = board.result(claim_draw=True)
            results.append(0.5 if r == "1/2-1/2" else
                           (1.0 if (r == "1-0") != swap else 0.0))
    finally:
        ea.quit()
        eb.quit()
    return results


def fit_rating(points):
    """One rating that best explains every match at once.

    Reading a rating off a single setting throws away the other matches and puts
    all the weight on whichever one happened to land nearest parity. Instead fit
    the single R that maximises the likelihood of all the observed scores under
    the logistic model, treating each game as a Bernoulli trial with a draw
    counted as half.

    The likelihood is unimodal in R, so a bisection on its derivative is enough
    and does not need a dependency to do it.
    """
    def dlogL(R):
        # d/dR of sum n_i * [p_i log q_i + (1-p_i) log (1-q_i)], q_i the model's
        # expected score. The constant factor ln(10)/400 is dropped: only the
        # root matters.
        return sum(n * (p - 1.0 / (1.0 + 10 ** ((E - R) / 400.0)))
                   for E, p, n in points)

    lo, hi = 0.0, 4000.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if dlogL(mid) > 0:
            lo = mid
        else:
            hi = mid
    R = (lo + hi) / 2
    # Standard error from the observed Fisher information: the curvature of the
    # same log-likelihood at its peak.
    c = math.log(10) / 400.0
    info = sum(n * c * c * (q := 1.0 / (1.0 + 10 ** ((E - R) / 400.0))) * (1 - q)
               for E, _, n in points)
    return R, (1.96 / math.sqrt(info) if info > 0 else float("inf"))


def main():
    ours, limit, games = sys.argv[1], sys.argv[2], int(sys.argv[3])
    points = []
    for elo in [int(x) for x in sys.argv[4:]]:
        jobs = [(ours, elo, 5000 + i, limit) for i in range(games // 2)]
        score = n = 0
        with mp.Pool(6) as pool:
            for res in pool.imap_unordered(play, jobs):
                for r in res:
                    score += r
                    n += 1
        p = score / n
        points.append((elo, p, n))
        if 0 < p < 1:
            diff = -400 * math.log10(1 / p - 1)
            se = 400 / math.log(10) * math.sqrt(p * (1 - p) / n) / (p * (1 - p))
            print(f"SF UCI_Elo {elo}: {n} games score {p:.3f}  "
                  f"implied {elo + diff:+.0f} ({diff:+.0f} vs SF) +/- {1.96*se:.0f}", flush=True)
        else:
            # A shutout carries no information about *how much* stronger, only
            # that it is. Keep it in the fit, where it still constrains R.
            print(f"SF UCI_Elo {elo}: {n} games score {p:.3f}  (shutout)", flush=True)

    if len(points) > 1:
        R, ci = fit_rating(points)
        print(f"\nmaximum-likelihood fit over {sum(n for _, _, n in points)} games: "
              f"{R:.0f} +/- {ci:.0f}", flush=True)
        print("residuals (observed score - model's expectation):", flush=True)
        for E, p, n in points:
            q = 1.0 / (1.0 + 10 ** ((E - R) / 400.0))
            print(f"  vs {E}: observed {p:.3f}  expected {q:.3f}  "
                  f"{'+' if p >= q else ''}{p - q:.3f}", flush=True)
        print("\nA systematic drift in those residuals means Stockfish's own scale "
              "and this engine's disagree about what an Elo is, and the single "
              "number above is hiding it.", flush=True)


if __name__ == "__main__":
    main()
