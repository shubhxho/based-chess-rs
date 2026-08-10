import os, chess, sys, subprocess, random

FENS = [
 "8/8/8/8/8/8/6k1/4K2R w K - 0 1",
 "8/8/8/8/8/8/1k6/R3K3 w Q - 0 1",
 "4k3/8/8/8/8/8/8/4K2R b K - 0 1",
 "8/2p5/8/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
 "8/8/1k6/2b5/2pP4/8/5K2/8 b - d3 0 1",
 "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
 "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
 "8/P1k5/K7/8/8/8/8/8 w - - 0 1",
 "K1k5/8/P7/8/8/8/8/8 w - - 0 1",
 "8/k1P5/8/1K6/8/8/8/8 w - - 0 1",
 "8/8/2k5/5q2/5n2/8/5K2/8 b - - 0 1",
 "3k4/3p4/8/K1P4r/8/8/8/8 b - - 0 1",
 "5k2/8/8/8/8/8/8/4K2R w K - 0 1",
 "r3k2r/1b4bq/8/8/8/8/7B/R3K2R w KQkq - 0 1",
 "r3k2r/8/3Q4/8/8/5q2/8/R3K2R b KQkq - 0 1",
 "2K2r2/4P3/8/8/8/8/8/3k4 w - - 0 1",
 "8/8/1P2K3/8/2n5/1q6/8/5k2 b - - 0 1",
 "4k3/1P6/8/8/8/8/K7/8 w - - 0 1",
 "8/P1k5/K7/8/8/8/8/8 w - - 0 1",
 "8/8/8/8/1k6/8/K1p5/8 b - - 0 1",
]
DEPTH = 5

def perft(board, d):
    if d == 0: return 1
    if d == 1: return board.legal_moves.count()
    n = 0
    for m in board.legal_moves:
        board.push(m); n += perft(board, d-1); board.pop()
    return n

cmds = []
for f in FENS:
    cmds.append(f"position fen {f}\ngo perft {DEPTH}\n")
inp = "".join(cmds) + "quit\n"
out = subprocess.run([os.environ.get("ENGINE","./target/release/sable")], input=inp, capture_output=True, text=True).stdout
mine = [int(l.split()[1]) for l in out.splitlines() if l.startswith("nodes ")]

bad = 0
for f, got in zip(FENS, mine):
    want = perft(chess.Board(f), DEPTH)
    ok = "PASS" if got == want else "FAIL"
    if got != want: bad += 1
    print(f"{ok} d{DEPTH} got={got:<12} want={want:<12} {f}")
print("failures:", bad)
