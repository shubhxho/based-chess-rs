import os, chess, random, subprocess
def perft(b,d):
    if d==0: return 1
    if d==1: return b.legal_moves.count()
    n=0
    for m in b.legal_moves:
        b.push(m); n+=perft(b,d-1); b.pop()
    return n
rng=random.Random(4242); fens=[]
for _ in range(120):
    b=chess.Board()
    for _ in range(rng.randint(4,60)):
        ms=list(b.legal_moves)
        if not ms: break
        b.push(rng.choice(ms))
    if b.is_game_over(): continue
    fens.append(b.fen())
D=4
inp="".join(f"position fen {f}\ngo perft {D}\n" for f in fens)+"quit\n"
out=subprocess.run([os.environ.get("ENGINE","./target/release/sable")],input=inp,capture_output=True,text=True).stdout
mine=[int(l.split()[1]) for l in out.splitlines() if l.startswith("nodes ")]
bad=0
for f,g in zip(fens,mine):
    w=perft(chess.Board(f),D)
    if g!=w:
        bad+=1; print("FAIL",f,"got",g,"want",w)
print(f"fuzz: {len(mine)} random positions at depth {D}, {bad} mismatches")
