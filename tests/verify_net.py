import sys, subprocess, struct, random, numpy as np, chess
sys.path.insert(0, ".")
import train as T

b=open('net.bin','rb').read()
magic,nin,h,nb=struct.unpack('<IIII',b[:16])
assert (nin,h,nb)==(T.IN,T.H,T.BUCKETS), (nin,h,nb)
o=16
ftq=np.frombuffer(b[o:o+nin*h],np.int8).reshape(nin,h); o+=nin*h
fbq=np.frombuffer(b[o:o+2*h],np.int16); o+=2*h
oq=np.frombuffer(b[o:o+nb*2*h],np.int8).reshape(nb,2*h); o+=nb*2*h
obq=np.frombuffer(b[o:o+4*nb],np.int32)

rng=random.Random(11); fens=[]
while len(fens)<80:
    bd=chess.Board()
    for _ in range(rng.randint(4,60)):
        ms=list(bd.legal_moves)
        if not ms: break
        bd.push(rng.choice(ms))
    if bd.is_game_over() or len(bd.piece_map())<5: continue
    fens.append(bd.fen())

# engine evals
inp="".join(f"position fen {f}\neval\n" for f in fens)+"quit\n"
out=subprocess.run(["./target/release/sable"],input=inp,capture_output=True,text=True).stdout
rust=[int(l.split()[1]) for l in out.splitlines() if l.startswith("eval ")]

# engine features for the same FENs
fin=("featdump\n"+"\n".join(fens)+"\n").encode()
raw=subprocess.run(["./target/release/sable"],input=fin,capture_output=True).stdout
feats=np.frombuffer(raw,dtype='<u2').reshape(-1,T.REC)

bad=0
for i,(f,r) in enumerate(zip(fens,rust)):
    us=feats[i,:T.MAX_F].astype(np.int32); them=feats[i,T.MAX_F:2*T.MAX_F].astype(np.int32)
    bk=int(feats[i,2*T.MAX_F])
    py=T.quantised_eval(ftq,fbq,oq,obq,us,them,bk)
    if py!=r:
        bad+=1
        if bad<=5: print("MISMATCH",f,"rust",r,"py",py)
print(f"{len(rust)} positions compared, {bad} mismatches")
