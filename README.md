# Sable

A UCI chess engine written entirely in Rust, with a distilled neural evaluation
trained in MLX on Apple silicon.

The engine is `#![no_std]`. There is no allocator, no third-party crate, and no
libc call anywhere in the source — every kernel interaction is a hand-written
`svc #0x80` trap. libSystem is linked only because Mach-O requires it for the
process entry stub.

The whole thing, network included, is a **248 KB binary** — 6% of the 4 MB
budget it was built to fit in.

```
cargo build --release
./target/release/sable
```

---

## What's in it

| Layer | Implementation |
|---|---|
| Board | Bitboards, 12 piece planes + mailbox, incremental Zobrist |
| Attacks | Magic bitboards, magics *searched* at startup and self-validated |
| Movegen | Fully legal — pins, check evasions and en-passant discovery resolved during generation |
| Search | Fail-soft PVS, TT, null move, LMR, singular extensions, SEE pruning |
| Eval | Tapered hand-crafted terms **plus** a 25 KB distilled network |
| I/O | Raw `read`/`write`/`poll`/`mmap` syscalls, hand-rolled integer formatting |

### Kernel interface

`src/sys.rs` is the entire OS dependency:

```rust
asm!(
    "svc #0x80",
    "cset {err}, cs",
    err = out(reg) err,
    inlateout("x0") a0 => ret,
    in("x1") a1, in("x2") a2, in("x16") n,
    options(nostack)
);
```

`read`, `write`, `poll`, `mmap`, `munmap`, `exit`. That is the complete list.
The transposition table is a raw `mmap` region, and `poll` on fd 0 is how `stop`
is noticed mid-search without ever blocking the search.

The clock does not even trap: it reads the arm64 generic timer registers
(`cntvct_el0` / `cntfrq_el0`) directly, which is monotonic, immune to NTP steps
mid-search, and free enough that the search can poll it constantly.

### Search

```
iterative deepening
  └ aspiration window, widened on each fail
      └ negamax (PVS)
          ├ transposition cutoff
          ├ whole-node pruning: reverse futility, razoring, null move
          ├ move loop
          │   ├ per-move pruning: late-move, futility, SEE
          │   ├ extensions: check, singular (with multi-cut)
          │   └ late-move reduction + re-search ladder
          └ quiescence at the horizon
```

Move ordering drives everything else: TT move, then SEE-classified captures,
killers, counter-move, then quiets ranked by butterfly history plus two plies of
continuation history. All history tables use gravity updates so they stay
responsive after millions of increments.

One detail worth calling out — the 2.4 MB continuation-history table is a static
of its own rather than a field of the searcher. A global only lands in BSS when
its *entire* initialiser is zero, and the searcher has non-zero fields. Splitting
it out is the difference between a 2.6 MB binary and a 215 KB one.

---

## The network

`768 → 32 → 1`, both perspectives sharing one weight matrix, int8 weights,
clipped ReLU, and 8 output layers selected by remaining material —
**25,196 bytes** total. Inference is written against ARM NEON intrinsics
directly (`vmovl_s8`, `vmlal_s16`, `vaddvq_s32`).

It is **additive**: the network predicts a correction to the hand-crafted
evaluation rather than replacing it.

That choice was measured, not assumed:

| Setup | Size | vs hand-crafted baseline |
|---|---|---|
| Network **replaces** hand-crafted eval | 24.1 KB | **−165 ± 69 Elo** (200 games) |
| Network **corrects** hand-crafted eval | 24.1 KB | **+55 ± 31 Elo** (500 games) |
| ... with material-bucketed output | 24.6 KB | **+57 ± 28 Elo** (600 games) |

A network of this size over plain piece-square features simply cannot represent
mobility or king safety — those depend on where pieces can *go*, not where they
are. A replacement network throws that knowledge away and lacks the capacity to
rediscover it. Predicting only the residual keeps everything the hand-crafted
terms already know and spends the whole parameter budget on what they miss.

Fit against the teacher, measured on the *quantised* network:

| Predictor | r | MAE | RMSE |
|---|---|---|---|
| hand-crafted alone | 0.9369 | 96.3 cp | 191.6 cp |
| + network, single output | 0.9551 | 93.5 cp | 167.3 cp |
| + network, 8 output buckets | 0.9553 | 90.3 cp | **161.3 cp** |

Hidden width was swept from 16 to 128 neurons and the fit plateaus at every
width — the bottleneck is the feature set, not capacity. That is why the last
480 bytes went into output buckets rather than a wider hidden layer.

All matches run at a fixed 20,000 nodes per move, so results do not depend on
machine load, with colours swapped on every opening pair.

### Distillation

The teacher is the engine's own search. `datagen` self-plays from randomised
openings, and every quiet position is labelled with the score a fixed-node search
returned, plus the eventual game result. The student is a static evaluation that
never searches — the same idea DeepMind used for searchless grandmaster-level
play, at a scale that fits in L1 instead of a TPU pod.

```
./sable datagen 400000 5000 <seed> > shard.txt     # self-play, engine is teacher
python train.py 4000000 12                          # MLX, exports net.bin
cargo build --release                               # net.bin is include_bytes!'d
```

Quantisation is part of the objective, not a post-processing step: weights are
projected back into the int8 box after every optimiser step, so the exported
network computes exactly the function the trainer converged to. This is verified
— `net.bin` round-trips through an independent NumPy reference that reproduces
`net.rs` operation for operation, and the two agree on every position tested.

---

## Verification

Move generation is checked against `python-chess` as an independent oracle, not
against remembered constants:

| Suite | Result |
|---|---|
| Classic perft (startpos, kiwipete, positions 3–6) | 6/6 exact, to depth 7 |
| Oracle-verified edge cases (castling, ep pins, promotion races) | 20/20 exact |
| Randomised fuzz, depth 4 | 119/119 exact |
| Rust NEON inference vs NumPy reference | 60/60 identical |
| Insufficient-material positions evaluate to exactly 0 | K vs K, K+N vs K |

```
perft 6                       # from any position
bench 13                      # fixed node count, deterministic
python arena.py A B 400 "nodes 20000" 8
```

`bench 13` is bit-identical across runs, which is what makes it usable as a
refactoring guard: removing the redundant transposition-move legality check left
the node count at exactly 1,321,821, proving the change was a pure speedup and
not a behaviour change.

Throughput is ~2.6 Mnps single-threaded on an M-series core; the network costs
about 15% of that.

---

## Layout

```
src/sys.rs       raw syscalls, SyncCell
src/bb.rs        bitboards, magic generation
src/pos.rs       position, make/unmake, Zobrist
src/movegen.rs   legal move generation
src/eval.rs      hand-crafted evaluation
src/net.rs       quantised network inference (NEON)
src/search.rs    PVS, pruning, ordering, time management
src/tt.rs        mmap-backed transposition table
src/datagen.rs   self-play data generation
src/uci.rs       protocol, perft, bench
train.py         MLX trainer and quantised exporter
arena.py         head-to-head match runner with Elo confidence intervals
publish_hf.py    uploads the network and trainer to the Hugging Face Hub
```

The network is on the Hub at
[`shubhxho/sable-chess-net`](https://huggingface.co/shubhxho/sable-chess-net),
with the blob format documented so it can be read without this engine.
