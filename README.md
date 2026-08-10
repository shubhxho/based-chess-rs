# Sable

A UCI chess engine written entirely in Rust, with a distilled neural evaluation
trained in MLX on Apple silicon.

The engine is `#![no_std]`. There is no allocator, no third-party crate, and no
libc call anywhere in the source — every kernel interaction is a hand-written
`svc #0x80` trap. libSystem is linked only because Mach-O requires it for the
process entry stub.

The whole thing, network included, is a **215 KB binary**.

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
| Eval | Tapered hand-crafted terms **plus** a 24 KB distilled network |
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

`read`, `write`, `poll`, `mmap`, `munmap`, `gettimeofday`, `exit`. That is the
complete list. The transposition table is a raw `mmap` region; `poll` on fd 0 is
how `stop` is noticed mid-search without ever blocking the search.

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
clipped ReLU, **24,716 bytes**. Inference is written against ARM NEON intrinsics
directly (`vmovl_s8`, `vmlal_s16`, `vaddvq_s32`).

It is **additive**: the network predicts a correction to the hand-crafted
evaluation rather than replacing it.

That choice was measured, not assumed. A replacement network of this size scores
**−165 ± 69 Elo** against the hand-crafted evaluation — 24 KB over plain
piece-square features simply cannot represent mobility or king safety, which
depend on where pieces can *go*, not where they are. Asking the same network for
only the residual keeps everything the hand-crafted terms already know and spends
the entire parameter budget on what they miss.

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

```
perft 6                       # from any position
bench 13                      # fixed node count, deterministic
python arena.py A B 400 "nodes 20000" 8
```

`bench 13` searches 1,321,821 nodes — bit-identical across runs, which is what
makes it usable as a refactoring guard.

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
```
