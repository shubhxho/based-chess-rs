# Sable

A chess engine written in Rust. It plays UCI, evaluates positions with a small
neural network, and that network was trained on games the engine played against
itself.

It is `#![no_std]`: no allocator, no third-party crate, no libc call anywhere in
the source. Every conversation with the kernel is a hand-written `svc #0x80`
trap, and libSystem gets linked only because Mach-O insists on it for the
process entry stub. That constraint isn't a goal in itself — it just means
there's nothing between the code and the machine to wonder about when something
is slow.

```
cargo build --release
./target/release/sable
```

Then talk UCI to it, or type `bench 13`, `perft 6`, `eval`, or `d` to see the
board.

---

## The network that didn't work

The obvious way to build a small neural evaluation is the standard NNUE input —
768 binary features, one per (piece, colour, square). I built that first. It
plays **165 Elo worse** than the hand-crafted evaluator it was meant to replace.

The natural reaction is "too small, make it bigger." So I swept the hidden layer
from 16 neurons to 128 — an 8x range — and the fit against the teacher barely
moved: r hovered around 0.93 the whole way. That flatness is the actual finding.
Capacity was never the constraint.

The constraint is that piece-square features describe where pieces **are**, and
almost everything that decides a chess position is about where pieces can
**go**. A knight on d5 is worth wildly different amounts depending on what it
attacks. A rook is worth a lot more on an open file. Neither fact is recoverable
from a one-hot square index, no matter how wide the layer behind it.

So the budget went into the input instead of the hidden layer. Alongside the 768
piece-square planes there are now 166 rows encoding mobility, passed pawns by
rank, isolated and doubled pawns, rooks on open and half-open files, the bishop
pair, king attackers, and king shelter — all computed from the board and looked
up in the same embedding table. Each row costs 32 bytes.

That was the whole difference:

| Input set | Size | r vs teacher | RMSE |
|---|---|---|---|
| Hand-crafted evaluation (baseline) | — | 0.937 | 192 cp |
| 768 piece-square features | 24.6 KB | 0.955 | 161 cp |
| 934 features, with mobility and structure | 29.8 KB | **0.970** | **130 cp** |

Same 32 neurons. Same optimiser. Same data. Five kilobytes of extra input beat
four times the hidden width.

Nor was it data-starved: retraining on 3.36M positions instead of 2M moves the
fit by nothing (r 0.970 → 0.968). Width didn't matter, data didn't matter, the
input set was the whole thing.

---

## How it's trained

The teacher is the engine's own alpha-beta search. `datagen` self-plays out of
randomised openings and labels every quiet position with the score a fixed-node
search returned, plus how the game eventually ended. The student is a static
evaluation that never searches — the same idea behind DeepMind's searchless
grandmaster-level chess, at a size that fits in L1 cache instead of a TPU pod.

Positions are thrown away when the side to move is in check or the best move is
a capture. In those positions the tactic decides the game, not the static
evaluation, and training on them just teaches the network to imitate search —
which it has no way to do.

```bash
./sable datagen 400000 5000 $SEED > data/shard.txt   # self-play; engine teaches
python train.py 2000000 14                            # MLX; writes net.bin
cargo build --release                                 # net.bin is include_bytes!'d
```

Two decisions in the trainer are worth explaining.

**The trainer never computes features.** It asks the engine for them, through a
`featdump` command that writes the active indices for each position. Two
implementations of one feature map is a bug that produces a network which loads
fine, runs fine, and is quietly wrong — the worst kind to track down. One
implementation, in `net.rs`, is the source of truth for both.

**Quantisation is part of the objective, not a step at the end.** Weights get
projected back into the int8 box after every optimiser step, so what ships is
the function the trainer actually converged to rather than a rounded-off
approximation of it. To be sure of that, `net.bin` is replayed through an
independent NumPy reference that reproduces `net.rs` operation by operation.
They agree on every position tested — the only disagreement I ever saw turned
out to be Python's floor division against Rust's truncation on negative scores,
which was a bug in the reference.

---

## What's inside

| Layer | Implementation |
|---|---|
| Board | Bitboards, 12 piece planes plus a mailbox, incremental Zobrist |
| Attacks | Magic bitboards; the magics are *searched* at startup, so they validate themselves |
| Movegen | Fully legal — pins, check evasions and en-passant discovery all resolved during generation |
| Search | Fail-soft PVS with TT, null move, ProbCut, LMR, singular extensions, SEE and history pruning, static-eval correction history |
| Eval | 934 -> 32 -> 1, int8, eight output buckets by material, NEON inference |
| I/O | Raw `read` / `write` / `poll` / `mmap`, hand-rolled integer formatting |

### Talking to the kernel

`src/sys.rs` is the entire OS dependency, and it is short:

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

`read`, `write`, `poll`, `mmap`, `munmap`, `exit`. That's the complete list. The
transposition table is a raw `mmap` region. `poll` on fd 0 is how `stop` gets
noticed mid-search without the search ever blocking on input.

The clock doesn't even trap: it reads the arm64 generic timer registers
(`cntvct_el0` / `cntfrq_el0`) directly. Monotonic, immune to an NTP step landing
in the middle of a search, and cheap enough to poll constantly.

### Search

```
iterative deepening
  └ aspiration window, widened on each fail
      └ negamax (PVS)
          ├ transposition cutoff
          ├ static eval, corrected by the search's own past residuals
          ├ whole-node pruning: reverse futility, razoring, null move, ProbCut
          ├ move loop
          │   ├ per-move pruning: late-move, futility, SEE, history
          │   ├ extensions: check, singular (with multi-cut)
          │   └ late-move reduction + re-search ladder
          └ quiescence at the horizon
```

Ordering is what makes the pruning safe, so it gets as much care as the pruning
rules: TT move first, then captures classified by static exchange evaluation,
killers, the counter-move, and finally quiets ranked by butterfly history plus
two plies of continuation history. Every history table uses gravity updates, so
it still responds to new information after millions of increments.

### The search corrects its own evaluation

A static evaluation is wrong in *patterns*, not at random: the same pawn
structure fools it the same way every time it appears. So the search records the
residual. Whenever a node returns a score the static evaluation did not predict —
and the bound actually proves the disagreement, and no capture decided it — the
difference is blended into three tables, indexed by pawn structure, by the
non-pawn piece layout, and by the move that led to the node. Later nodes add the
remembered residual back before pruning on it.

Nothing is stored: the correction is re-derived on every probe, and the
transposition table keeps the *uncorrected* evaluation, so a stale correction
cannot outlive the table that produced it. The combined nudge is capped at 72
centipawns — it is a correction, not a second evaluator.

Time management follows the same instinct. The soft budget stretches while the
root move is still changing and shrinks once it has held for several iterations,
and stretches again when the score is falling — a score in motion is a score
worth more time.

One trap worth writing down. The 2.4 MB continuation-history table lives in a
static of its own rather than as a field of the searcher. A global only lands in
BSS when its *entire* initialiser is zero, and the searcher has non-zero fields —
so as a field, all 2.4 MB of zeros get written into the executable. Moving it
out keeps them there — the same trick applies to the correction tables.

---

## Does it work?

Move generation is checked against `python-chess`, because I didn't trust my own
memory of perft constants — and I was right not to. Half the "known" values I
first wrote down were wrong, and the oracle is what told me the engine was fine
and my test data wasn't.

| Suite | Result |
|---|---|
| Classic perft (startpos, kiwipete, positions 3–6) | 6/6 exact, to depth 7 |
| Oracle-verified edge cases (castling, ep pins, promotion races) | 20/20 exact |
| Randomised positions, depth 4 | 119/119 exact |
| Rust NEON inference vs NumPy reference | 80/80 identical |
| Insufficient material evaluates to exactly 0 | K vs K, K+N vs K |

`bench 13` is bit-identical run to run, which makes it a proper refactoring
guard. When I removed a redundant legality check on the transposition move, the
node count stayed at exactly 1,321,821 — proof the change was a pure speedup and
not a silent behaviour change.

Throughput is around 2.7 Mnps on a single M-series core.

Matches are run at a fixed node count rather than a fixed time, so results don't
shift with machine load, and colours are swapped on every opening pair. Every
result below is 20,000 nodes per move, self-play from randomised openings:

| Opponent | Games | Score | Elo |
|---|---|---|---|
| 1.1 (before correction history and the time-management rework) | 800 | 0.546 | **+32 ±24** |
| 1.1, at 100,000 nodes per move | 300 | 0.533 | +23 ±39 |
| the same build without ProbCut and killer hygiene | 2400 | 0.508 | +6 ±14 |
| `sable-std` (1.0 network build) | 200 | 0.578 | +54 ±49 |
| `sable-net` (first network release) | 200 | 0.573 | +51 ±49 |
| `sable-hce` (same search, no network) | 200 | 0.635 | +96 ±50 |

The search change also pays for itself in nodes: `bench 12` reaches the same
depth on 677k nodes where 1.1 needed 854k, a 21% reduction.

One reduction rule did not survive this process. Reducing late quiet moves
harder when the static evaluation sat well below alpha cut the bench node count
by a further 10% — and lost the 800-game match that tested it. Fewer nodes is
not the same thing as more strength, which is the entire reason the matches get
run.

```bash
cargo test --release     # 18 unit tests, ~0.2s
bash tests/run_all.sh    # the above plus the python-chess cross-checks
```

`cargo test` works because `no_std` and `no_main` are conditional on not
building tests — a `no_main` binary has nowhere to put a test harness. The
shipped binary is unaffected; a test build is a different binary entirely.

The unit tests cover the invariants that are cheap to state and miserable to
debug once broken: perft counts, that unmake restores every field rather than
just the hash, that the incremental Zobrist key matches a fresh parse, that
magic attacks agree with a naive ray walk, SEE on known exchanges, and that
mirroring a position leaves its features and its evaluation unchanged.

That last one is worth a note. It first compared the *scores* of hand-mirrored
FEN constants — and failed, because one of my constants had a pawn on the wrong
side. It now mirrors programmatically and compares the feature multisets before
the scores, over 160+ positions, so it cannot be wrong about its own data and a
failure says which layer broke.

To compare two evaluations you need two binaries. The network is embedded at
compile time, so "no network" is just a build with a header the loader rejects:

```bash
cargo build --release && cp target/release/sable sable-std      # with the network

cp net.bin /tmp/net.keep
printf '\0\0\0\0\0\0\0\0' > net.bin                       # invalid header
cargo build --release && cp target/release/sable sable-hce      # falls back to
cp /tmp/net.keep net.bin && cargo build --release               # hand-crafted eval

python arena.py ./sable-std ./sable-hce 400 "nodes 20000" 9
```

---

## Layout

```
src/sys.rs       raw syscalls, SyncCell
src/bb.rs        bitboards, magic generation
src/pos.rs       position, make/unmake, Zobrist
src/movegen.rs   legal move generation
src/eval.rs      hand-crafted evaluation (baseline and fallback)
src/net.rs       features + quantised inference — source of truth for both
src/search.rs    PVS, pruning, ordering, time management
src/tt.rs        mmap-backed transposition table
src/datagen.rs   self-play data generation
src/uci.rs       protocol, perft, bench, featdump
train.py         MLX trainer and quantised exporter
arena.py         match runner with Elo confidence intervals
publish_hf.py    uploads the network to the Hugging Face Hub
src/tests.rs     unit tests (cargo test)
tests/           perft suites, the python-chess oracle, inference verification
.github/         CI: fmt, clippy -D warnings, tests, perft, size budget
```

The network is on the Hub at
[`shubhxho/sable-chess-net`](https://huggingface.co/shubhxho/sable-chess-net),
with the format documented well enough to read it without this engine.

## Honest limitations

- It is distilled from itself. With no external engine available, the ceiling is
  the teacher's own search quality, not a stronger reference.
- Accumulators are refreshed in full rather than updated incrementally, which
  costs roughly 15% nps. At 32 neurons a whole matrix row is four NEON
  registers, so the bookkeeping isn't obviously worth it — but it's the first
  thing I'd try next.
- Single-threaded. The `Threads` UCI option is accepted and ignored.
- No opening book, no endgame tablebases.
