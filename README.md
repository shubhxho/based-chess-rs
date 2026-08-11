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

The one thing that did move it later was the *teacher*. Relabelling the whole
set with a search about 30 Elo stronger, at 6,000 nodes a move instead of 5,000
and with repeated positions thrown away, gave a network that beats the one it
replaced by +23.5 ± 24.1 Elo over 800 games, and by +23.0 ± 21.6 over a further
1000 — the same margin twice. The student can only be as good as what it is
shown.

Doing it a second time bought nothing. Four million more positions, labelled by
that better network and the search around it, trained a net that *lost* to its
own teacher — by 20 ± 24 Elo over 800 games and 24 ± 22 over another 1000.
Mixing old and new shards came out at +4 ± 24, and adding bucket-balanced
sample weights to that at +5 ± 22.

What finally got something out of the second round was noticing that the two
rounds *overlap*. The engine drops duplicate positions inside one generation
run, but two runs rediscover the same openings, and in a mixed set the duplicate
carries the older, weaker teacher's label. Deduplicating across shards and
keeping the newer label on the overlap is worth +12 ± 22 and +12 ± 20 over two
runs — about +12 across 2200 games, and that is the network that ships now.

Weighting the older shards down on top of that, on the theory that the newer
teacher knows more, loses 24 ± 22. The old positions are worth as much as the
new ones; it is only their stale *labels* that aren't.

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
    inlateout("x1") a1 => _, inlateout("x2") a2 => _,
    inlateout("x16") n => _, lateout("x17") _,
    options(nostack)
);
```

`read`, `write`, `poll`, `mmap`, `munmap`, `exit`. That's the complete list. The
transposition table is a raw `mmap` region. `poll` on fd 0 is how `stop` gets
noticed mid-search without the search ever blocking on input.

Every argument register is an output, and that detail cost a real bug. The
obvious way to write this is `in("x1") a1`, which reads correctly — x1 is an
input — but it promises the compiler that x1 still holds the same value when
the kernel returns. Darwin returns a second value in x1 and traps through x16,
so the promise is false, and the compiler is free to keep a live variable in a
register the kernel is about to overwrite.

It did. `Tt::resize` computed a byte count, called `mmap`, and stored the byte
count into the table — except the byte count was living in x1 across the call,
so what got stored was the kernel's leftover zero. A table with the right
cluster count and a length of zero works perfectly on every probe and store,
and silently does nothing in `clear()`, which memsets `bytes` bytes.

The visible symptom was a table that never cleared: after any `setoption name
Hash`, `ucinewgame` and the `Clear Hash` button both became no-ops, so every
game inherited the previous game's entries. The startup `resize` escaped it only
because there is no mapping to unmap yet, and the different branch happened to
put the byte count somewhere else. `bench 13` run twice in one process is the
shortest way to see it: 1,399,778 nodes then 543,535.

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

Behind the "static eval" line sits a 512 KB direct-mapped cache keyed on the
Zobrist key. The network is a pure function of the position, and a search asks
about the same position repeatedly — transpositions, the re-search after a
fail-high, null-move verification, a node a later iteration walks into again —
so most of the feature extraction was rediscovering a number computed
microseconds earlier. It buys about 3% at `bench 13` and returns bit-identical
node counts, which is the useful property: a cache that changed the search would
be a cache that was wrong.

A slot is one `u64`: `tag:40 | generation:8 | score:16`. Scores clamp to
±20,000 so sixteen bits hold one exactly, which leaves the other forty-eight to
spend on being sure — a forty-bit tag makes a wrong answer from a key collision
256 times rarer than a thirty-two-bit one would. The generation byte is how the
table empties: bumping a counter invalidates everything at once, where the first
version memset half a megabyte. That memset was big enough to decide the sizing
— 18- and 20-bit tables measured *slower* than 16-bit purely because of it.
Making the clear free and re-running the sweep, 16 bits still wins (461 ms
against 463, 470 and 478), so the answer was right for the wrong reason and is
now right for the right one: past 512 KB the table stops fitting the caches that
make it worth having.

What survives the cache runs through a tighter accumulator. Thirty-two hidden
neurons are four NEON registers, so both perspectives now stay in the register
file for the whole walk over the feature list — the earlier version called an
`acc += row` helper per feature and reloaded and stored the accumulator around
each one, eighty round trips to memory per evaluation for arithmetic that never
had to leave. Cache and accumulator together: 492 ms to 459 ms on `bench 13`,
best of nine runs each, same 1,399,778 nodes.

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

## Things that didn't work

Recorded because a negative result nobody writes down gets re-attempted.

**How any of this is measured.** The machine drifts: over one afternoon the
same unmodified binary produced medians of 495, 476, 455 and 428 ms. Comparing
two medians taken minutes apart therefore measures the weather. `tests/paircmp.sh`
runs the two binaries adjacently and keeps the *difference* per pair, alternating
which one goes first, so whatever the machine is doing lands on both halves and
cancels. Run against itself it reports a median delta of 0 ms with an interval of
±3 ms over 21 pairs — that is the noise floor, and anything smaller than it was
never a result. Node count is the correctness check throughout: 1,399,778 at
`bench 13` and 9,623,933 at `bench 18`. A change that moves either one is a
change to the search, whatever else it claims to be.

**Deferring the losing-capture test into move selection.** Quiescence scores
every noisy move up front, which runs a full swap evaluation on each, and then
usually leaves after two or three of them. Carrying the winning-band score as an
optimistic placeholder and resolving it only when a move wins a scan is exactly
equivalent — the score is an upper bound on the truth, resolution only lowers
it, and the scan keeps the earliest maximum, so the same move comes out, ties
included. Node counts confirmed it to the digit at two depths. It is also not
faster: resolving in place forces the scan to restart, and that O(n) rescan
costs about what the skipped evaluation saves. 428 against 430 ms over 41 pairs.

**Scoring quiescence's captures at quiescence's own threshold.** Move ordering
sorts a capture above the quiet moves when its swap value clears -20, and then
quiescence, which will not search anything below 0, asks a second time with the
tighter number. Scoring at 0 in the first place should collapse the two: the
moves that differ are exactly those the second test was going to throw away, and
dropping them earlier cannot change which moves get searched or in what order.
The argument survives contact with the awkward cases too — the tt move and queen
promotions are scored by what they are rather than what they win, so they keep
their own test, and a quiescence node that is *in check* searches losing
captures, so it keeps the old threshold.

It still moves the node count: 1,400,014 against 1,399,778 at depth 13, and
9,326,301 against 9,623,933 at depth 18. Fewer nodes, which is the direction
that flatters it, and no explanation for either number in the argument above.
Something in the premise is false — most likely that `see_ge` is exactly
monotone in its threshold, given the pruning it does internally on the way to an
answer. A change that moves the tree is a change to how the engine plays,
measurable only by playing games, and it was sold as a free speedup. Reverted
unmeasured.

**Two smaller versions of the same lesson.** Splitting `features_both` on a
const generic so the hot path carries no `n < MAX_F` compare — the bound is
provably `2·pieces + 12 ≤ 76` for any legal position, so those branches really
are dead — read +1 ms over 25 pairs and then **-5 ms over 41**, faster in 16 of
them. Not noise in the end: the second monomorphisation adds 16 KB, and the
instruction cache charges more for it than the branches ever cost. The other way
to reach the same place is to size the buffer past the 140 features a 64-piece
FEN could produce and drop the checks with no duplication, but `MAX_F` is
written into the featdump header and mirrored in `train.py`, so that grows every
training record by half for a saving already shown to be under the noise floor.
The guards stay. Merging the piece-square and mobility walks, which scan the
same four bitboards twice, measured +9 ms over 25 pairs and then -2 ms over 41.
That first number is why the interval is printed.

**Deferred quiet scoring.** Two thirds of the quiet moves a node generates are
never picked — the node cuts, or late-move pruning takes the tail — and each one
costs three history lookups scattered across four megabytes. Leaving them
unscored until selection first reaches the quiets cut the tree 13% (1,399,778
nodes to 1,220,249 at `bench 13`) and ran 18% faster in wall clock.

It also lost: **−9.6 ± 24.1 Elo over 800 games** at 20k nodes. Not significant,
but centred the wrong way, and the whole point was to gain. The reason it moves
the search at all is worth keeping in mind — scoring quiets late reads history
that the node's own earlier children have already updated, so it is a different
ordering rather than the same one arrived at more cheaply. Two implementations
of it: the first also broke ties differently, because selection swaps as it
picks and a swap-then-reconsider version compares a permuted tail. Quiet scores
tie constantly at zero, so that alone changed the tree.

**Deduplicating attack generation** (see the commit; kept, but worth zero).

**Every eval-cache size from 12 to 22 bits.** 16 wins, but 12 through 15 are
inside the noise and 18 through 22 lose slowly to their own footprint. There is
no cliff to find here.

**Making the eval cache associative — and the measurement that killed it.**
Instrumenting the direct-mapped table looked like an argument for associativity:
at `bench 13` it takes 973,613 probes, 43% of which land on a slot already
holding a live entry for a different position. That is a lot of apparent
conflict, so a four-way set-associative version was written — 2^14 sets of four
ways, same 512 KB, sets aligned to 64 bytes so a probe still touches one cache
line. It measured **exactly zero** over 24 interleaved runs.

The reason is the number that should have been measured first. Rebuilding with
a 24-bit table — 128 MB, essentially conflict-free, 2,843 conflicts left out of
973,613 probes — raises the hit rate from **14.4% to 16.6%**. That 2.2 points is
the entire prize available to *any* change in cache geometry: about 21,000
evaluations skipped out of 833,000, under 1% of runtime, which is below the
noise floor of the benchmark. The 128 MB version is itself slower (491 ms vs
459) purely on footprint.

So the misses are not conflict misses that a smarter table could recover. They
are positions that genuinely never come back: only one evaluation in six is of a
position the search has already evaluated. The cache is within two points of its
own ceiling and there is nothing left in it.

**Hoisting the cache probe above the material-draw test**, so a hit also skips
`is_material_draw()` and so the hand-crafted evaluator gets cached in a binary
built without a network. Neutral on the network build, and — the more
interesting half — neutral on the hand-crafted build too, where it is the
difference between no memoisation at all and a 512 KB table in front of the
evaluator. The hand-crafted evaluation is cheap enough that caching it does not
pay for the probe.

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
- Accumulators are refreshed in full rather than updated incrementally. Only the
  768 piece-square rows could be updated that way at all — mobility, king
  attackers and most of the other 166 rows change on nearly every move — so an
  incremental path would cover part of the work and complicate make/unmake for
  the rest. The eval cache took the repeated-position half of that win instead.
- Single-threaded. The `Threads` UCI option is accepted and ignored.
- No opening book, no endgame tablebases.
