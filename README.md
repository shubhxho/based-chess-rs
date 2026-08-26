# Neutron-o1

A chess engine written in Rust. It plays UCI, evaluates positions with a 60 KB
neural network, and that network was trained on games the engine played against
itself. It lands somewhere around **2800 Elo** — see [What that adds up
to](#what-that-adds-up-to) for how that was measured and why the number carries
a wide error bar.

This file is written as a lab notebook rather than a feature list. Most of it is
things that didn't work, because those are the parts I'd have wanted to read
first, and one of them turned out to be worth more than everything else here
combined: for months I was measuring every architecture change through an
uncontrolled variable, and [everything I concluded from
that](#how-loud-the-evaluation-is) was wrong.

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
moved: r hovered around 0.93 the whole way. That flatness looked like the actual
finding, and I wrote for a long time that capacity was never the constraint.

I was about half right, and it took until much later to find out which half.
Width does buy something — just not enough to rescue this input set, and not
enough to show up in the fit at all. The story of how that stayed hidden is
[further down](#the-width-sweep-rerun-with-the-scale-fixed), because it turns on
a mistake that was contaminating every comparison in this file, including this
one.

The constraint is that piece-square features describe where pieces **are**, and
almost everything that decides a chess position is about where pieces can
**go**. A knight on d5 is worth wildly different amounts depending on what it
attacks. A rook is worth a lot more on an open file. Neither fact is recoverable
from a one-hot square index, no matter how wide the layer behind it.

So the budget went into the input instead of the hidden layer. Alongside the 768
piece-square planes there are now 166 rows encoding mobility, passed pawns by
rank, isolated and doubled pawns, rooks on open and half-open files, the bishop
pair, king attackers, and king shelter — all computed from the board and looked
up in the same embedding table.

That was the whole difference. These were all measured back when the hidden
layer was 32 neurons, so the sizes are the old ones:

| Input set | Size | r vs teacher | RMSE |
|---|---|---|---|
| Hand-crafted evaluation (baseline) | — | 0.937 | 192 cp |
| 768 piece-square features | 24.6 KB | 0.955 | 161 cp |
| 934 features, with mobility and structure | 29.8 KB | **0.970** | **130 cp** |

Same 32 neurons. Same optimiser. Same data. Five kilobytes of extra input beat
four times the hidden width.

Nor was it data-starved at that size: retraining on 3.36M positions instead of
2M moves the fit by nothing (r 0.970 → 0.968). Width didn't matter, and at that
scale data didn't either — the input set was the whole thing.

"At that scale" is doing real work in that sentence, and it took a later
experiment to find the edge of it. Deduplicating every shard ever generated into
one 10.2M-position set and training on all of it is worth **+15.5 Elo, 95%
confidence [+5.6, +25.5]**, pooled over 3000 games against the network it
replaced. Three independent sets of openings, +23.7, +8.0 and +14.9 — which is
also a fair illustration of why one 1000-game match is a sample rather than a
result. So the data ceiling was real where it was measured and simply not where
it was assumed: three times the data changed nothing, five times it did.

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

Two further rounds of relabelling both lost, and the second one is the
informative one. By then the teacher really was stronger — the +15.5 Elo network
below — and all 10.2M positions were rescored by it rather than a third of them.
The result was the best fit this project had produced at the time, **r=0.9750**
against the then-shipping network's 0.9609, and it lost by **41 ± 22 over 1000
games**.

Put beside the +15.5 Elo result, that suggests what the labels are actually
worth. Training on every shard ever made — ten million positions carrying labels
from several generations of teacher, most of them weaker than today's — beats
training on one clean generation. Rewriting all of those labels with a single
current teacher, however strong, throws that away: it makes every label agree,
and agreeing labels agree with the student's own biases too. Diversity in the
teaching, not quality, looks like the thing being paid for. The fit gets better
and the engine gets worse, which is the whole argument for judging networks by
games.

The first of the two rounds, run before that teacher existed, lost
**44 ± 22 over 1000 games** — the clearest negative result here, and worth
reading as the boundary of the trick that worked twice. Relabelling pays when
the new teacher is genuinely stronger than the one that wrote the labels. These
labels had already been written by a 6,000-node search; the engine doing the
rewriting searches a slightly smaller tree per node but is not a materially
stronger teacher at the same budget, so there was no new signal to add. What the
round did change was the sample: 3.26M positions rather than the full 10.2M, and
a further 105k dropped because their new score cleared the 2,000cp bar that
separates a position a static evaluation can learn from a position decided by a
tactic. Relabelling is not a repeatable lever. It is worth exactly as much as
the teacher improvement behind it.

**The blend between the teacher's score and the game result** is already where
it should be. `EVAL_W` decides how much of the target is the search's evaluation
and how much is who actually won; it sits at 0.9. Moving it, measured over 1000
games each against the shipping network:

| `EVAL_W` | meaning | result |
|---|---|---|
| 0.7 | triple the weight on game outcomes | **-67.2 ± 21.9** |
| 0.95 | slightly less outcome | -2.8 [-15.3, +9.6] over 3000 games |
| 1.0 | ignore outcomes entirely | -7.6 ± 21.5 |

The curve is sharp on one side and flat on the other, and 0.9 is at the top of
it. A game result is a delayed and noisy verdict on one position — the side that
was winning here often lost for reasons forty plies away — while the search's
score is a direct opinion about this position. Ten percent of the target is
apparently the right amount of that noise: enough to anchor the network to
something outside its own search, not enough to drown the signal.

`EVAL_W=0.95` is the reason the confirmation runs exist. Its first 1000 games
read +6.3 and looked like a small win; two more opening sets took it to -3.6 and
-11.2, and the pooled 3000 games put it slightly *below* the network it was
meant to beat.

## How loud the evaluation is

This is the part I did not see coming, and it is worth more than everything
above it put together.

It started as routine housekeeping. I retrained the shipping network — same
architecture, same 10.2M positions, same everything — expecting a net I could
use as a baseline for architecture experiments. It came out *better* by every
measure I had. Correlation with the teacher went from 0.9794 to 0.9811. Mean
error dropped from 102 centipawns to 82.

Then it lost by **38.0 ± 21.7 over 1000 games**.

I nearly filed that under "fit isn't Elo" — this file already says that in three
places — and moved on. What stopped me was checking the one statistic I had
never looked at: not how *close* the evaluations were, but how *spread out*.
Over 20,000 positions the old network's evaluations had a standard deviation of
549 centipawns. The teacher's were 654. The old network had been quietly
understating every position for its entire life, and my better retrain had fixed
that, landing at 642 — faithful to the teacher, right where a good student
should be.

The fix, if that was really the problem, is embarrassingly small: multiply the
output layer by a constant. It cannot change which position the network prefers,
the ranking is untouched, r does not move by a thousandth. It only changes how
loudly the network says what it already thought. Sweeping it, 1000 games each
against the old network:

| output gain | result |
|---|---|
| 1.00 | **-38.0 ± 21.7** |
| 0.90 | +7.0 |
| 0.85 | +19 over 2000 games |
| 0.80 | +43.3 |
| 0.75 | +56.1 |
| 0.70 | **+59.6 ± 21.9**, and +52.2 ± 21.8 on a second set of openings |
| 0.65 | +53.9 ± 21.8 |
| 0.60 | +58.6 ± 21.8 |
| 0.55 | +47.9 ± 21.7 |

A hundred Elo between the top of that curve and the bottom, and the network
knows exactly the same things at every point on it.

Here is what I think is going on. The search never consumes the evaluation as a
number on its own — it consumes it as one side of a comparison against a margin.
Reverse futility, razoring, the null-move verification bound, the static side of
late-move reductions: all of them ask "is this evaluation more than N centipawns
above beta?" Those margins are constants, tuned over months against a network
that happened to speak quietly. Hand the search a correctly calibrated
evaluation and every one of those thresholds is now effectively too small, so it
prunes in places it shouldn't. The network got better and the system got worse,
because only half the system was updated.

That is a testable story, so I tested it. `MARGIN` in `search.rs` scales all five
of those thresholds at once. If the margins really are the mechanism, then taking
the *natural-scale* network — the one that loses by 38 — and widening the margins
to match should recover the loss. 1000 games at each setting, against the engine
that ships:

| `MARGIN` | vs shipped |
|---|---|
| 100 (the tuned values) | -82.8 ± 22.1 |
| 143 | -45.1 ± 21.7 |
| 200 | **-26.5 ± 21.6** |

So the mechanism is real and it is most of the story: widening five constants
recovers 56 Elo of an 83 Elo hole, monotonically, with no retraining anywhere.
It is also **not all** of the story, because the best margin setting still lands
27 Elo behind simply shipping a quieter network.

I don't have a clean answer for the remaining 27. The honest candidates are the
eval-scale quantities `MARGIN` doesn't touch — the aspiration window's initial
delta, the correction-history tables which learn a correction *in centipawns*
and so have their own implicit scale, and every static evaluation that gets
written into the transposition table and compared against later. Scaling the
network is a single change that fixes all of them at once, which is probably why
it wins. Scaling the margins fixes five of them by hand.

That is the argument for the gain being a real fix rather than a hack, and also
the argument against the version of this section I wrote first, which claimed
the margins were simply *the* explanation. They are about two-thirds of it.

Two things convinced me this isn't an artifact of where I fitted it.

**It doesn't need tuning.** The gain shipped at 0.70, and when I later reran the
sweep against the wider network that ships now, 1000 games at each of 0.55,
0.60, 0.65, 0.75 and 0.80 came back at -4.2, +4.5, -3.8, -11.1 and -3.8, all
± 21.5. Five thousand games, and not one of them can be told apart from 0.70.
The plateau is wide, it sits in the same place for both architectures, and the
sixty Elo comes from *not being at 1.00* rather than from finding a precise
value. These are also the cleanest measurements in this file, incidentally: one
set of weights rescaled several ways, so no retraining variance enters at all.

**It doesn't fall off with depth.** The constant was fitted at 20,000 nodes a
move, which is exactly where an overfitted parameter would look best. 1000 games
at each budget against the same opponent:

| nodes per move | result |
|---|---|
| 5,000 | +27.5 ± 21.6 |
| 10,000 | +55.4 ± 21.8 |
| 20,000 | +62.6 ± 12.6 (3000 games) |
| 50,000 | +63.2 ± 21.9 |
| 100,000 | +62.5 ± 21.9 |
| 200,000 | +55.0 ± 21.8 |

The win holds inside its own error bars from 20k out to 200k, ten times past the
tuning point. What drops away is the *shallow* end — at 5,000 nodes it is less
than half. That is the right direction for the explanation above: a shallow
search prunes less and leans on the static evaluation less often, so how loud it
is matters less. The 200k reading sits seven Elo under 100k, which one match
cannot separate from noise, but it is also the shape a fixed constant would make
if it slowly stopped suiting a deeper tree. Unresolved, and worth re-measuring
before anyone trusts 0.70 at long time controls.

Fixed nodes is what this file measures, because it doesn't move with machine
load — and because it hands the new network its slower evaluation for free. On
the clock the same comparison gives **+42.3 ± 24.3** over 800 games at 100ms a
move and **+51.6 ± 34.4** over 400 at 300ms. So about twenty of the sixty-three
Elo is the harness being generous, and the rest is real at every time control I
tried.

Pooled over three independent sets of openings at 20,000 nodes, the gain plus
the width change below is worth **+62.6 ± 12.6 Elo** over the previous release:
+62.9, +65.7, +59.3. Those three agree far more closely than any earlier result
in this file, which is what an effect well clear of the noise floor looks like.

One last thing this reframes. Two rounds of relabelling above lost while fitting
the teacher better than anything before them, and I read that at the time as
evidence that fitting too well is itself harmful. There is a duller explanation
available now: a network that fits its teacher more closely also inherits its
spread more closely, and nothing in that pipeline was correcting the scale. Both
of those experiments deserve rerunning at a gain — a knob that did not exist
when they were judged.

It lives in `train.py` as `OUT_SCALE`, applied to the output layer at export
rather than to the score in `net.rs`, so `net.bin` stays the single description
of what the engine computes and the trainer's own quantised reference still
agrees with the engine byte for byte.

## The width sweep, rerun with the scale fixed

Once the gain existed, every architecture claim in this file became suspect —
they were all measured before it, which means they were all measured *through*
it. A wide network and a narrow one trained the same way don't just differ in
width; they differ in how loud they are, and I now knew that was worth sixty
Elo. So I went back and ran them again with the gain pinned at 0.70, 1000 games
each at 20,000 nodes:

| change | size | vs that network |
|---|---|---|
| 32 → 64 hidden neurons | 30 KB → 60 KB | **+12.5 [+0.1, +24.9]** over 3000 games |
| squared clipped ReLU, 32 neurons | 30 KB | -3.8 ± 21.5 |
| squared clipped ReLU, 64 neurons | 60 KB | +14.3 ± 21.6 |
| four king buckets on the whole feature block, 32 neurons | 118 KB | +8.3 ± 21.5 |
| king buckets and 64 neurons together | 235 KB | +4.9 ± 21.5 |

Only the first survives. Every one of those five had measured *negative* the
first time round — the widest of them read -18.4 — so "width does nothing" was
never a fact about width. It was an artifact of comparing evaluations that were
on different scales.

That width row is pooled over three matches and — deliberately — over **two
separately trained networks**. Same recipe, same seed, run twice. They measured
+13.6 ± 21.6 and +18.4 ± 21.6, and +6.3 ± 21.5.

That gap is worth dwelling on. MLX's reductions aren't bit-deterministic, so two
identical runs land on different weights. Their validation losses were 0.00340
and 0.00341 — indistinguishable, as you'd hope. In games they were twelve Elo
apart, which is wider than any single match's confidence interval. The
uncomfortable implication is that one 1000-game match against one trained
network resolves neither the opening set nor the training run, and most of this
file is built out of exactly that. I have tried to say so wherever it matters.

The network that ships is the one that measured **+6.3**, not the +18 sibling,
because it is the one `NET_H=64 python train.py` actually reproduces. The
sibling was an export-time rescale of the same run and I would rather ship the
weaker reproducible artifact than the stronger unreproducible one.

Width is not free either: 64 neurons is 6% more time per node, and a
node-limited match hides that by construction. At a real time control, 600 games
at 100ms a move, it wins by +18.0 ± 27.8.

And then width stops, immediately. 128 neurons scores **exactly even** — -0.0 ±
21.5 over 1000 games — while costing 85% more time per node, because sixteen
128-bit accumulator registers per perspective is more than the register file
holds and the inner loop starts spilling to memory. So the original plateau was
real. It just started one doubling later than I measured it, and the fit
statistics never showed the difference at any point.

The king buckets are the more interesting failure. They help on their own and
*hurt* once the network is also wider, which reads less like two independent
improvements and more like two ways of spending the same capacity.

## What that adds up to

Every number so far is a margin over some other build of this same engine, which
tells you the direction of travel and nothing about where it has arrived. Two
measurements fix that.

First, a gauntlet against every binary this repo has kept — 600 games each at
20,000 nodes a move, plus a self-play control to check the harness isn't lying
to me:

| opponent | result |
|---|---|
| the engine against itself (control) | +5.2 ± 34.1 |
| the previous release | +62.6 ± 12.6 (3000 games) |
| `sable-new` | +112.7 ± 29.3 |
| `sable-old`, `sable-std` | +130.3 ± 29.8 (both — they play identically) |
| `sable-net` | +132.9 ± 29.9 |
| `sable-net-v1` | +150.7 ± 30.5 |
| `sable-hce`, the hand-crafted evaluator | +156.2 ± 30.7 |

The control row is what makes the rest readable. The same binary against itself
comes back at +5.2 with zero comfortably inside the interval, so colour swapping
and pair ordering aren't quietly handing anyone an advantage. And `sable-old`
and `sable-std` returning byte-identical scores isn't a fluke — they evaluate
every position identically, so they play identical games.

Second, an outside anchor, because nothing above escapes this repository.
Stockfish with `UCI_LimitStrength` plays to a nominal rating, so playing it at
several settings puts the engine on a scale someone else maintains. 300 games at
each of five settings — 1500 in total — at 100ms a move:

| Stockfish `UCI_Elo` | score | implied |
|---|---|---|
| 2600 | 0.772 | 2812 |
| 2700 | 0.638 | 2799 |
| 2800 | 0.472 | 2780 |
| 2900 | 0.410 | 2837 |
| 3000 | 0.328 | 2876 |

**About 2800.** Fitting one rating to all 1500 games by maximum likelihood gives
2819 ± 19; interpolating the point where the score crosses 0.5 gives 2783. I
would quote the range rather than either endpoint, and ±40 is honest where ±19
is not.

The reason those two disagree is the interesting part, and it is not noise. The
residuals from the one-parameter fit drift monotonically — -0.008, -0.027,
-0.056, +0.024, +0.067 as the setting rises — which means the engine loses less
to Stockfish's strongest settings than a logistic curve says it should. Let the
slope float and it fits at 0.83: a hundred of Stockfish's nominal points behave
like about eighty-three real ones over this range. `UCI_LimitStrength` reaches
its target by degrading play in discrete internal steps, so there is no reason
its scale should be linear, and here it isn't.

That is why the crossover is the number worth quoting. Where two engines score
0.5 against each other they are equal by definition, and that point doesn't care
whether the slope is right — which is fortunate, because it isn't.
`tests/calibrate.py` reproduces all of it and needs Stockfish on `PATH`, which
nothing else in this repo does. It is still someone else's approximate
self-calibration and still not a CCRL or FIDE number.


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

The short version, before the long one:

| Layer | Implementation |
|---|---|
| Board | Bitboards, 12 piece planes plus a mailbox, incremental Zobrist |
| Attacks | Magic bitboards; the magics are *searched* at startup, so they validate themselves |
| Movegen | Fully legal — pins, check evasions and en-passant discovery all resolved during generation |
| Search | Fail-soft PVS with TT, null move, ProbCut, LMR, singular extensions, SEE and history pruning, static-eval correction history |
| Eval | 934 -> 64 -> 1, int8, eight output buckets by material, NEON inference |
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

Quiescence scores its captures against its own threshold, which is worth a
paragraph because it is the one change here that had to be settled by playing
rather than by measuring. Ordering sorts a capture above the quiet moves when
its swap value clears -20; quiescence will not search one below 0. Scoring at 0
up front makes the sign of the score answer the question, so the swap evaluation
runs once per capture instead of twice. The moves that change bands are exactly
the ones quiescence was going to discard.

That reasoning is not quite airtight — the node count moves, by a little at
shallow depths and 3.1% at depth 18, most likely because `see_ge` is not exactly
monotone in its threshold given how much pruning it does on the way to an
answer. A changed tree is a changed engine, so the benchmark stops being
evidence. Six hundred games at 20,000 nodes a move put it at **+10 Elo ± 28**:
neutral, with the direction favourable and the interval far too wide to call it
anything else. Kept because neutral strength for less work is still a win —
9,326,301 nodes against 9,623,933 at depth 18, and about 3% quicker there in
wall clock.

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
first wrote down were wrong. The oracle is what told me the engine was fine and
my test data wasn't, which is a humbling way to spend an evening and the reason
every correctness claim below is checked against something I didn't write.

| Suite | Result |
|---|---|
| Classic perft (startpos, kiwipete, positions 3–6) | 6/6 exact, to depth 8 |
| Oracle-verified edge cases (castling, ep pins, promotion races) | 20/20 exact |
| Randomised positions, depth 4 and depth 5 | 119/119 exact at both |
| Rust NEON inference vs NumPy reference | 80/80 identical |
| Insufficient material evaluates to exactly 0 | K vs K, K+N vs K |

`bench 13` is bit-identical run to run, which makes it a proper refactoring
guard. When I removed a redundant legality check on the transposition move, the
node count stayed at exactly 1,321,821 — proof the change was a pure speedup and
not a silent behaviour change.

Throughput is around 3.3 Mnps on a single M-series core.

Matches are run at a fixed node count rather than a fixed time, so results don't
shift with machine load, and colours are swapped on every opening pair. Every
result below is 20,000 nodes per move, self-play from randomised openings:

| Opponent | Games | Score | Elo |
|---|---|---|---|
| 1.1 (before correction history and the time-management rework) | 800 | 0.546 | **+32 ±24** |
| 1.1, at 100,000 nodes per move | 300 | 0.533 | +23 ±39 |
| the same build without ProbCut and killer hygiene | 2400 | 0.508 | +6 ±14 |

Those were search changes, measured against the network of the day. The
[gauntlet above](#what-that-adds-up-to) is the current picture against every old
binary, at 600 games each rather than 200 — the three rows this table used to
carry for `sable-std`, `sable-net` and `sable-hce` are superseded by it, and
they were all short enough to have ±49 intervals anyway.

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

Recorded because a negative result nobody writes down just gets re-attempted,
usually by me, usually six weeks later.

**First, how any of this gets measured.** This machine drifts. In one afternoon
the same untouched binary reported medians of 495, 476, 455 and 428 ms — a
spread wider than anything I changed that day. So comparing two medians taken
minutes apart doesn't measure the change, it measures the weather.

`tests/paircmp.sh` runs both binaries back to back and keeps the *difference*
from each pair, alternating which one goes first. Whatever the machine is doing
hits both halves and cancels. Point it at two copies of the same binary and it
says 0 ms, ±3 ms, over 21 pairs. That ±3 ms is the floor. Anything smaller than
it was never a result, and I have the retracted claims to prove it.

The correctness check is the node count: 1,376,993 at `bench 13` and 12,865,968
at `bench 18`. If a change moves either number it changed the search, whatever
it says on the label — and then benchmarks cannot settle it and it has to go and
play games.

**Making quiescence decide captures lazily.** Quiescence scores every capture
before it looks at any of them, and each score costs a full swap evaluation.
Then it usually leaves after two or three moves. All that work, thrown away.

The fix looks obvious: give each capture the score it would have if it won, and
only run the real test when a move actually wins a scan. That's exact, and the
proof is nicer than I expected — a provisional score is always an upper bound,
resolving it only ever lowers it, and the scan keeps the earliest maximum, so
the same move comes out every time, ties and all. Node counts agreed to the
digit at two depths.

It just isn't faster. Resolving a score in place makes the scan start over, and
that restart costs about what the skipped evaluation saves. 428 against 430 ms
over 41 pairs. Deleted.

**Two smaller versions of the same lesson.** `features_both` checks `n < MAX_F`
before every feature it writes, and those checks are dead: the count can't
exceed `2·pieces + 12`, which is 76 on a full board against a limit of 96. Split
the function on a const generic and the hot copy carries none of them. It read
+1 ms over 25 pairs, then **-5 ms over 41**, faster in only 16 of them. Not
noise — the second copy of the function adds 16 KB, and the instruction cache
charges more for that than the compares ever did.

There's another way to get there: make the buffer bigger than any FEN could
fill, and drop the checks with nothing duplicated. But `MAX_F` is written into
the featdump header and mirrored in `train.py`, so that grows every training
record by half to buy something already measured below the floor. The checks
stay.

Merging the piece-square and mobility walks, which scan the same four bitboards
twice over, read +9 ms across 25 pairs and then -2 ms across 41. The first
number is exactly why the interval gets printed next to the median.

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

## Stronger supervised data on Apple Silicon

The MLX trainer is already the Apple-Silicon training path; Rust remains the
low-latency inference and search path.  `prepare_hf.py` now streams
[`Lichess/chess-position-evaluations`](https://huggingface.co/datasets/Lichess/chess-position-evaluations),
a CC0 corpus with Stockfish scores, depth, node count and PVs.  It keeps only
valid, non-mate, quiet, non-check positions and writes the existing augmented
shard format.  It records the dataset revision and every filter in
`hf_source.json`.

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements-training.txt

# First run a small, inspectable pilot. The default source revision is pinned.
.venv/bin/python prepare_hf.py data/lichess-sf --max-positions 200000
ENGINE="$PWD/target/release/sable" DATA_DIR=data/lichess-sf \
  DATA_GLOB='aug_hf_*.txt' EVAL_W=1 .venv/bin/python train.py 0 20
.venv/bin/python tests/verify_net.py
```

The source scores are white-relative and the trainer converts them to the
mover's perspective. `EVAL_W=1` is required because these score-only rows use a
neutral dummy game result. Start with a deterministic 2–10M-position cap: the
trainer intentionally materialises its corpus and feature cache. The cache is
now keyed by the ordered FEN corpus, so labels can be relabelled without a
feature dump but an equal-sized different corpus cannot silently reuse features.

This makes a credible route to a 3000-Elo *candidate*, not a rating promise.
Use a stronger teacher/data mix only when `tests/verify_net.py` passes, then
accept the network only after long paired arena runs across separate opening
sets and a fresh Stockfish calibration. Fit loss alone is not an Elo result.

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
prepare_hf.py    streams, filters, and records labelled Lichess training shards
relabel_sf.py    optional stronger Stockfish relabeller (SF_NODES is reproducible)
arena.py         match runner with Elo confidence intervals
publish_hf.py    uploads the network to the Hugging Face Hub
src/tests.rs     unit tests (cargo test)
tests/           perft suites, the python-chess oracle, inference verification
tests/paircmp.sh paired A/B timing -- the only honest way to read a change here
tests/calibrate.py  rating anchor against Stockfish (needs it on PATH)
.github/         CI: fmt, clippy -D warnings, tests, perft, size budget
```

The network is on the Hub at
[`shubhxho/sable-chess-net`](https://huggingface.co/shubhxho/sable-chess-net),
with the format documented well enough to read it without this engine.

## Honest limitations

- It is distilled from itself. The ceiling is its own search quality, not a
  stronger reference — Stockfish appears here only as a measuring stick, never
  as a teacher, and nothing it plays has ever been trained on.
- The rating is an anchor, not a rating. See [What that adds up
  to](#what-that-adds-up-to): five Stockfish settings disagree by 96 Elo, and
  Stockfish's own `UCI_Elo` calibration is approximate.
- Most single results here are one 1000-game match, and two identically-trained
  networks have measured twelve Elo apart. Treat any margin under about 20 Elo
  as a hint rather than a result unless it was confirmed on a second set of
  openings, which the important ones were.
- Accumulators are refreshed in full rather than updated incrementally. Only the
  768 piece-square rows could be updated that way at all — mobility, king
  attackers and most of the other 166 rows change on nearly every move — so an
  incremental path would cover part of the work and complicate make/unmake for
  the rest. The eval cache took the repeated-position half of that win instead.
  Worth knowing before anyone attempts it: timing the two halves of `evaluate`
  at nanosecond resolution puts feature extraction at 51 ms and the accumulator
  walk at 28 ms of a 472 ms `bench 13`. The whole accumulator step is 6% of
  runtime, so the entire prize for making it incremental is under 4%, and only
  part of that is reachable.
- Single-threaded. The `Threads` UCI option is accepted and ignored.
- No opening book, no endgame tablebases.
