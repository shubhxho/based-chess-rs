---
license: mit
library_name: mlx
pipeline_tag: other
inference: false
language:
  - en
tags:
  - chess
  - chess-engine
  - nnue
  - mlx
  - apple-silicon
  - quantization
  - int8
  - distillation
  - knowledge-distillation
  - self-play
  - uci
  - rust
  - no-std
  - edge
metrics:
  - elo
  - pearsonr
co2_eq_emissions:
  emissions: 1.4
  source: "estimated: 6 minutes of Apple M-series GPU at roughly 20W, at 0.7 kgCO2eq/kWh"
  training_type: "distillation from self-play search labels"
  geographical_location: "India"
  hardware_used: "Apple M-series (MLX, unified memory)"
model-index:
  - name: sable-chess-net
    results:
      - task:
          type: other
          name: Chess play (engine strength)
        dataset:
          type: self-play
          name: Sable self-play, 10.1M deduplicated positions
        metrics:
          - type: elo
            name: Elo vs previous release (3000 games, 20k nodes/move)
            value: 63.0
          - type: elo
            name: Elo anchored to Stockfish UCI_Elo (200 games per setting, 100ms/move)
            value: 2800
          - type: pearsonr
            name: Correlation with teacher search (rank quality, gain-invariant)
            value: 0.9803
---

# Sable — a 60 KB standalone chess evaluation network

The complete evaluation function for the [Sable](https://github.com/shubhxho/sable)
chess engine, distilled from that engine's own search and trained with
[MLX](https://github.com/ml-explore/mlx) on Apple silicon.

**60,976 bytes.** It is the entire evaluation — there is no hand-crafted term
underneath it, and no framework needed to run it.

```
934 features -> 64 hidden (per perspective, shared) -> clipped ReLU -> 1 of 8 output buckets
```

The engine around it plays at roughly **2800 Elo**, anchored against Stockfish's
`UCI_Elo` settings. That anchor is worth about ±70, for reasons set out under
[Playing strength](#playing-strength).

If you read one section, make it [the output gain](#the-output-gain). A single
constant multiplying the output layer — which cannot change which position the
network prefers — was worth about 60 Elo, and getting it wrong had been
poisoning every architecture comparison in this project for months.

## What this is, in one paragraph

It's the evaluation function out of a chess engine I wrote. The engine searched
its own games, and this network was trained to guess what that search would have
said without doing the search. It's 60 KB of int8 weights, it runs on integer
SIMD with no framework underneath it, and it is not a PyTorch model — you can't
`from_pretrained` it. If you want to *use* it, you want
[the engine](https://github.com/shubhxho/sable); if you want to *read* it, the
[format section](#format) is complete enough to parse `net.bin` in twenty lines
of NumPy, which is included below.

- **Developed by:** [@shubhxho](https://huggingface.co/shubhxho)
- **Model type:** quantised int8 feedforward evaluation network (NNUE-style), distilled from tree search
- **Inputs:** 934 sparse binary features per side, computed by the engine
- **Output:** one scalar, centipawns, from the side to move's point of view
- **Trained with:** [MLX](https://github.com/ml-explore/mlx) on Apple silicon
- **License:** MIT
- **Repository:** https://github.com/shubhxho/sable

## What it's for, and what it isn't

**Use it for:** running the Sable engine; reading a small, complete, honestly
documented example of a quantisation-aware distilled evaluation; lifting the
format or the training loop for your own engine. The whole thing is MIT and I'd
be glad to see it reused.

**Don't expect it to:** work as a general chess model, produce moves on its own,
or load into a transformers pipeline. It has no notion of a legal move. Hand it
a position and it returns a number; everything that makes that number useful —
move generation, search, pruning, time management — lives in the engine, and the
number is close to meaningless without it. The gain section below is a long
argument for exactly that point: the same weights are worth a hundred Elo more
or less depending on the search wrapped around them.

**Bias and risk, honestly:** it's a chess evaluator. The realistic harm is
someone cheating at online chess with it, which is true of every engine ever
published and which this one is far too weak to be attractive for. The more
interesting caveat is epistemic: it was distilled entirely from its own search,
so it has inherited that search's blind spots and there is no external teacher
anywhere in the loop to catch them.

## The input set is the whole story

The obvious design uses the standard NNUE input: 768 binary features, one per
(piece, colour, square). I built that first. At this size it played **165 Elo
worse** than the hand-crafted evaluator it was supposed to replace, which was a
memorable afternoon.

The instinct is that it's too small. It isn't. Sweeping the hidden layer from 16
to 128 neurons — an 8x range — barely moves the fit against the teacher; r sits
near 0.93 the whole way. That flatness is the finding: capacity was not the
binding constraint.

It was not *nothing*, either, and it took a later result to separate the two.
Every width comparison here predates the output gain below, so each one measured
a wide network against a differently-scaled narrow one. Held at a fixed gain, 64
neurons beat 32 by +12.5 Elo [+0.1, +24.9] over 3000 games and 128 beat 64 by
nothing at all. The plateau is real; it starts one doubling later than this
sweep said, and the fit numbers never showed the difference.

Here's the actual problem. Piece-square features describe where pieces **are**,
and almost everything that decides a chess position is about where they can
**go**. A knight on d5 is worth wildly different amounts depending on what it
attacks. A rook is worth much more on an open file. Neither fact is recoverable
from a one-hot square index, no matter how wide you make the layer behind it —
the information simply isn't in the input.

So the budget went into the input instead of the hidden layer. Alongside the 768
piece-square planes sit 166 rows encoding mobility, passed pawns by rank,
isolated and doubled pawns, rooks on open and half-open files, the bishop pair,
king attackers and king shelter — all computed from the board by the engine and
looked up in the same embedding table. Each row costs 64 bytes.

| Input set | Size | r vs teacher | MAE | RMSE |
|---|---|---|---|---|
| Hand-crafted evaluation (baseline) | — | 0.937 | 96.3 cp | 191.6 cp |
| 768 piece-square features | 24.6 KB | 0.955 | 90.3 cp | 161.3 cp |
| 934 features, with mobility and structure | 29.8 KB | **0.970** | **79.8 cp** | **130.1 cp** |

Same 32 neurons, same optimiser, same data. Five kilobytes of extra input beat
four times the hidden width.

## Playing strength

Measured at a fixed 20,000 nodes per move so results don't move with machine
load, from randomised openings, colours swapped on every pair:

| Matchup | Result |
|---|---|
| 768-feature net **replacing** hand-crafted eval | −165 ± 69 Elo (200 games) |
| 768-feature net **correcting** hand-crafted eval | +57 ± 28 Elo (600 games) |
| **934-feature standalone net** vs hand-crafted eval | +35 ± 34 Elo (400 games) |
| **934-feature standalone** vs the 768-feature hybrid | −3 ± 34 Elo (400 games) |
| **the rescaled 32-neuron net** vs the previous release | +59.6 ± 21.9, +52.2 ± 21.8 (2000 games) |
| **this network** (64 neurons) vs that | +12.5 [+0.1, +24.9] (3000 games) |
| **this network** vs the previous release | **+63.0 ± 12.6** (3000 games) |
| the same, on the clock | +42.3 ± 24.3 at 100ms, +51.6 ± 34.4 at 300ms |

The fourth row is the one that decided the shape of this network. The standalone
version is statistically indistinguishable from the hybrid in games, while carrying no
hand-crafted evaluation at all and tracking the teacher considerably better. The
same feature idea that turned a −165 Elo replacement into a viable one is what
makes the standalone version possible.

Worth being straight about: the standalone net fits the teacher much better
(RMSE 130 vs 161 cp) than the hybrid but does not out-play it. Better regression
against a search's output is not the same thing as better move ordering inside
one, and these match lengths cannot resolve a difference this small.

For an absolute figure, the engine was played against Stockfish under
`UCI_LimitStrength`, 200 games at each setting, 100ms a move:

| Stockfish `UCI_Elo` | score | implied |
|---|---|---|
| 2200 | 0.958 | 2741 |
| 2500 | 0.853 | 2805 |
| 2800 | 0.465 | **2776** |
| 3000 | 0.335 | 2881 |

Call it **2800**. The 2800 row deserves the most weight because it is nearest
parity and extrapolates least. The four anchors disagree by 140 Elo, and that
spread is the honest precision — this locates the engine on someone else's scale
rather than rating it, and it is not a CCRL or FIDE number.

## The output gain

This is the part worth reading even if nothing else here interests you.

A network distilled from a search learns to reproduce that search's score, and
that includes reproducing its **spread**. Measured over 20,000 positions, the
previous release evaluated with a standard deviation of 549 centipawns where its
teacher sat at 654 — it had been quietly understating every position for its
whole life. Retraining the same architecture on the same data fixed that, landing
at 642, and improved every fit statistic: r from 0.9794 to 0.9811, mean error
from 102cp to 82cp.

That better network lost by **38.0 ± 21.7 over 1000 games**.

Multiplying its output layer by a constant is the only thing that then separates
the two. It cannot reorder the network's preferences — r does not move — it only
changes how loud the evaluation is. Swept at 1000 games each against the previous
release: gain 1.00 gives -38.0, 0.90 gives +7.0, 0.80 gives +43.3, 0.70 gives
+59.6, 0.60 gives +58.6, 0.55 gives +47.9.

A hundred Elo across that curve, with the network knowing exactly the same things
at every point on it. The likely mechanism is that a search never consumes a
static evaluation alone — it compares it against margins, in centipawns, for
reverse futility, razoring, null-move verification and late-move reductions.
Those margins were tuned against an evaluation that happened to speak quietly.
Fix the network's calibration without fixing them and every threshold fires in
the wrong place.

It ships at `OUT_SCALE = 0.70`, applied to the output layer at export rather than
to the score in the engine, so this file stays the single description of what the
engine computes. Rerunning the sweep against the 64-neuron network put 0.55
through 0.80 all within noise of 0.70 across another 5000 games: the plateau is
wide and did not move with the architecture. The 60 Elo comes from not being at
1.00, not from finding a precise value.

## Architecture

- **Perspective pairing**: features are built twice per position, once from each
  side's point of view, with squares mirrored and colours relabelled so block 0
  is always "mine". One weight matrix serves both sides, so the network learns a
  single function of "my position" rather than two of "white's position".
- **Weights**: int8 feature transformer (`QA = 127`), int16 biases, int8 output
  layer (`QB = 64`), output scaled to centipawns by `SCALE = 400`.
- **Output buckets**: 8 output layers selected by remaining material. The
  feature transformer stays shared — what changes across a game is how the same
  signals should be weighed, not what they are.
- **Inference**: ARM NEON intrinsics (`vmovl_s8`, `vmlal_s16`, `vaddvq_s32`).

| Tensor | Shape | Type | Bytes |
|---|---|---|---|
| `ft_w` | 934 x 64 | int8 | 59,776 |
| `ft_b` | 64 | int16 | 128 |
| `out_w` | 8 x 128 | int8 | 1,024 |
| `out_b` | 8 | int32 | 32 |
| header | magic, inputs, hidden, buckets | uint32 | 16 |
| | | **total** | **60,976** |

### Feature-space layout

| Rows | Block | Meaning |
|---|---|---|
| 0–767 | piece-square | `(relative_colour, piece_type, square)` |
| 768–863 | mobility | `(relative_colour, N/B/R/Q, moves 0..11)`, one per piece |
| 864–879 | passed pawns | `(relative_colour, rank)`, one per passed pawn |
| 880–887 | isolated pawns | `(relative_colour, count 0..3)` |
| 888–895 | doubled pawns | `(relative_colour, count 0..3)` |
| 896–901 | rooks, open file | `(relative_colour, count 0..2)` |
| 902–907 | rooks, half-open | `(relative_colour, count 0..2)` |
| 908–909 | bishop pair | `(relative_colour)` |
| 910–925 | king attackers | `(relative_colour, attackers 0..7)` |
| 926–933 | king shelter | `(relative_colour, pawns 0..3)` |

Output bucket, which must be reproduced exactly, integer division included:

```python
bucket = min((max(pieces_on_board - 1, 0) * 8) // 32, 7)
```

## Training

The teacher is the engine's **own alpha-beta search** — the distillation
principle behind DeepMind's searchless grandmaster-level chess, at a size that
fits in L1 cache rather than a TPU pod. The student never searches.

- **Data**: 10.2M positions from engine self-play out of randomised openings of
  8 to 16 plies, labelled at 3k to 6k nodes/move, deduplicated by FEN across
  every generation run ever made. The first two plies of real play are skipped —
  those are the engine repairing whatever the random opening did.

  This supersedes the previous release, which trained on 3.4M positions from a
  single generation on the theory that one teacher beats an average of several.
  Measured over 3000 games, that theory is worth **-15.5 Elo**: training on
  everything, older labels included, beats training on the newest shard alone by
  +15.5 with 95% confidence [+5.6, +25.5]. The older labels are weaker but they
  are not noise, and there are seven million of them.
- **Filtering**: positions are dropped when the side to move is in check or the
  best move is a capture. There the tactic decides the game, not the static
  evaluation, and training on them only teaches the network to imitate search —
  which it has no mechanism to do.
- **Objective**: MSE in win-probability space,
  `sigmoid(net / 400)` against `0.9 * sigmoid(search / 400) + 0.1 * result`.
- **Optimiser**: AdamW, batch 16384, lr 1e-2 with one warmup epoch then cosine
  decay over 15 epochs. 5% of positions are held out; the exported network is
  the epoch that did best on them, not the last one.
- **Output gain**: the exported output layer is multiplied by `OUT_SCALE`, 0.70.
  This is not part of the objective and it does not change which position the
  network prefers; it only makes every evaluation quieter by a constant. A
  network trained to reproduce a search's score reproduces its spread as well,
  and the search plays substantially worse when handed one. The same network
  exported at gain 1.00 loses 38.0 ± 21.7 to the previous release; at 0.70 it
  wins by 59.6 ± 21.9. See README.md for the full sweep.

Data volume is not the constraint either: retraining on the full 3.36M against
2M moves the fit by nothing worth reporting (r 0.970 -> 0.968, RMSE 130.1 ->
130.8 cp). Between that and the width sweep, the feature set was the only thing
that ever mattered.

A second iteration of the same idea did **not** pay off. Four million fresh
positions, labelled by the network below and the search that ships with it,
produced a network that lost to its own teacher by 20.0 +/- 24.1 over 800 games
and 24.4 +/- 21.6 over another 1000 — about 22 Elo down across 1800 games, twice
in a row. Mixing those shards with the previous round's (6M positions in total)
landed at +4.3 +/- 24.1, and doing the same with bucket-balanced sample weights
at +4.9 +/- 21.5: nothing, either way.

The overlap between rounds was the missing piece. Self-play deduplicates within
a generation run but not across them, so a mixed set grades the shared openings
twice, with the older and weaker teacher's label surviving. Deduplicating across
shards and keeping the newer label on the overlap gives **+12.9 +/- 21.5 over
1000 games and +11.9 +/- 19.7 over 1200** — about +12 across 2200 — and that is
the network described here.

Weighting older shards down as well (`SHARD_DECAY` below 1) loses 24.0 +/- 21.6
and stays off by default. The old positions carry their weight; only their
labels were stale. One round of relabelling against a
stronger search was worth about 23 Elo and the next round was worth zero, so
the gain came from the teacher's jump in strength rather than from iterating,
and there is no free ladder here.

What did move: the teacher. Relabelling from scratch with a search roughly 30
Elo stronger, at 6k nodes instead of 5k and with duplicates removed, produced a
network that beats the one it replaces by **+23.5 ± 24.1 Elo over 800 games**,
and by +23.0 ± 21.6 over a further 1000 — the same margin twice.
Its fit numbers against that harder, less repetitive data (r 0.974, MAE 85.1,
RMSE 137.7 cp) are not comparable to the table above, which was measured on the
old shards — a better teacher gives you harder targets, so a bigger residual
against a better opponent is the expected shape of an improvement.

### Features come from the engine, never from the trainer

The trainer doesn't compute features. It asks the engine for them, through a
`featdump` command that dumps the active indices for each position, and reads
them back.

This is worth the awkwardness. Two implementations of one feature map is a bug
class where the trainer and the engine quietly disagree about what feature 431
means, and what you get is a network that loads cleanly, runs at full speed, and
plays slightly badly for reasons nothing will point you at. I would rather pipe
a gigabyte of indices through a subprocess than debug that. `src/net.rs` is the
single source of truth for both sides.

### Quantisation-aware by construction

Weights are projected back into the int8 box **after every optimiser step**,
never rounded at the end:

```python
model.ft = mx.clip(model.ft, -127.0 / QA, 127.0 / QA)
model.out = mx.clip(model.out, -127.0 / QB, 127.0 / QB)
```

So the exported network computes the function the trainer actually converged to,
rather than a rounded-off approximation of it.

Verified rather than asserted: `net.bin` gets replayed through an independent
NumPy reference that reproduces the Rust inference operation for operation, and
the two agree on 80/80 test positions. The only disagreement that check has ever
turned up was Python's floor division against Rust's truncation on negative
scores — which was a bug in the reference, not the engine, and exactly the kind
of thing the check exists to find.

## How to get started

The fastest path is the engine itself:

```bash
git clone https://github.com/shubhxho/sable && cd sable
cargo build --release
./target/release/sable          # then speak UCI, or type `bench 13`, `eval`, `d`
```

`net.bin` is baked into the binary with `include_bytes!`, so the build already
contains this network — there's nothing to download at runtime. To read the
weights directly instead, see the NumPy snippet under [Format](#format).

## Format

Little-endian, tightly packed, no framework dependency:

```
magic   u32   0x334C4253 ("SBL3")
inputs  u32   934
hidden  u32   64
buckets u32   8
ft_w    i8[934 * 64]     row-major [feature][neuron]
ft_b    i16[64]
out_w   i8[8 * 128]      row-major [bucket][neuron];
                         within a bucket, first 64 = side to move,
                         last 64 = opponent
out_b   i32[8]
```

The header carries `inputs`, `hidden` and `buckets`, so read those rather than
hardcoding them — this network was 32 hidden neurons until recently and the
loader rejects a file whose header disagrees with the build rather than
misreading it.

```python
import struct, numpy as np
b = open("net.bin", "rb").read()
magic, IN, H, B = struct.unpack("<IIII", b[:16]); o = 16
ft_w  = np.frombuffer(b[o:o+IN*H], np.int8).reshape(IN, H);    o += IN*H
ft_b  = np.frombuffer(b[o:o+2*H], np.int16);                   o += 2*H
out_w = np.frombuffer(b[o:o+B*2*H], np.int8).reshape(B, 2*H);  o += B*2*H
out_b = np.frombuffer(b[o:o+4*B], np.int32)

# given active feature indices per perspective and the piece count
acc = lambda idx: np.clip(ft_b.astype(np.int32) + ft_w[idx].sum(0), 0, 127)
k   = min(max(pieces - 1, 0) * B // 32, B - 1)
total = int((np.concatenate([acc(us), acc(them)]) * out_w[k]).sum()) + int(out_b[k])
centipawns = int(total * 400 / (127 * 64))   # truncate toward zero
```

## Reproducing

```bash
cargo build --release
for i in $(seq 1 9); do
  ./target/release/sable <<< "datagen 400000 5000 $((i*7919))" > data/shard$i.txt &
done; wait

python train.py 10220706 15     # dumps features via the engine, writes net.bin
                                # NET_H=64 and OUT_SCALE=0.70 are the defaults
cargo build --release           # net.bin is include_bytes!'d into the binary
cp target/release/sable sable-std

# The network is embedded at compile time, so "no network" means building with
# a header the loader rejects; it then falls back to the hand-crafted eval.
cp net.bin /tmp/net.keep
printf '\0\0\0\0\0\0\0\0' > net.bin
cargo build --release && cp target/release/sable sable-hce
cp /tmp/net.keep net.bin && cargo build --release

python arena.py ./sable-std ./sable-hce 400 "nodes 20000" 9
```

The two comparison binaries are build artefacts, not repository contents —
`.gitignore` covers `sable-*` precisely so a stale one cannot be mistaken for
the current engine.

## Limitations

- Distilled from itself. The ceiling is the engine's own search quality rather
  than a stronger reference. Stockfish appears in this repository only as a
  measuring stick; nothing it plays has ever been trained on.
- The 2800 figure is an anchor, not a rating. Four Stockfish settings imply
  ratings spread across 140 Elo, and Stockfish's own `UCI_Elo` calibration is
  approximate and fitted at longer time controls than the 100ms used here.
- Computing mobility and king-attacker features costs throughput: the engine
  runs about **3.1 Mnps** at `bench 13` on one M-series core, and widening the
  hidden layer to 64 neurons cost 8% per node on its own. A direct-mapped cache
  of finished evaluations did most of it: the search asks about the same
  position often enough (transpositions, re-searches, null-move verification)
  that a good deal of the feature extraction was repeat work. The rest came from
  answering the pawn-structure questions for the whole board with file fills
  instead of pawn by pawn. The network build being the faster of the two is not
  a claim that a network is cheaper than a hand-crafted evaluator; it is that
  the cache and the extraction rewrite between them now more than cover the
  difference.
- Accumulators are refreshed in full rather than updated incrementally. At 64
  neurons a matrix row is eight NEON registers, and most of the 166 non-piece-
  square rows change on almost every move anyway, so an incremental update would
  only cover the piece-square part. The eval cache took the easy half of that win
  for a fraction of the complexity, and the refresh itself now keeps both
  perspectives in registers for the whole feature list rather than storing the
  accumulator back to memory once per row.

## Environmental impact

Rounding to something honest: about **six minutes** of Apple M-series GPU time
per training run, on hardware that draws roughly 20W doing this. Call it 1.4
gCO2eq — a gram and a half, less than boiling a mug of water. The 10.1M-position
dataset it trains on took considerably longer to generate than the network takes
to train, and the arena matches behind the Elo figures in this card dwarf both:
several tens of thousands of games at 20,000 nodes each. If you want the real
carbon cost of this project, it's in the measurement, not the training.

## Citation

```bibtex
@software{sable_chess_net,
  author  = {shubhxho},
  title   = {Sable: a 60 KB distilled chess evaluation network},
  year    = {2026},
  url     = {https://github.com/shubhxho/sable},
  note    = {Trained with MLX on Apple silicon; int8 quantisation-aware}
}
```

## Contact

Issues and questions: https://github.com/shubhxho/sable/issues

## License

MIT.
