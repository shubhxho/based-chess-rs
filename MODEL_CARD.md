---
license: mit
tags:
  - chess
  - nnue
  - mlx
  - quantization
  - int8
  - distillation
library_name: mlx
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

## The input set is the whole story

The obvious design uses the standard NNUE input: 768 binary features, one per
(piece, colour, square). Built that way, at this size, the network plays **165
Elo worse** than the hand-crafted evaluator it replaces.

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

Piece-square features describe where pieces **are**. Almost everything that
decides a chess position is about where they can **go**. A knight's value swings
wildly with what it attacks; a rook's with whether its file is open. Neither is
recoverable from a one-hot square index at any width.

So the budget went into the input. Alongside the 768 piece-square planes sit 166
rows encoding mobility, passed pawns by rank, isolated and doubled pawns, rooks
on open and half-open files, the bishop pair, king attackers and king shelter.
Each row costs 64 bytes.

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

The trainer does not compute features. It asks the engine for them through a
`featdump` command that emits the active indices per position. Two
implementations of one feature map is a bug class that yields a network which
loads, runs, and is quietly wrong — very hard to find afterwards. `src/net.rs`
is the single source of truth for both training and inference.

### Quantisation-aware by construction

Weights are projected back into the int8 box **after every optimiser step**,
never rounded at the end:

```python
model.ft = mx.clip(model.ft, -127.0 / QA, 127.0 / QA)
model.out = mx.clip(model.out, -127.0 / QB, 127.0 / QB)
```

So the exported network computes the function the trainer converged to. Verified,
not asserted: `net.bin` is replayed through an independent NumPy reference that
reproduces the Rust inference operation for operation, and the two agree on
80/80 test positions. The only disagreement ever seen was Python's floor
division against Rust's truncation on negative scores — a bug in the reference.

## Format

Little-endian, tightly packed, no framework dependency:

```
magic   u32   0x334C4253 ("SBL3")
inputs  u32   934
hidden  u32   32
buckets u32   8
ft_w    i8[934 * 32]     row-major [feature][neuron]
ft_b    i16[32]
out_w   i8[8 * 64]       row-major [bucket][neuron];
                         within a bucket, first 32 = side to move,
                         last 32 = opponent
out_b   i32[8]
```

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

python train.py 3400000 16      # dumps features via the engine, writes net.bin
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

- Distilled from itself. With no external engine available, the ceiling is the
  teacher's own search quality rather than a stronger reference.
- Computing mobility and king-attacker features used to cost real throughput:
  2.5 Mnps against 3.1 for the hand-crafted evaluator on the same core. That gap
  is now closed and slightly reversed — `bench 13`, median of five runs, is
  **3.37 Mnps with the network against 3.25 without it**. A direct-mapped cache
  of finished evaluations did most of it: the search asks about the same
  position often enough (transpositions, re-searches, null-move verification)
  that a good deal of the feature extraction was repeat work. The rest came from
  answering the pawn-structure questions for the whole board with file fills
  instead of pawn by pawn. The network build being the faster of the two is not
  a claim that a network is cheaper than a hand-crafted evaluator; it is that
  the cache and the extraction rewrite between them now more than cover the
  difference.
- Accumulators are refreshed in full rather than updated incrementally. At 32
  neurons a matrix row is four NEON registers, and most of the 166 non-piece-
  square rows change on almost every move anyway, so an incremental update would
  only cover the piece-square part. The eval cache took the easy half of that win
  for a fraction of the complexity, and the refresh itself now keeps both
  perspectives in registers for the whole feature list rather than storing the
  accumulator back to memory once per row.

## License

MIT.
