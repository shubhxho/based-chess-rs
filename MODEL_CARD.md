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

# Sable — a 25 KB distilled chess evaluation network

A quantised neural evaluation for the [Sable](https://github.com/shubhxho/sable)
chess engine. Trained with [MLX](https://github.com/ml-explore/mlx) on Apple
silicon and shipped as **25,196 bytes** of int8 weights, embedded directly into
a 248 KB `no_std` Rust engine binary.

## Architecture

```
768 inputs  ->  32 hidden (per perspective, shared weights)  ->  clipped ReLU  ->  1
```

- **Inputs**: 12 piece-square planes (`piece_colour x piece_type x square`),
  built twice per position — once from each side's point of view, with the board
  vertically mirrored for the black perspective. One weight matrix serves both,
  so the network learns a single function of "my position" rather than two of
  "white's position".
- **Weights**: int8 feature transformer (`QA = 127`), int16 biases, int8 output
  layer (`QB = 64`). Output scales to centipawns by `SCALE = 400`.
- **Output buckets**: 8 output layers, selected by remaining material. The
  feature transformer stays shared — what changes across a game is how the same
  positional signals should be weighed, not what they are.
- **Inference**: ARM NEON intrinsics (`vmovl_s8`, `vmlal_s16`, `vaddvq_s32`).

### Index formulas

Both must be reproduced exactly, integer division included, or the network
reads the wrong weights.

```python
# Feature index, for perspective `p` (0 = white), piece colour `c`,
# piece type `t` (0..5 = P,N,B,R,Q,K), square `sq` (0 = a1, 63 = h8)
feature = ((0 if c == p else 1) * 6 + t) * 64 + (sq if p == 0 else sq ^ 56)

# Output bucket, from the total number of pieces on the board
bucket = min((max(pieces - 1, 0) * 8) // 32, 7)
```

| Tensor | Shape | Type | Bytes |
|---|---|---|---|
| `ft_w` | 768 x 32 | int8 | 24,576 |
| `ft_b` | 32 | int16 | 64 |
| `out_w` | 8 x 64 | int8 | 512 |
| `out_b` | 8 | int32 | 32 |
| header | magic, hidden, buckets | uint32 | 12 |
| | | **total** | **25,196** |

## The network is additive

It predicts a **correction** to a hand-crafted evaluation, not a replacement for
it. This was measured, not assumed.

| Setup | Size | Result vs hand-crafted baseline |
|---|---|---|
| Network **replaces** hand-crafted eval | 24.1 KB | **-165 +/- 69 Elo** (200 games) |
| Network **corrects** hand-crafted eval | 24.1 KB | **+55 +/- 31 Elo** (500 games) |
| ... with material-bucketed output | 24.6 KB | **+57 +/- 28 Elo** (600 games) |

All matches at a fixed 20,000 nodes per move, so the comparison is independent
of machine load, with colours swapped on every opening pair.

Worth being precise about the last row: bucketing measurably improves the fit
against the teacher (RMSE -15.8% vs -11.2%), but +57 +/- 28 and +55 +/- 31 are
statistically indistinguishable in games. The bucketed net is kept because the
regression result is unambiguous and it costs 480 bytes; the Elo difference
between the two is not something these match lengths can resolve.

24 KB over plain piece-square features cannot represent mobility or king safety
— those depend on where pieces can *go*, not where they are. A replacement
network throws that knowledge away and has to rediscover it without the
representational capacity to do so. Predicting only the residual keeps
everything the hand-crafted terms already know and spends the whole parameter
budget on what they miss.

Fit against the teacher, measured on the quantised network:

| Predictor | r | MAE | RMSE | RMSE vs baseline |
|---|---|---|---|---|
| hand-crafted alone | 0.9369 | 96.3 cp | 191.6 cp | — |
| + network, single output | 0.9551 | 93.5 cp | 167.3 cp | -11.2% |
| + network, 8 output buckets | 0.9553 | 90.3 cp | 161.3 cp | **-15.8%** |

Hidden width was swept from 16 to 128 neurons; the fit plateaus at every width.
The bottleneck is the feature set, not capacity — which is exactly why the
extra 480 bytes went into output buckets rather than into wider hidden layers.

## Training

The teacher is the engine's **own alpha-beta search** — the same distillation
principle behind DeepMind's searchless grandmaster-level chess, at a scale that
fits in L1 cache rather than a TPU pod. The student never searches.

- **Data**: 3.36M positions from engine self-play out of randomised openings,
  in two iterations (3.0M at 3k nodes/move, then 0.35M at 5k nodes/move from the
  stronger net-equipped engine).
- **Filtering**: positions are dropped when the side to move is in check or the
  best move is a capture. In those positions the tactic decides the game, not
  the static evaluation, and training on them teaches the network to imitate
  search — which it cannot do.
- **Objective**: MSE in win-probability space,
  `sigmoid((hce + net) / 400)` against `0.9 * sigmoid(search / 400) + 0.1 * result`.
- **Optimiser**: AdamW, batch 16384, 18 epochs, lr 1e-2 decayed 0.75x per epoch.

### Quantisation-aware by construction

Weights are projected back into the int8 box **after every optimiser step**, not
rounded at the end:

```python
model.ft = mx.clip(model.ft, -127.0 / QA, 127.0 / QA)
model.out = mx.clip(model.out, -127.0 / QB, 127.0 / QB)
```

So the exported network computes exactly the function the trainer converged to.
This is verified rather than asserted: `net.bin` is replayed through an
independent NumPy reference reproducing the Rust inference operation for
operation, and the two agree on 60/60 test positions (the only initial
disagreement was Python's floor division vs Rust's truncation on negative
scores — a bug in the reference, not the engine).

## Reproducing

```bash
cargo build --release
for i in $(seq 1 9); do
  ./target/release/sable <<< "datagen 400000 5000 $((i*7919))" > data/shard$i.txt &
done; wait

# label each position with the hand-crafted eval the network must correct
awk -F'|' '{print "position fen "$1"\nevalhce"}' data/shard1.txt \
  | ./sable-hce | awk '{print $2}' > /tmp/e1.txt
paste -d'|' data/shard1.txt /tmp/e1.txt > data/aug1.txt

python train.py 4000000 18      # writes net.bin
cargo build --release           # net.bin is include_bytes!'d into the binary
python arena.py ./sable-net ./sable-hce 600 "nodes 20000" 9
```

## Format

Little-endian, tightly packed, no framework dependency:

```
magic   u32   0x324C4253 ("SBL2")
hidden  u32   32
buckets u32   8
ft_w    i8[768 * 32]      row-major [feature][neuron]
ft_b    i16[32]
out_w   i8[8 * 64]        row-major [bucket][neuron];
                          within a bucket, first 32 = side to move,
                          last 32 = opponent
out_b   i32[8]
```

Reference decode:

```python
import struct, numpy as np
b = open("net.bin", "rb").read()
magic, H, B = struct.unpack("<III", b[:12]); o = 12
ft_w  = np.frombuffer(b[o:o+768*H], np.int8).reshape(768, H);  o += 768*H
ft_b  = np.frombuffer(b[o:o+2*H], np.int16);                   o += 2*H
out_w = np.frombuffer(b[o:o+B*2*H], np.int8).reshape(B, 2*H);  o += B*2*H
out_b = np.frombuffer(b[o:o+4*B], np.int32)
```

Evaluation, given active feature indices for each perspective:

```python
acc = lambda idx: np.clip(ft_b.astype(np.int32) + ft_w[idx].sum(0), 0, 127)
k   = min((max(pieces - 1, 0) * B) // 32, B - 1)
total = int((np.concatenate([acc(us), acc(them)]) * out_w[k]).sum()) + int(out_b[k])
centipawns = int(total * 400 / (127 * 64))   # truncate toward zero
```

## Limitations

- Trained by self-distillation. With no external engine available, the ceiling
  is set by the teacher's own search quality, not by a stronger reference.
- The 768-feature representation plateaus around r = 0.97 against the teacher
  regardless of hidden width — tested from 16 to 128 neurons. The bottleneck is
  the feature set, not capacity. King-bucketed features would lift it, but do
  not fit the size budget.
- Accumulators are refreshed in full rather than updated incrementally, costing
  roughly 15% nps. At 32 neurons the whole matrix row is four NEON registers, so
  the bookkeeping for incremental updates is not obviously worth it.

## License

MIT.
