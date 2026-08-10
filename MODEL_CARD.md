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

# Sable — a 24 KB distilled chess evaluation network

A quantised neural evaluation for the [Sable](https://github.com/shubhxho/sable)
chess engine. Trained with [MLX](https://github.com/ml-explore/mlx) on Apple
silicon and shipped as **24,716 bytes** of int8 weights, embedded directly into
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
- **Inference**: ARM NEON intrinsics (`vmovl_s8`, `vmlal_s16`, `vaddvq_s32`).

| Tensor | Shape | Type | Bytes |
|---|---|---|---|
| `ft_w` | 768 x 32 | int8 | 24,576 |
| `ft_b` | 32 | int16 | 64 |
| `out_w` | 64 | int8 | 64 |
| `out_b` | 1 | int32 | 4 |
| header | magic + hidden size | uint32 | 8 |
| | | **total** | **24,716** |

## The network is additive

It predicts a **correction** to a hand-crafted evaluation, not a replacement for
it. This was measured, not assumed.

| Setup | Result vs hand-crafted baseline |
|---|---|
| Network **replaces** hand-crafted eval | **-165 +/- 69 Elo** |
| Network **corrects** hand-crafted eval | **see repository** |

24 KB over plain piece-square features cannot represent mobility or king safety
— those depend on where pieces can *go*, not where they are. A replacement
network throws that knowledge away and has to rediscover it without the
representational capacity to do so. Predicting only the residual keeps
everything the hand-crafted terms already know and spends the whole parameter
budget on what they miss.

Fit against the teacher, measured on the quantised network:

| Predictor | r | MAE | RMSE |
|---|---|---|---|
| hand-crafted alone | 0.9420 | 95.3 cp | 188.4 cp |
| hand-crafted + network | 0.9551 | 93.5 cp | 167.3 cp |

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

## Format

Little-endian, tightly packed, no framework dependency:

```
magic  u32   0x4E4C4253 ("SBLN")
hidden u32   32
ft_w   i8[768 * 32]     row-major [feature][neuron]
ft_b   i16[32]
out_w  i8[64]           first 32 = side to move, last 32 = opponent
out_b  i32
```

Reference decode:

```python
import struct, numpy as np
b = open("net.bin", "rb").read()
magic, H = struct.unpack("<II", b[:8]); o = 8
ft_w = np.frombuffer(b[o:o+768*H], np.int8).reshape(768, H); o += 768*H
ft_b = np.frombuffer(b[o:o+2*H], np.int16);                  o += 2*H
out_w = np.frombuffer(b[o:o+2*H], np.int8);                  o += 2*H
out_b = struct.unpack("<i", b[o:o+4])[0]
```

Evaluation, given active feature indices for each perspective:

```python
acc  = lambda idx: np.clip(ft_b.astype(np.int32) + ft_w[idx].sum(0), 0, 127)
total = int((np.concatenate([acc(us), acc(them)]) * out_w).sum()) + out_b
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
