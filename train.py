#!/usr/bin/env python3
"""Distil the engine's search into a 24 KB network, trained with MLX.

The teacher is the engine's own alpha-beta search: every training position
carries the score the search returned at fixed node count, plus the eventual
result of the self-play game it came from. The student is a static evaluation
function that never searches. This is the same trick DeepMind used to get a
searchless transformer to grandmaster strength -- distil a search's output into
a network -- at a scale that fits in L1 cache instead of a TPU pod.

Architecture:  768 -> 32 (per perspective, shared weights) -> clipped ReLU -> 1

Quantisation is baked into the training objective, not bolted on afterwards:
weights are clipped every step to the range int8 can represent, so the exported
network computes the same function the trainer converged to.
"""

import os
import struct
import sys
import time

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

H = int(os.environ.get("NET_H", "32"))   # hidden neurons per perspective
IN = 768               # 12 piece-square planes
MAX_FEATURES = 32      # a legal position has at most 32 pieces
QA, QB = 127, 64       # int8 scales for the two layers
SCALE = 400            # network units -> centipawns
EVAL_WEIGHT = float(os.environ.get("EVAL_W", "0.6"))  # teacher score vs game result
SIGMOID_K = float(os.environ.get("SIG_K", "400"))  # centipawns -> win probability

PIECE = {c: i for i, c in enumerate("PNBRQKpnbrqk")}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def parse(paths, limit=None):
    """FEN text -> (us_idx, them_idx, score, wdl, hce), all from the mover's view."""
    us_all, them_all, sc_all, wdl_all, hce_all = [], [], [], [], []
    n = 0
    t0 = time.time()
    for path in paths:
        with open(path, "rb") as fh:
            for line in fh:
                parts = line.split(b"|")
                if len(parts) != 4:
                    continue
                fen = parts[0].split()
                if len(fen) < 2:
                    continue
                board, stm = fen[0], fen[1]
                white = stm == b"w"
                try:
                    score = int(parts[1])
                    result = int(parts[2])
                    # The engine already reports this one from the mover's view.
                    hce = int(parts[3])
                except ValueError:
                    # A shard truncated mid-line, or a paste that ran short.
                    continue

                us = np.full(MAX_FEATURES, IN, dtype=np.int16)
                them = np.full(MAX_FEATURES, IN, dtype=np.int16)
                k = 0
                sq = 56
                for ch in board:
                    if ch == 0x2F:            # '/'
                        sq -= 16
                    elif 0x31 <= ch <= 0x38:  # '1'..'8'
                        sq += ch - 0x30
                    else:
                        p = PIECE[chr(ch)]
                        colour, pt = p // 6, p % 6
                        if k < MAX_FEATURES:
                            # Mirror the board for whichever side is "us".
                            if white:
                                us[k] = ((0 if colour == 0 else 1) * 6 + pt) * 64 + sq
                                them[k] = ((0 if colour == 1 else 1) * 6 + pt) * 64 + (sq ^ 56)
                            else:
                                us[k] = ((0 if colour == 1 else 1) * 6 + pt) * 64 + (sq ^ 56)
                                them[k] = ((0 if colour == 0 else 1) * 6 + pt) * 64 + sq
                            k += 1
                        sq += 1
                # Flip both labels to the side to move.
                if not white:
                    score = -score
                    result = 2 - result
                us_all.append(us)
                them_all.append(them)
                sc_all.append(score)
                wdl_all.append(result * 0.5)
                hce_all.append(hce)
                n += 1
                if limit and n >= limit:
                    break
        print(f"  {path}: {n} positions ({time.time()-t0:.0f}s)", flush=True)
        if limit and n >= limit:
            break
    return (
        np.stack(us_all),
        np.stack(them_all),
        np.array(sc_all, dtype=np.float32),
        np.array(wdl_all, dtype=np.float32),
        np.array(hce_all, dtype=np.float32),
    )


def load(paths, cache=None, limit=None):
    cache = cache or f"data/res_{limit or 'all'}.npz"
    if os.path.exists(cache):
        z = np.load(cache)
        print(f"cache hit: {len(z['sc'])} positions")
        return z["us"], z["them"], z["sc"], z["wdl"], z["hce"]
    us, them, sc, wdl, hce = parse(paths, limit)
    np.savez(cache, us=us, them=them, sc=sc, wdl=wdl, hce=hce)
    return us, them, sc, wdl, hce


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Net(nn.Module):
    """Row `IN` of `ft` is a permanent zero pad, so unused feature slots
    contribute nothing and every position can use a fixed-width index array."""

    def __init__(self):
        super().__init__()
        # Small init: the accumulator sums up to 32 rows, so per-weight scale
        # has to stay well under the clipped-ReLU ceiling of 1.
        self.ft = mx.random.normal((IN + 1, H)) * 0.02
        self.ft_b = mx.zeros((H,))
        self.out = mx.random.normal((2 * H,)) * 0.1
        self.out_b = mx.zeros((1,))

    def __call__(self, us, them):
        pad = mx.zeros((1, H))
        w = mx.concatenate([self.ft[:IN], pad], axis=0)
        acc_us = w[us].sum(axis=1) + self.ft_b
        acc_them = w[them].sum(axis=1) + self.ft_b
        a = mx.clip(mx.concatenate([acc_us, acc_them], axis=1), 0.0, 1.0)
        return (a * self.out).sum(axis=1) + self.out_b


def clip_weights(model):
    """Project back into the box int8 can represent. Doing this every step
    means the quantised network is the network that was trained, not an
    approximation of it."""
    model.ft = mx.clip(model.ft, -127.0 / QA, 127.0 / QA)
    model.out = mx.clip(model.out, -127.0 / QB, 127.0 / QB)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(model, path=None):
    path = path or os.environ.get("NET_OUT", "net.bin")
    ft = np.array(model.ft[:IN])
    ft_b = np.array(model.ft_b)
    out = np.array(model.out)
    out_b = float(np.array(model.out_b)[0])

    ft_q = np.clip(np.round(ft * QA), -127, 127).astype(np.int8)
    ft_b_q = np.clip(np.round(ft_b * QA), -32767, 32767).astype(np.int16)
    out_q = np.clip(np.round(out * QB), -127, 127).astype(np.int8)
    out_b_q = int(round(out_b * QA * QB))

    blob = struct.pack("<II", 0x4E4C4253, H)
    blob += ft_q.reshape(-1).tobytes()          # row-major [feature][neuron]
    blob += ft_b_q.tobytes()
    blob += out_q.tobytes()
    blob += struct.pack("<i", out_b_q)
    with open(path, "wb") as fh:
        fh.write(blob)
    print(f"wrote {path}: {len(blob)} bytes ({len(blob)/1024:.1f} KB)")
    return ft_q, ft_b_q, out_q, out_b_q


def quantised_eval(ft_q, ft_b_q, out_q, out_b_q, us, them):
    """Reference implementation of exactly what net.rs does, used to prove the
    Rust inference and the trainer agree. Returns the *correction* only."""
    acc_u = ft_b_q.astype(np.int32) + ft_q[us[us < IN]].sum(axis=0)
    acc_t = ft_b_q.astype(np.int32) + ft_q[them[them < IN]].sum(axis=0)
    a = np.concatenate([np.clip(acc_u, 0, QA), np.clip(acc_t, 0, QA)])
    total = int((a * out_q.astype(np.int32)).sum()) + out_b_q
    # Truncate toward zero, matching Rust's integer division. Python's `//`
    # floors, which differs by one on negative scores.
    return int(total * SCALE / (QA * QB))


# ---------------------------------------------------------------------------

def main():
    shards = sorted(
        os.path.join("data", f) for f in os.listdir("data") if f.endswith(".txt")
    )
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    us, them, sc, wdl, hce = load(shards, limit=limit)
    n = len(sc)
    print(f"{n} positions, {epochs} epochs")

    # Blend the teacher's score with the game result. The score is precise but
    # only as good as the search; the result is noisy but grounded in truth.
    target = EVAL_WEIGHT * (1.0 / (1.0 + np.exp(-sc / SIGMOID_K))) + (1 - EVAL_WEIGHT) * wdl
    target = target.astype(np.float32)

    us_mx = mx.array(us.astype(np.int32))
    them_mx = mx.array(them.astype(np.int32))
    y_mx = mx.array(target)
    # The hand-crafted evaluation enters the loss as a fixed offset, in the same
    # normalised units the network outputs. The network therefore only ever has
    # to learn the part the hand-crafted terms get wrong.
    hce_mx = mx.array(hce / SIGMOID_K)

    model = Net()
    mx.eval(model.parameters())
    batch = 16384
    steps_per_epoch = n // batch
    opt = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)

    def loss_fn(model, u, t, base, y):
        pred = base + model(u, t) * (SCALE / SIGMOID_K)
        return mx.mean((mx.sigmoid(pred) - y) ** 2)

    grad_fn = nn.value_and_grad(model, loss_fn)

    for ep in range(epochs):
        # Learning-rate decay; the last epochs are about settling into the
        # quantisation grid, not exploring.
        opt.learning_rate = 1e-2 * (0.75 ** ep)
        perm = np.random.permutation(n)
        total, t0 = 0.0, time.time()
        for i in range(steps_per_epoch):
            idx = mx.array(perm[i * batch : (i + 1) * batch])
            u = us_mx[idx]
            t = them_mx[idx]
            y = y_mx[idx]
            base = hce_mx[idx]
            loss, grads = grad_fn(model, u, t, base, y)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            clip_weights(model)
            total += float(loss)
        print(
            f"epoch {ep+1:2d}/{epochs}  loss {total/steps_per_epoch:.5f}  "
            f"lr {opt.learning_rate.item():.5f}  {time.time()-t0:.0f}s",
            flush=True,
        )

    ftq, fbq, oq, obq = export(model)

    # The number that matters: does hand-crafted + correction track the teacher
    # better than hand-crafted alone?
    k = min(6000, n)
    pick = np.random.choice(n, k, replace=False)
    corr = np.array([quantised_eval(ftq, fbq, oq, obq, us[i], them[i]) for i in pick])
    truth = sc[pick]
    base = hce[pick]
    for label, pred in (("hand-crafted alone", base), ("hand-crafted + net", base + corr)):
        r = np.corrcoef(pred, truth)[0, 1]
        mae = np.mean(np.abs(pred - truth))
        rmse = np.sqrt(np.mean((pred - truth) ** 2))
        print(f"  {label:20s} r={r:.4f}  mae={mae:5.1f}cp  rmse={rmse:5.1f}cp")


if __name__ == "__main__":
    main()
