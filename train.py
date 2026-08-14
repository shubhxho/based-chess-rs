#!/usr/bin/env python3
"""Distil the engine's search into a 30 KB network, trained with MLX.

The teacher is the engine's own alpha-beta search. Every training position
carries the score that search returned at a fixed node count, plus the result of
the self-play game it came from. The student is a static evaluation that never
searches -- the idea behind DeepMind's searchless grandmaster-level chess, at a
size that fits in L1 cache rather than a TPU pod.

    934 -> 32 (per perspective, shared weights) -> clipped ReLU -> 1 of 8 buckets

Two things here are deliberate and load-bearing.

Features come from the engine, never from this file. `featdump` writes the
active indices for each position and we read them back. Re-deriving them in
Python would mean two implementations of one feature map, and when those drift
you get a network that loads, runs, and is quietly wrong -- the worst kind of
bug to chase.

Quantisation is part of the objective rather than a post-processing step.
Weights are projected back into the int8 box after every optimiser step, so the
exported network computes the function the trainer actually converged to.
"""

import os
import struct
import subprocess
import sys
import time

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

KB = int(os.environ.get("NET_KB", "1"))   # king buckets; must match net.rs
IN = 934 * KB            # feature rows; must match net.rs
MAX_F = 96               # feature slots per perspective
PAD = IN                 # index of the permanent zero row
H = int(os.environ.get("NET_H", "32"))
BUCKETS = int(os.environ.get("NET_B", "8"))
QA, QB = 127, 64         # int8 scales for the two layers
SCALE = 400              # network units -> centipawns
EVAL_WEIGHT = float(os.environ.get("EVAL_W", "0.9"))
WEIGHTED = os.environ.get("WEIGHTED", "1") != "0"
WEIGHT_CLAMP = float(os.environ.get("WEIGHT_CLAMP", "3"))
SHARD_DECAY = float(os.environ.get("SHARD_DECAY", "1.0"))
SIGMOID_K = float(os.environ.get("SIG_K", "400"))
OUT_SCALE = float(os.environ.get("OUT_SCALE", "0.7"))   # export-time gain; see export()
ENGINE = os.environ.get("ENGINE", "./target/release/sable")

FEAT_MAGIC = b"SBF2"     # featdump stream header


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def read_labels(paths, limit):
    """Pull FEN text and labels out of the augmented self-play shards.

    Positions are deduplicated across shards, later shard wins. The engine
    deduplicates within one generation run, but two runs months apart still
    rediscover the same openings, and in a mixed set the duplicate carries the
    *older*, weaker teacher's label. Keeping the last occurrence means a mixed
    set is the union of what each teacher saw with the better label on the
    overlap, rather than a set where the overlap is graded twice.

    Each shard also carries a provenance weight: with SHARD_DECAY below 1, older
    shards count for less, so a mixed set can lean on the newer teacher without
    throwing the older positions away.
    """
    seen = {}
    fens, sc, wdl, src = [], [], [], []
    for si, path in enumerate(paths):
        with open(path, "rb") as fh:
            for line in fh:
                parts = line.split(b"|")
                if len(parts) < 3:
                    continue
                try:
                    score = int(parts[1])
                    result = int(parts[2])
                except ValueError:
                    continue          # shard truncated mid-line
                fen = parts[0].strip()
                if len(fen.split()) < 2:
                    continue
                white = fen.split()[1] == b"w"
                # Everything is stored from the mover's point of view.
                if not white:
                    score = -score
                    result = 2 - result
                prev = seen.get(fen)
                if prev is None:
                    seen[fen] = len(fens)
                    fens.append(fen)
                    sc.append(score)
                    wdl.append(result * 0.5)
                    src.append(si)
                else:
                    # Later shard wins: newer teacher, better label.
                    sc[prev] = score
                    wdl[prev] = result * 0.5
                    src[prev] = si
                if limit and len(fens) >= limit:
                    break
        print(f"  {path}: {len(fens)} unique positions", flush=True)
        if limit and len(fens) >= limit:
            break
    n_shards = max(src) + 1 if src else 1
    age = np.array([SHARD_DECAY ** (n_shards - 1 - i) for i in src], np.float32)
    return fens, np.array(sc, np.float32), np.array(wdl, np.float32), age


def parse_features(path, n_expected):
    """Unpack a featdump stream into padded index arrays.

    The stream is self-describing: a header, then one variable-length record per
    position. Records are packed rather than padded to MAX_F, so the array width
    here is the widest position actually seen -- typically well under the
    ninety-six slots the engine allows for, which is the difference between the
    feature cache fitting in memory and not.
    """
    with open(path, "rb") as fh:
        head = fh.read(8)
    if len(head) < 8 or head[:4] != FEAT_MAGIC:
        return None
    n_in, max_f = struct.unpack("<HH", head[4:])
    if n_in != IN or max_f != MAX_F:
        raise SystemExit(f"featdump says IN={n_in} MAX_F={max_f}, trainer says {IN} {MAX_F}")

    raw = np.fromfile(path, dtype="<u2", offset=8)

    # Walk the record starts once. Each length is only knowable after the
    # previous record has been stepped over, so this is the one loop in the
    # function. It reads through a memoryview rather than indexing the array:
    # `raw[pos]` builds a NumPy scalar object per word, `mv[pos]` hands back a
    # plain int, and over millions of records that is most of the walk.
    mv = memoryview(raw)
    counts = np.empty(n_expected, np.int32)
    starts = np.empty(n_expected, np.int64)
    pos, k, total = 0, 0, len(raw)
    while pos < total and k < n_expected:
        n = mv[pos]
        counts[k] = n
        starts[k] = pos + 1
        pos += 2 * n + 2          # n, us[n], them[n], bucket
        k += 1
    if k != n_expected or pos != total:
        raise SystemExit(f"featdump returned {k} records for {n_expected} positions")

    width = int(counts.max())
    # uint16 because the widest index is IN, and these two arrays are the
    # largest thing the trainer holds -- int32 doubled the resident set for
    # nothing. Batches are widened on their way to MLX.
    us = np.full((n_expected, width), PAD, np.uint16)
    them = np.full((n_expected, width), PAD, np.uint16)

    # Scatter in blocks. The index temporaries are one entry per *active
    # feature*, so doing the whole set at once allocates several gigabytes of
    # int64 to fill an array a fraction of that size; a block at a time keeps
    # them at a few megabytes and runs no slower.
    block = 1 << 18
    for lo in range(0, n_expected, block):
        hi = min(lo + block, n_expected)
        c = counts[lo:hi].astype(np.int64)
        ends = np.cumsum(c)
        # Ragged arange: 0..n-1 within each record, concatenated.
        offsets = np.arange(int(ends[-1]), dtype=np.int64) - np.repeat(ends - c, c)
        rows = np.repeat(np.arange(hi - lo, dtype=np.int64), c)
        base = np.repeat(starts[lo:hi], c)
        us[lo:hi][rows, offsets] = raw[base + offsets]
        them[lo:hi][rows, offsets] = raw[base + np.repeat(c, c) + offsets]

    buckets = raw[starts + 2 * counts].astype(np.int32)
    return us, them, buckets, width


def dump_features(fens, cache):
    """Ask the engine for the feature indices of every position."""
    if os.path.exists(cache):
        parsed = parse_features(cache, len(fens))
        if parsed is not None:
            print(f"cache hit: {cache}")
            return parsed
        print("cache stale, regenerating")

    cmds = os.path.join(os.path.dirname(cache) or ".", "_fens.txt")
    t0 = time.time()
    with open(cmds, "wb") as fh:
        fh.write(b"featdump\n")
        for f in fens:
            fh.write(f + b"\n")
    with open(cmds, "rb") as stdin, open(cache, "wb") as stdout:
        subprocess.run([ENGINE], stdin=stdin, stdout=stdout, check=True)
    os.remove(cmds)

    parsed = parse_features(cache, len(fens))
    if parsed is None:
        raise SystemExit("featdump produced no usable header")
    print(f"  featdump: {len(fens)} positions in {time.time()-t0:.0f}s")
    return parsed


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Net(nn.Module):
    """Row `IN` of `ft` is a permanent zero pad, so unused feature slots
    contribute nothing and every position can use one fixed-width index array.

    The output layer is one of `BUCKETS` sets, picked by remaining material. The
    feature transformer stays shared: what changes over a game is how the same
    positional signals should be weighed, not what they are.
    """

    def __init__(self):
        super().__init__()
        # The accumulator sums ~80 rows, so per-weight scale has to stay well
        # under the clipped-ReLU ceiling of 1.
        self.ft = mx.random.normal((IN, H)) * 0.015
        self.ft_b = mx.zeros((H,))
        self.out = mx.random.normal((BUCKETS, 2 * H)) * 0.1
        self.out_b = mx.zeros((BUCKETS,))

    def __call__(self, us, them, bucket):
        w = mx.concatenate([self.ft, mx.zeros((1, H))], axis=0)
        acc_us = w[us].sum(axis=1) + self.ft_b
        acc_them = w[them].sum(axis=1) + self.ft_b
        a = mx.clip(mx.concatenate([acc_us, acc_them], axis=1), 0.0, 1.0)
        return (a * self.out[bucket]).sum(axis=1) + self.out_b[bucket]


def clip_weights(model):
    """Project back into the box int8 can represent."""
    model.ft = mx.clip(model.ft, -127.0 / QA, 127.0 / QA)
    model.out = mx.clip(model.out, -127.0 / QB, 127.0 / QB)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(model, path=None):
    """Quantise and write the network, quieter than it was trained.

    OUT_SCALE multiplies the whole output layer, so every evaluation comes back
    smaller by the same factor and the ranking of positions is untouched. It is
    the last thing applied and it is not part of the objective, because it is
    not about fitting the teacher: a network trained to reproduce the search's
    score reproduces its *spread* too, and the search plays worse when handed
    one. Measured against the same network exported at several gains, all else
    equal, over 1000 games each at 20,000 nodes -- see MODEL_CARD.md.

    Scaling the exported weights rather than the score in net.rs keeps the
    network file the single description of what the engine computes, so the
    trainer's own quantised reference still agrees with the engine byte for
    byte.
    """
    path = path or os.environ.get("NET_OUT", "net.bin")
    ft = np.array(model.ft)
    ft_b = np.array(model.ft_b)
    out = np.array(model.out) * OUT_SCALE
    out_b = np.array(model.out_b) * OUT_SCALE

    ft_q = np.clip(np.round(ft * QA), -127, 127).astype(np.int8)
    ft_b_q = np.clip(np.round(ft_b * QA), -32767, 32767).astype(np.int16)
    out_q = np.clip(np.round(out * QB), -127, 127).astype(np.int8)
    out_b_q = np.round(out_b * QA * QB).astype(np.int32)

    blob = struct.pack("<IIII", 0x334C4253, IN, H, BUCKETS)  # "SBL3"
    blob += ft_q.reshape(-1).tobytes()      # row-major [feature][neuron]
    blob += ft_b_q.tobytes()
    blob += out_q.reshape(-1).tobytes()     # row-major [bucket][neuron]
    blob += out_b_q.tobytes()
    with open(path, "wb") as fh:
        fh.write(blob)
    print(f"wrote {path}: {len(blob)} bytes ({len(blob)/1024:.1f} KB)")
    return ft_q, ft_b_q, out_q, out_b_q


def quantised_eval(ft_q, ft_b_q, out_q, out_b_q, us, them, bucket):
    """Exactly what net.rs does, in NumPy. Used to prove the two agree."""
    acc = lambda idx: np.clip(
        ft_b_q.astype(np.int32) + ft_q[idx[idx < IN]].sum(axis=0), 0, QA
    )
    a = np.concatenate([acc(us), acc(them)])
    total = int((a * out_q[bucket].astype(np.int32)).sum()) + int(out_b_q[bucket])
    # Truncate toward zero, matching Rust integer division. Python's `//` floors,
    # which differs by one on negative scores.
    return int(total * SCALE / (QA * QB))


# ---------------------------------------------------------------------------

def main():
    shards = sorted(
        os.path.join("data", f)
        for f in os.listdir("data")
        if f.startswith("aug") and f.endswith(".txt")
    )
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 15

    seed = int(os.environ.get("SEED", "42"))
    np.random.seed(seed)
    mx.random.seed(seed)

    fens, sc, wdl, age = read_labels(shards, limit)
    # The feature map is part of the cache key. Changing IN changes what every
    # index means, and a cache named only for the position count would be read
    # back as indices into a table that no longer exists.
    us, them, buckets, width = dump_features(fens, f"data/feat_{limit or 'all'}_{IN}.bin")
    n = len(sc)
    print(
        f"{n} positions, {epochs} epochs, {(us.nbytes + them.nbytes)/1e6:.0f} MB of "
        f"features at width {width} of {MAX_F}, seed {seed}"
    )

    # Held-out positions never touched by the optimiser. The exported network
    # is the epoch that did best here, not whatever the last epoch happened
    # to land on.
    val_n = min(200_000, n // 20)
    val_idx = np.random.permutation(n)[:val_n]
    train_mask = np.ones(n, bool)
    train_mask[val_idx] = False
    train_idx = np.flatnonzero(train_mask)

    # Blend the teacher's score with the game result. The score is precise but
    # only as good as the search; the result is noisy but grounded in truth.
    target = (
        EVAL_WEIGHT / (1.0 + np.exp(-sc / SIGMOID_K)) + (1 - EVAL_WEIGHT) * wdl
    ).astype(np.float32)

    # Per-position weights. Self-play spends most of its plies in the middlegame,
    # so the material buckets are far from evenly filled and an unweighted mean
    # quietly trains the crowded buckets at the expense of the sparse ones. The
    # correction is inverse bucket frequency, clamped: bucket balance is worth
    # nudging, not worth letting a handful of endgame positions dominate.
    counts = np.bincount(buckets, minlength=BUCKETS).astype(np.float64)
    share = np.where(counts > 0, counts.sum() / (BUCKETS * np.maximum(counts, 1)), 1.0)
    share = np.clip(share, 1.0 / WEIGHT_CLAMP, WEIGHT_CLAMP)
    weight = (share[buckets] * age).astype(np.float32)
    if not WEIGHTED:
        weight = age.astype(np.float32)
    print(
        "  bucket counts " + " ".join(f"{int(c)}" for c in counts) + "\n"
        "  weights       " + " ".join(f"{w:.2f}" for w in share)
    )

    model = Net()
    mx.eval(model.parameters())
    batch = 16384
    steps = len(train_idx) // batch
    base_lr = float(os.environ.get("LR", "1e-2"))
    opt = optim.AdamW(learning_rate=base_lr, weight_decay=0.0)

    def loss_fn(model, u, t, bk, y, w):
        # SCALE / SIGMOID_K converts network units into the sigmoid's argument.
        err = (mx.sigmoid(model(u, t, bk) * (SCALE / SIGMOID_K)) - y) ** 2
        return mx.sum(err * w) / mx.sum(w)

    grad_fn = nn.value_and_grad(model, loss_fn)

    def val_loss():
        total, m = 0.0, 0
        for i in range(0, val_n, batch):
            idx = val_idx[i : i + batch]
            total += float(
                loss_fn(
                    model,
                    mx.array(us[idx].astype(np.int32)),
                    mx.array(them[idx].astype(np.int32)),
                    mx.array(buckets[idx]),
                    mx.array(target[idx]),
                    mx.array(weight[idx]),
                )
            ) * len(idx)
            m += len(idx)
        return total / m

    best = (float("inf"), None)
    warmup = 1
    for ep in range(epochs):
        # One linear warmup epoch, then cosine to ~0: early steps explore,
        # late steps settle into the quantisation grid.
        if ep < warmup:
            opt.learning_rate = base_lr * (ep + 1) / warmup
        else:
            import math
            t = (ep - warmup) / max(1, epochs - warmup - 1)
            opt.learning_rate = base_lr * 0.5 * (1 + math.cos(math.pi * t))
        perm = train_idx[np.random.permutation(len(train_idx))]
        total, t0 = 0.0, time.time()
        for i in range(steps):
            idx = perm[i * batch : (i + 1) * batch]
            loss, grads = grad_fn(
                model,
                mx.array(us[idx].astype(np.int32)),
                mx.array(them[idx].astype(np.int32)),
                mx.array(buckets[idx]),
                mx.array(target[idx]),
                mx.array(weight[idx]),
            )
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            clip_weights(model)
            total += float(loss)
        vl = val_loss()
        star = ""
        if vl < best[0]:
            best = (vl, [mx.array(p) for p in (model.ft, model.ft_b, model.out, model.out_b)])
            star = " *"
        print(
            f"epoch {ep+1:2d}/{epochs}  loss {total/steps:.5f}  val {vl:.5f}{star}  "
            f"lr {opt.learning_rate.item():.5f}  {time.time()-t0:.0f}s",
            flush=True,
        )

    if best[1] is not None:
        model.ft, model.ft_b, model.out, model.out_b = best[1]
        print(f"exporting best-val checkpoint (val {best[0]:.5f})")
    ftq, fbq, oq, obq = export(model)

    # How well does the quantised network track the teacher it was distilled
    # from? Reported on the quantised weights, since those are what ship.
    # Reported on held-out positions: this is generalisation, not memorisation.
    k = min(6000, val_n)
    pick = np.random.choice(val_idx, k, replace=False)
    pred = np.array(
        [quantised_eval(ftq, fbq, oq, obq, us[i], them[i], buckets[i]) for i in pick]
    )
    truth = sc[pick]
    r = np.corrcoef(pred, truth)[0, 1]
    print(
        f"  quantised net vs teacher: r={r:.4f}  "
        f"mae={np.mean(np.abs(pred-truth)):5.1f}cp  "
        f"rmse={np.sqrt(np.mean((pred-truth)**2)):5.1f}cp"
    )


if __name__ == "__main__":
    main()
