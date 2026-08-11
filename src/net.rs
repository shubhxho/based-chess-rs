//! The evaluation network.
//!
//! This is the whole evaluation. There is no hand-crafted term underneath it.
//!
//! Getting here took one failed design worth recording. A network over the
//! usual 768 piece-square inputs, at this size, plays about 165 Elo *worse*
//! than a hand-crafted evaluator. Widening it does not help: the fit against
//! the teacher plateaus at essentially the same place for 16, 32, 64 and 128
//! neurons. Capacity was never the problem. Piece-square features describe
//! where pieces *are*, and most of what decides a chess position — mobility,
//! king safety, passed pawns — is about where pieces can *go*. That is not in
//! the input, so no amount of width recovers it.
//!
//! So the remaining budget went into the input rather than the hidden layer.
//! Alongside the 768 piece-square planes sit 166 rows encoding mobility,
//! passed pawns, pawn structure, rook files, the bishop pair, king attackers
//! and king shelter — computed from the board and looked up in the same
//! embedding table. Each row costs 32 bytes. The whole network is 30,508.
//!
//! This file is the single source of truth for feature extraction. The trainer
//! never re-implements it; it asks the engine for indices through `featdump`.
//! A trainer that disagrees with the engine about what feature 431 means is a
//! bug that yields a plausible-looking network which quietly plays badly, and
//! it is miserable to find after the fact.

use crate::bb::*;
use crate::pos::*;
use crate::sys::SyncCell;

const BLOB: &[u8] = include_bytes!("../net.bin");
const MAGIC: usize = 0x334C_4253; // "SBL3" little-endian

/// Hidden neurons per perspective.
pub const H: usize = 32;
/// Output-layer sets, indexed by remaining material.
pub const BUCKETS: usize = 8;

// --- feature-space layout; each constant is the first row of its block
const PSQ: usize = 0; //         768 rows: (rel_colour, piece, square)
const MOB: usize = 768; //        96 rows: (rel_colour, N/B/R/Q, mobility 0..11)
const PASSED: usize = 864; //     16 rows: (rel_colour, rank)
const ISOLATED: usize = 880; //    8 rows: (rel_colour, count 0..3)
const DOUBLED: usize = 888; //     8 rows
const ROOK_OPEN: usize = 896; //   6 rows: (rel_colour, count 0..2)
const ROOK_SEMI: usize = 902; //   6 rows
const PAIR: usize = 908; //        2 rows
const KING_ATT: usize = 910; //   16 rows: (rel_colour, attackers 0..7)
const SHELTER: usize = 926; //     8 rows: (rel_colour, pawns 0..3)
pub const IN: usize = 934;

/// Upper bound on simultaneously active features. A normal position reaches
/// roughly 80; the slack absorbs promotion-heavy positions.
pub const MAX_F: usize = 96;

/// Quantisation scales; the trainer applies the same ones.
const QA: i32 = 127; // feature-transformer / activation range
const QB: i32 = 64; // output weights
const SCALE: i32 = 400; // network units -> centipawns

struct Net {
    ft_w: [i8; IN * H],
    ft_b: [i16; H],
    out_w: [i8; BUCKETS * 2 * H],
    out_b: [i32; BUCKETS],
    loaded: bool,
}

static NET: SyncCell<Net> = SyncCell::new(Net {
    ft_w: [0; IN * H],
    ft_b: [0; H],
    out_w: [0; BUCKETS * 2 * H],
    out_b: [0; BUCKETS],
    loaded: false,
});

#[inline(always)]
fn net() -> &'static Net {
    unsafe { NET.as_ref() }
}

/// Direct-mapped cache of finished evaluations.
///
/// The network is a pure function of the position, and a search asks about the
/// same position many times over: transpositions, the re-search after a
/// fail-high, null-move verification, and the static evaluation taken at a node
/// that a later iteration visits again. Extracting eighty features and running
/// two accumulations to rediscover a number computed a microsecond ago is most
/// of what the evaluator does.
///
/// One `u64` per slot, packed as `tag:40 | generation:8 | score:16`. The index
/// is the low bits of the key and the tag is bits 24 and up, so the two never
/// overlap: a slot only answers for a position that agrees on both. Scores are
/// clamped to ±20,000, so sixteen bits hold one exactly.
///
/// The generation is what makes the table cheap to empty. The first version
/// zeroed all of it, and that memset was large enough to decide the sizing:
/// 18-bit and 20-bit tables both measured *slower* than 16-bit, because the
/// clear between bench positions cost more than the extra hits were worth.
/// Bumping a counter invalidates every entry at once, so the size question is
/// now about cache footprint alone.
const CACHE_BITS: usize = 16;
static CACHE: SyncCell<[u64; 1 << CACHE_BITS]> = SyncCell::new([0; 1 << CACHE_BITS]);
static GEN: SyncCell<u8> = SyncCell::new(0);

#[inline(always)]
fn pack(key: u64, gen: u8, score: i32) -> u64 {
    (key >> 24 << 24) | ((gen as u64) << 16) | (score as i16 as u16 as u64)
}

/// Not needed for correctness — a cached score is as valid as the day it was
/// stored — but `bench` and datagen want each position measured from a cold
/// start, and the search's own `clear` is where that is expressed.
pub fn clear_cache() {
    let g = unsafe { GEN.as_mut() };
    *g = g.wrapping_add(1);
    // Eight bits of generation wrap after 256 clears, and an entry that old
    // would start answering again. That only happens once every 256 clears, so
    // pay for the real erase then.
    if *g == 0 {
        for e in unsafe { CACHE.as_mut() }.iter_mut() {
            *e = 0;
        }
    }
}

#[inline(always)]
pub fn is_loaded() -> bool {
    net().loaded
}

/// Expected layout, little-endian, tightly packed:
///   magic u32 | inputs u32 | hidden u32 | buckets u32
///   | ft_w i8[IN*H] | ft_b i16[H] | out_w i8[BUCKETS*2H] | out_b i32[BUCKETS]
///
/// A header mismatch is not an error — it just leaves the network unloaded and
/// the engine falls back to the hand-crafted evaluation, so a half-built tree
/// still produces a playable binary.
pub fn init() {
    let need = 16 + IN * H + 2 * H + BUCKETS * 2 * H + BUCKETS * 4;
    if BLOB.len() < need {
        return;
    }
    let rd32 = |o: usize| u32::from_le_bytes([BLOB[o], BLOB[o + 1], BLOB[o + 2], BLOB[o + 3]]) as usize;
    if rd32(0) != MAGIC || rd32(4) != IN || rd32(8) != H || rd32(12) != BUCKETS {
        return;
    }
    let n = unsafe { NET.as_mut() };
    let mut o = 16;
    for i in 0..IN * H {
        n.ft_w[i] = BLOB[o + i] as i8;
    }
    o += IN * H;
    for i in 0..H {
        n.ft_b[i] = i16::from_le_bytes([BLOB[o + 2 * i], BLOB[o + 2 * i + 1]]);
    }
    o += 2 * H;
    for i in 0..BUCKETS * 2 * H {
        n.out_w[i] = BLOB[o + i] as i8;
    }
    o += BUCKETS * 2 * H;
    for i in 0..BUCKETS {
        n.out_b[i] = i32::from_le_bytes([
            BLOB[o + 4 * i],
            BLOB[o + 4 * i + 1],
            BLOB[o + 4 * i + 2],
            BLOB[o + 4 * i + 3],
        ]);
    }
    n.loaded = true;
}

// ---------------------------------------------------------------------------
// Feature extraction
// ---------------------------------------------------------------------------

/// The active feature indices for a position, from **both** perspectives at
/// once. `a` receives `persp`'s view, `b` receives the opponent's.
///
/// Both views describe the same board; only the index arithmetic differs
/// (whose pieces count as "mine", and whether squares are mirrored). Computing
/// the expensive part — mobility, king attackers, pawn structure — once and
/// emitting two indices from it measured between 3% and 13% more nodes per
/// second across repeated runs, against walking the board twice.
///
/// Colours are relative: block 0 is always "mine", block 1 always "theirs", and
/// squares are mirrored for black. One weight matrix therefore serves both
/// sides, and the network learns a single function of "my position" rather than
/// two functions of "white's position".
pub fn features_both(pos: &Position, persp: usize, a: &mut [u16; MAX_F], b: &mut [u16; MAX_F]) -> usize {
    let mut n = 0usize;
    let occ = pos.occ();

    // Every knight, bishop, rook and queen is asked for its attack set exactly
    // once. Mobility wants it for the piece's own colour and king safety wants
    // the same board from the other side, so the first version generated each
    // one twice — and a queen's attack set is two magic lookups. The counts are
    // filled in during the mobility walk and emitted after both colours are
    // done, because the pieces that bear on white's king are black's, and they
    // are not seen until the second pass.
    let zone = [
        king_attacks(pos.king_sq(WHITE)) | bit(pos.king_sq(WHITE)),
        king_attacks(pos.king_sq(BLACK)) | bit(pos.king_sq(BLACK)),
    ];
    let mut attackers = [0usize; 2];

    for c in 0..2 {
        // Relative colour under each perspective. The two are always opposite,
        // because the perspectives themselves are.
        let ra = if c == persp { 0 } else { 1 };
        let rb = 1 - ra;
        let them = c ^ 1;
        let our_pawns = pos.pieces(c, PAWN_P);
        let their_pawns = pos.pieces(them, PAWN_P);

        // A fact whose index depends only on relative colour.
        macro_rules! put {
            ($base:expr, $stride:expr, $v:expr) => {
                if n < MAX_F {
                    a[n] = ($base + ra * $stride + $v) as u16;
                    b[n] = ($base + rb * $stride + $v) as u16;
                    n += 1;
                }
            };
        }

        // --- piece-square
        for pt in 0..6 {
            let mut bb = pos.pieces(c, pt);
            while bb != 0 {
                let sq = pop_lsb(&mut bb);
                let (sa, sb) = if persp == WHITE { (sq, sq ^ 56) } else { (sq ^ 56, sq) };
                if n < MAX_F {
                    a[n] = (PSQ + (ra * 6 + pt) * 64 + sa) as u16;
                    b[n] = (PSQ + (rb * 6 + pt) * 64 + sb) as u16;
                    n += 1;
                }
            }
        }

        // --- mobility, one feature per piece
        for pt in [KNIGHT_P, BISHOP_P, ROOK_P, QUEEN_P] {
            let mut bb = pos.pieces(c, pt);
            while bb != 0 {
                let sq = pop_lsb(&mut bb);
                let att = match pt {
                    KNIGHT_P => knight_attacks(sq),
                    BISHOP_P => bishop_attacks(sq, occ),
                    ROOK_P => rook_attacks(sq, occ),
                    _ => queen_attacks(sq, occ),
                };
                let m = popcount(att & !pos.color[c]) as usize;
                put!(MOB, 4 * 12, (pt - 1) * 12 + m.min(11));
                if att & zone[them] != 0 {
                    attackers[them] += 1;
                }
            }
        }

        // --- pawn structure, asked of the whole board instead of pawn by pawn
        //
        // Every question here was being answered one pawn at a time, with a
        // `file_bb`, an adjacent-file mask and a `popcount` each, and
        // `passed_mask` rebuilding the same two masks a second time. All three
        // are functions of the pawn sets, so the file fills answer them for
        // sixteen pawns at the cost of a few shifts.
        //
        // isolated: no friendly pawn on either neighbouring file. `west`/`east`
        // clip at the edge files exactly as the per-pawn mask did.
        let our_files = file_fill(our_pawns);
        let isolated = popcount(our_pawns & !(west(our_files) | east(our_files))) as usize;
        // doubled: a friendly pawn strictly above or strictly below on the same
        // file. Counts every pawn on a shared file, as the loop did -- not the
        // number of surplus pawns.
        let doubled = popcount(our_pawns & (nfill(our_pawns << 8) | sfill(our_pawns >> 8))) as usize;
        // passed: no enemy pawn ahead on this file or either neighbour. Smearing
        // the enemy pawns sideways first puts a bit on file `f` at rank `r`
        // whenever an enemy pawn stands on `f-1`, `f` or `f+1` at that rank, so
        // one fill then covers all three files.
        let blockers = their_pawns | west(their_pawns) | east(their_pawns);
        let stopped = if c == WHITE { sfill(blockers >> 8) } else { nfill(blockers << 8) };
        // Ascending square order, which is the order the per-pawn loop emitted.
        let mut bb = our_pawns & !stopped;
        while bb != 0 {
            let sq = pop_lsb(&mut bb);
            let rel_rank = if c == WHITE { rank_of(sq) } else { 7 - rank_of(sq) };
            put!(PASSED, 8, rel_rank);
        }
        put!(ISOLATED, 4, isolated.min(3));
        put!(DOUBLED, 4, doubled.min(3));

        // --- rooks on open and half-open files
        let mut open = 0usize;
        let mut semi = 0usize;
        let mut bb = pos.pieces(c, ROOK_P);
        while bb != 0 {
            let sq = pop_lsb(&mut bb);
            let fb = file_bb(file_of(sq));
            if our_pawns & fb == 0 {
                if their_pawns & fb == 0 {
                    open += 1;
                } else {
                    semi += 1;
                }
            }
        }
        put!(ROOK_OPEN, 3, open.min(2));
        put!(ROOK_SEMI, 3, semi.min(2));

        if more_than_one(pos.pieces(c, BISHOP_P)) {
            put!(PAIR, 1, 0);
        }
    }

    // --- king safety, now that both sides' attackers have been counted
    for c in 0..2 {
        let ra = if c == persp { 0 } else { 1 };
        let rb = 1 - ra;
        if n < MAX_F {
            a[n] = (KING_ATT + ra * 8 + attackers[c].min(7)) as u16;
            b[n] = (KING_ATT + rb * 8 + attackers[c].min(7)) as u16;
            n += 1;
        }
        let shelter = (popcount(zone[c] & pos.pieces(c, PAWN_P)) as usize).min(3);
        if n < MAX_F {
            a[n] = (SHELTER + ra * 4 + shelter) as u16;
            b[n] = (SHELTER + rb * 4 + shelter) as u16;
            n += 1;
        }
    }

    n
}

/// Single-perspective view, for the training-data dump.
#[cfg_attr(not(test), allow(dead_code))]
pub fn features(pos: &Position, persp: usize, out: &mut [u16; MAX_F]) -> usize {
    let mut other = [0u16; MAX_F];
    features_both(pos, persp, out, &mut other)
}

/// Output bucket, from the number of pieces left. Must match the trainer.
#[inline(always)]
pub fn bucket_of(pieces: usize) -> usize {
    (pieces.saturating_sub(1) * BUCKETS / 32).min(BUCKETS - 1)
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

/// A hidden layer this size is four NEON registers, so the accumulators stay in
/// them for the whole walk over the feature list. The obvious version — one
/// `acc += row` helper called per feature — reloads and restores the
/// accumulator around every single row, which is eighty round trips to memory
/// per evaluation for arithmetic that never needed to leave the register file.
const _: () = assert!(H.is_multiple_of(8), "the accumulator is walked eight lanes at a time");

/// Both perspectives at once. They read different rows but the same feature
/// count, so pairing them halves the loop overhead and gives the two
/// independent load-add chains something to interleave with.
fn accumulate_both(us: &mut [i16; H], them: &mut [i16; H], fu: &[u16], ft: &[u16], count: usize) {
    let n = net();

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        const V: usize = H / 8;
        let mut a = [vdupq_n_s16(0); V];
        let mut b = [vdupq_n_s16(0); V];
        for j in 0..V {
            a[j] = vld1q_s16(n.ft_b.as_ptr().add(j * 8));
            b[j] = a[j];
        }
        for i in 0..count {
            let ra = n.ft_w.as_ptr().add(*fu.get_unchecked(i) as usize * H);
            let rb = n.ft_w.as_ptr().add(*ft.get_unchecked(i) as usize * H);
            for j in 0..V {
                a[j] = vaddq_s16(a[j], vmovl_s8(vld1_s8(ra.add(j * 8))));
                b[j] = vaddq_s16(b[j], vmovl_s8(vld1_s8(rb.add(j * 8))));
            }
        }
        for j in 0..V {
            vst1q_s16(us.as_mut_ptr().add(j * 8), a[j]);
            vst1q_s16(them.as_mut_ptr().add(j * 8), b[j]);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        us.copy_from_slice(&n.ft_b);
        them.copy_from_slice(&n.ft_b);
        for i in 0..count {
            let (ba, bb) = (fu[i] as usize * H, ft[i] as usize * H);
            for j in 0..H {
                us[j] += n.ft_w[ba + j] as i16;
                them[j] += n.ft_w[bb + j] as i16;
            }
        }
    }
}

/// Clipped ReLU followed by the output dot product, fused so the activations
/// never leave registers.
#[inline(always)]
fn propagate(acc: &[i16; H], w: &[i8]) -> i32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let zero = vdupq_n_s16(0);
        let top = vdupq_n_s16(QA as i16);
        let mut sum = vdupq_n_s32(0);
        let mut i = 0;
        while i + 8 <= H {
            let a = vminq_s16(vmaxq_s16(vld1q_s16(acc.as_ptr().add(i)), zero), top);
            let ww = vmovl_s8(vld1_s8(w.as_ptr().add(i)));
            sum = vmlal_s16(sum, vget_low_s16(a), vget_low_s16(ww));
            sum = vmlal_high_s16(sum, a, ww);
            i += 8;
        }
        let mut total = vaddvq_s32(sum);
        while i < H {
            total += (acc[i].clamp(0, QA as i16) as i32) * w[i] as i32;
            i += 1;
        }
        total
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut total = 0i32;
        for i in 0..H {
            total += (acc[i].clamp(0, QA as i16) as i32) * w[i] as i32;
        }
        total
    }
}

/// Evaluation in centipawns, from the side to move's point of view.
pub fn evaluate(pos: &Position) -> i32 {
    // An erased slot is all-zero, which is a real entry for the one position in
    // a trillion whose key has forty zero bits on top and whose score is zero.
    // That costs a zero instead of a zero; no validity bit is worth the space.
    let slot = (pos.key as usize) & ((1 << CACHE_BITS) - 1);
    let gen = unsafe { *GEN.as_ref() };
    let want = pack(pos.key, gen, 0);
    let c = unsafe { CACHE.as_mut() };
    let hit = c[slot];
    if hit & !0xFFFF == want {
        return hit as u16 as i16 as i32;
    }

    let n = net();
    let mut fu = [0u16; MAX_F];
    let mut ft = [0u16; MAX_F];
    let count = features_both(pos, pos.stm, &mut fu, &mut ft);
    let mut us = [0i16; H];
    let mut them = [0i16; H];
    accumulate_both(&mut us, &mut them, &fu, &ft, count);
    let b = bucket_of(popcount(pos.occ()) as usize);
    let w = &n.out_w[b * 2 * H..(b + 1) * 2 * H];
    let out = propagate(&us, &w[..H]) + propagate(&them, &w[H..]) + n.out_b[b];
    let score = (out * SCALE / (QA * QB)).clamp(-20_000, 20_000);
    c[slot] = pack(pos.key, gen, score);
    score
}
