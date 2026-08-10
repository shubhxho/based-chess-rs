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
    let rd32 =
        |o: usize| u32::from_le_bytes([BLOB[o], BLOB[o + 1], BLOB[o + 2], BLOB[o + 3]]) as usize;
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

/// Squares ahead of `sq` on its own and the adjacent files, from `c`'s view.
fn passed_mask(c: usize, sq: usize) -> Bb {
    let f = file_of(sq);
    let files = file_bb(f) | (file_bb(f) & !FILE_A) >> 1 | (file_bb(f) & !FILE_H) << 1;
    let ahead = if c == WHITE {
        !0u64 << ((rank_of(sq) + 1) * 8)
    } else {
        (1u64 << (rank_of(sq) * 8)) - 1
    };
    files & ahead
}

/// The active feature indices for `persp`'s view of the position.
///
/// Colours are relative: block 0 is always "mine", block 1 always "theirs", and
/// squares are mirrored for black. One weight matrix therefore serves both
/// sides, and the network learns a single function of "my position" rather than
/// two functions of "white's position".
pub fn features(pos: &Position, persp: usize, out: &mut [u16; MAX_F]) -> usize {
    let mut n = 0usize;
    let occ = pos.occ();

    macro_rules! put {
        ($idx:expr) => {
            if n < MAX_F {
                out[n] = $idx as u16;
                n += 1;
            }
        };
    }

    for c in 0..2 {
        let rel_c = if c == persp { 0 } else { 1 };
        let them = c ^ 1;
        let our_pawns = pos.pieces(c, PAWN_P);
        let their_pawns = pos.pieces(them, PAWN_P);

        // --- piece-square
        for pt in 0..6 {
            let mut b = pos.pieces(c, pt);
            while b != 0 {
                let sq = pop_lsb(&mut b);
                let rel_sq = if persp == WHITE { sq } else { sq ^ 56 };
                put!(PSQ + (rel_c * 6 + pt) * 64 + rel_sq);
            }
        }

        // --- mobility, one feature per piece
        for pt in [KNIGHT_P, BISHOP_P, ROOK_P, QUEEN_P] {
            let mut b = pos.pieces(c, pt);
            while b != 0 {
                let sq = pop_lsb(&mut b);
                let att = match pt {
                    KNIGHT_P => knight_attacks(sq),
                    BISHOP_P => bishop_attacks(sq, occ),
                    ROOK_P => rook_attacks(sq, occ),
                    _ => queen_attacks(sq, occ),
                };
                let m = popcount(att & !pos.color[c]) as usize;
                put!(MOB + (rel_c * 4 + (pt - 1)) * 12 + m.min(11));
            }
        }

        // --- pawn structure
        let mut isolated = 0usize;
        let mut doubled = 0usize;
        let mut b = our_pawns;
        while b != 0 {
            let sq = pop_lsb(&mut b);
            let fb = file_bb(file_of(sq));
            let adjacent = (fb & !FILE_A) >> 1 | (fb & !FILE_H) << 1;
            if our_pawns & adjacent == 0 {
                isolated += 1;
            }
            if popcount(our_pawns & fb) > 1 {
                doubled += 1;
            }
            if their_pawns & passed_mask(c, sq) == 0 {
                let rel_rank = if c == WHITE { rank_of(sq) } else { 7 - rank_of(sq) };
                put!(PASSED + rel_c * 8 + rel_rank);
            }
        }
        put!(ISOLATED + rel_c * 4 + isolated.min(3));
        put!(DOUBLED + rel_c * 4 + doubled.min(3));

        // --- rooks on open and half-open files
        let mut open = 0usize;
        let mut semi = 0usize;
        let mut b = pos.pieces(c, ROOK_P);
        while b != 0 {
            let sq = pop_lsb(&mut b);
            let fb = file_bb(file_of(sq));
            if our_pawns & fb == 0 {
                if their_pawns & fb == 0 {
                    open += 1;
                } else {
                    semi += 1;
                }
            }
        }
        put!(ROOK_OPEN + rel_c * 3 + open.min(2));
        put!(ROOK_SEMI + rel_c * 3 + semi.min(2));

        if more_than_one(pos.pieces(c, BISHOP_P)) {
            put!(PAIR + rel_c);
        }

        // --- king safety: how many enemy pieces bear on the king's neighbourhood
        let ksq = pos.king_sq(c);
        let zone = king_attacks(ksq) | bit(ksq);
        let mut attackers = 0usize;
        for pt in [KNIGHT_P, BISHOP_P, ROOK_P, QUEEN_P] {
            let mut e = pos.pieces(them, pt);
            while e != 0 {
                let sq = pop_lsb(&mut e);
                let att = match pt {
                    KNIGHT_P => knight_attacks(sq),
                    BISHOP_P => bishop_attacks(sq, occ),
                    ROOK_P => rook_attacks(sq, occ),
                    _ => queen_attacks(sq, occ),
                };
                if att & zone != 0 {
                    attackers += 1;
                }
            }
        }
        put!(KING_ATT + rel_c * 8 + attackers.min(7));
        put!(SHELTER + rel_c * 4 + (popcount(zone & our_pawns) as usize).min(3));
    }

    n
}

/// Output bucket, from the number of pieces left. Must match the trainer.
#[inline(always)]
pub fn bucket_of(pieces: usize) -> usize {
    (pieces.saturating_sub(1) * BUCKETS / 32).min(BUCKETS - 1)
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

fn accumulate(pos: &Position, persp: usize, acc: &mut [i16; H]) {
    let n = net();
    acc.copy_from_slice(&n.ft_b);
    let mut feat = [0u16; MAX_F];
    let count = features(pos, persp, &mut feat);
    for &f in feat.iter().take(count) {
        let base = f as usize * H;
        add_row(acc, &n.ft_w[base..base + H]);
    }
}

/// `acc += row`, widening int8 to int16. Written against the NEON intrinsics
/// directly; the scalar path exists only for non-aarch64 builds.
#[inline(always)]
fn add_row(acc: &mut [i16; H], row: &[i8]) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let mut i = 0;
        while i + 8 <= H {
            let w = vmovl_s8(vld1_s8(row.as_ptr().add(i)));
            let a = vld1q_s16(acc.as_ptr().add(i));
            vst1q_s16(acc.as_mut_ptr().add(i), vaddq_s16(a, w));
            i += 8;
        }
        while i < H {
            acc[i] += row[i] as i16;
            i += 1;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for i in 0..H {
        acc[i] += row[i] as i16;
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
    let n = net();
    let mut us = [0i16; H];
    let mut them = [0i16; H];
    accumulate(pos, pos.stm, &mut us);
    accumulate(pos, pos.stm ^ 1, &mut them);
    let b = bucket_of(popcount(pos.occ()) as usize);
    let w = &n.out_w[b * 2 * H..(b + 1) * 2 * H];
    let out = propagate(&us, &w[..H]) + propagate(&them, &w[H..]) + n.out_b[b];
    (out * SCALE / (QA * QB)).clamp(-20_000, 20_000)
}
