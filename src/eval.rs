//! Hand-crafted evaluation.
//!
//! Tapered between a middlegame and an endgame score by remaining material.
//! This is the fallback evaluator and, more importantly, the *teacher* the
//! network is distilled against, so it is written to be stable and cheap
//! rather than clever.

use crate::bb::*;
use crate::pos::*;

pub const MATE: i32 = 32_000;
pub const MATE_IN_MAX: i32 = MATE - MAX_PLY as i32;
pub const INF: i32 = 32_500;

// Middlegame / endgame piece values. Index by piece type.
pub const MG_VAL: [i32; 6] = [82, 337, 365, 477, 1025, 0];
pub const EG_VAL: [i32; 6] = [94, 281, 297, 512, 936, 0];
/// Used by SEE and by capture ordering, where a single scale is enough.
pub const SEE_VAL: [i32; 7] = [100, 325, 335, 500, 975, 10_000, 0];

/// Game-phase weight per piece; 24 = full board.
const PHASE_W: [i32; 6] = [0, 1, 1, 2, 4, 0];
const MAX_PHASE: i32 = 24;

// Piece-square tables, white's point of view, a1 = 0.
#[rustfmt::skip]
const MG_PST: [[i32; 64]; 6] = [
// Pawn
[  0,   0,   0,   0,   0,   0,   0,   0,
 -35,  -1, -20, -23, -15,  24,  38, -22,
 -26,  -4,  -4, -10,   3,   3,  33, -12,
 -27,  -2,  -5,  12,  17,   6,  10, -25,
 -14,  13,   6,  21,  23,  12,  17, -23,
  -6,   7,  26,  31,  65,  56,  25, -20,
  98, 134,  61,  95,  68, 126,  34, -11,
   0,   0,   0,   0,   0,   0,   0,   0],
// Knight
[-105, -21, -58, -33, -17, -28, -19,  -23,
  -29, -53, -12,  -3,  -1,  18, -14,  -19,
  -23,  -9,  12,  10,  19,  17,  25,  -16,
  -13,   4,  16,  13,  28,  19,  21,   -8,
   -9,  17,  19,  53,  37,  69,  18,   22,
  -47,  60,  37,  65,  84, 129,  73,   44,
  -73, -41,  72,  36,  23,  62,   7,  -17,
 -167, -89, -34, -49,  61, -97, -15, -107],
// Bishop
[-33,  -3, -14, -21, -13, -12, -39, -21,
   4,  15,  16,   0,   7,  21,  33,   1,
   0,  15,  15,  15,  14,  27,  18,  10,
  -6,  13,  13,  26,  34,  12,  10,   4,
  -4,   5,  19,  50,  37,  37,   7,  -2,
 -16,  37,  43,  40,  35,  50,  37,  -2,
 -26,  16, -18, -13,  30,  59,  18, -47,
 -29,   4, -82, -37, -25, -42,   7,  -8],
// Rook
[-19, -13,   1,  17,  16,   7, -37, -26,
 -44, -16, -20,  -9,  -1,  11,  -6, -71,
 -45, -25, -16, -17,   3,   0,  -5, -33,
 -36, -26, -12,  -1,   9,  -7,   6, -23,
 -24, -11,   7,  26,  24,  35,  -8, -20,
  -5,  19,  26,  36,  17,  45,  61,  16,
  27,  32,  58,  62,  80,  67,  26,  44,
  32,  42,  32,  51,  63,   9,  31,  43],
// Queen
[ -1, -18,  -9,  10, -15, -25, -31, -50,
 -35,  -8,  11,   2,   8,  15,  -3,   1,
 -14,   2, -11,  -2,  -5,   2,  14,   5,
  -9, -26,  -9, -10,  -2,  -4,   3,  -3,
 -27, -27, -16, -16,  -1,  17,  -2,   1,
 -13, -17,   7,   8,  29,  56,  47,  57,
 -24, -39,  -5,   1, -16,  57,  28,  54,
 -28,   0,  29,  12,  59,  44,  43,  45],
// King
[-15,  36,  12, -54,   8, -28,  24,  14,
   1,   7,  -8, -64, -43, -16,   9,   8,
 -14, -14, -22, -46, -44, -30, -15, -27,
 -49,  -1, -27, -39, -46, -44, -33, -51,
 -17, -20, -12, -27, -30, -25, -14, -36,
  -9,  24,   2, -16, -20,   6,  22, -22,
  29,  -1, -20,  -7,  -8,  -4, -38, -29,
 -65,  23,  16, -15, -56, -34,   2,  13],
];

#[rustfmt::skip]
const EG_PST: [[i32; 64]; 6] = [
// Pawn
[  0,   0,   0,   0,   0,   0,   0,   0,
  13,   8,   8,  10,  13,   0,   2,  -7,
   4,   7,  -6,   1,   0,  -5,  -1,  -8,
  13,   9,  -3,  -7,  -7,  -8,   3,  -1,
  32,  24,  13,   5,  -2,   4,  17,  17,
  94, 100,  85,  67,  56,  53,  82,  84,
 178, 173, 158, 134, 147, 132, 165, 187,
   0,   0,   0,   0,   0,   0,   0,   0],
// Knight
[-29, -51, -23, -15, -22, -18, -50, -64,
 -42, -20, -10,  -5,  -2, -20, -23, -44,
 -23,  -3,  -1,  15,  10,  -3, -20, -22,
 -18,  -6,  16,  25,  16,  17,   4, -18,
 -17,   3,  22,  22,  22,  11,   8, -18,
 -24, -20,  10,   9,  -1,  -9, -19, -41,
 -25,  -8, -25,  -2,  -9, -25, -24, -52,
 -58, -38, -13, -28, -31, -27, -63, -99],
// Bishop
[-23,  -9, -23,  -5,  -9, -16,  -5, -17,
 -14, -18,  -7,  -1,   4,  -9, -15, -27,
 -12,  -3,   8,  10,  13,   3,  -7, -15,
  -6,   3,  13,  19,   7,  10,  -3,  -9,
  -3,   9,  12,   9,  14,  10,   3,   2,
   2,  -8,   0,  -1,  -2,   6,   0,   4,
  -8,  -4,   7, -12,  -3, -13,  -4, -14,
 -14, -21, -11,  -8,  -7,  -9, -17, -24],
// Rook
[ -9,   2,   3,  -1,  -5, -13,   4, -20,
  -6,  -6,   0,   2,  -9,  -9, -11,  -3,
  -4,   0,  -5,  -1,  -7, -12,  -8, -16,
   3,   5,   8,   4,  -5,  -6,  -8, -11,
   4,   3,  13,   1,   2,   1,  -1,   2,
   7,   7,   7,   5,   4,  -3,  -5,  -3,
  11,  13,  13,  11,  -3,   3,   8,   3,
  13,  10,  18,  15,  12,  12,   8,   5],
// Queen
[-33, -28, -22, -43,  -5, -32, -20, -41,
 -22, -23, -30, -16, -16, -23, -36, -32,
 -16, -27,  15,   6,   9,  17,  10,   5,
 -18,  28,  19,  47,  31,  34,  39,  23,
   3,  22,  24,  45,  57,  40,  57,  36,
 -20,   6,   9,  49,  47,  35,  19,   9,
 -17,  20,  32,  41,  58,  25,  30,   0,
  -9,  22,  22,  27,  27,  19,  10,  20],
// King
[-53, -34, -21, -11, -28, -14, -24, -43,
 -27, -11,   4,  13,  14,   4,  -5, -17,
 -19,  -3,  11,  21,  23,  16,   7,  -9,
 -18,  -4,  21,  24,  27,  23,   9, -11,
  -8,  22,  24,  27,  26,  33,  26,   3,
  10,  17,  23,  15,  20,  45,  44,  13,
 -12,  17,  14,  17,  17,  38,  23,  11,
 -74, -35, -18, -18, -11,  15,   4, -17],
];

/// Flip a square to the other side's perspective.
#[inline(always)]
const fn flip(sq: usize) -> usize {
    sq ^ 56
}

const ISOLATED: [i32; 2] = [-12, -14];
const DOUBLED: [i32; 2] = [-8, -22];
const PASSED_MG: [i32; 8] = [0, 2, 6, 14, 30, 58, 96, 0];
const PASSED_EG: [i32; 8] = [0, 8, 16, 32, 60, 100, 150, 0];
const BISHOP_PAIR: [i32; 2] = [24, 48];
const ROOK_OPEN: [i32; 2] = [26, 12];
const ROOK_SEMI: [i32; 2] = [12, 6];
const TEMPO: i32 = 12;

// Mobility bonuses, indexed by piece then by attacked-square count.
#[rustfmt::skip]
const MOB_MG: [[i32; 28]; 4] = [
 [-25,-11, -2,  4,  9, 13, 17, 20, 23,  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
 [-30,-16, -6,  0,  6, 11, 14, 17, 19, 21, 23, 25, 27, 29, 0,0,0,0,0,0,0,0,0,0,0,0,0,0],
 [-24,-12, -6, -2,  1,  4,  8, 11, 14, 16, 18, 20, 21, 22, 23, 0,0,0,0,0,0,0,0,0,0,0,0,0],
 [-15, -9, -5, -2,  0,  2,  4,  6,  8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
];
#[rustfmt::skip]
const MOB_EG: [[i32; 28]; 4] = [
 [-32,-16, -4,  4, 10, 15, 19, 22, 25,  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
 [-40,-20, -8,  0,  8, 14, 19, 23, 26, 29, 31, 33, 35, 37, 0,0,0,0,0,0,0,0,0,0,0,0,0,0],
 [-36,-18, -6,  4, 12, 19, 25, 30, 34, 38, 41, 44, 46, 48, 50, 0,0,0,0,0,0,0,0,0,0,0,0,0],
 [-20,-12, -6,  0,  6, 11, 16, 20, 24, 28, 31, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66],
];

/// Danger weight per attacker type against the enemy king zone.
const KING_ATT_W: [i32; 6] = [0, 20, 20, 40, 80, 0];

struct Acc {
    mg: i32,
    eg: i32,
}

/// Full evaluation: the hand-crafted term plus, when a network is embedded,
/// the learned correction to it.
///
/// The network is *additive* rather than a replacement. A 24 KB net over plain
/// piece-square features cannot represent mobility or king safety -- those
/// depend on where pieces can go, not where they are -- so asking it to
/// reproduce the whole function from scratch throws that knowledge away. Asking
/// it only for the residual keeps everything the hand-crafted terms already
/// know and spends the entire parameter budget on what they miss.
pub fn evaluate(pos: &Position) -> i32 {
    let base = hand_crafted(pos);
    if crate::net::is_loaded() {
        (base + crate::net::evaluate(pos)).clamp(-20_000, 20_000)
    } else {
        base
    }
}

pub fn hand_crafted(pos: &Position) -> i32 {
    if pos.is_material_draw() {
        return 0;
    }
    let mut a = Acc { mg: 0, eg: 0 };
    let mut phase = 0i32;
    let occ = pos.occ();

    for c in 0..2 {
        let sign = if c == WHITE { 1 } else { -1 };
        let them = c ^ 1;
        let our_pawns = pos.pieces(c, PAWN_P);
        let their_pawns = pos.pieces(them, PAWN_P);
        let their_king = pos.king_sq(them);
        let king_zone = king_attacks(their_king) | bit(their_king);
        let mut danger = 0i32;
        let mut attackers = 0i32;

        for pt in 0..6 {
            let mut b = pos.pieces(c, pt);
            phase += PHASE_W[pt] * popcount(b) as i32;
            while b != 0 {
                let sq = pop_lsb(&mut b);
                let rel = if c == WHITE { sq } else { flip(sq) };
                a.mg += sign * (MG_VAL[pt] + MG_PST[pt][rel]);
                a.eg += sign * (EG_VAL[pt] + EG_PST[pt][rel]);

                match pt {
                    KNIGHT_P | BISHOP_P | ROOK_P | QUEEN_P => {
                        let att = match pt {
                            KNIGHT_P => knight_attacks(sq),
                            BISHOP_P => bishop_attacks(sq, occ),
                            ROOK_P => rook_attacks(sq, occ),
                            _ => queen_attacks(sq, occ),
                        };
                        // Squares controlled by enemy pawns are not real mobility.
                        let safe = att & !pos.color[c] & !pawn_shield_attacks(their_pawns, them);
                        let n = popcount(safe) as usize;
                        a.mg += sign * MOB_MG[pt - 1][n];
                        a.eg += sign * MOB_EG[pt - 1][n];
                        if att & king_zone != 0 {
                            danger += KING_ATT_W[pt] * popcount(att & king_zone) as i32;
                            attackers += 1;
                        }
                        if pt == ROOK_P {
                            let f = file_bb(file_of(sq));
                            if our_pawns & f == 0 {
                                if their_pawns & f == 0 {
                                    a.mg += sign * ROOK_OPEN[0];
                                    a.eg += sign * ROOK_OPEN[1];
                                } else {
                                    a.mg += sign * ROOK_SEMI[0];
                                    a.eg += sign * ROOK_SEMI[1];
                                }
                            }
                        }
                    }
                    PAWN_P => {
                        let f = file_of(sq);
                        let fb = file_bb(f);
                        let adjacent = (fb & !FILE_A) >> 1 | (fb & !FILE_H) << 1;
                        if our_pawns & adjacent == 0 {
                            a.mg += sign * ISOLATED[0];
                            a.eg += sign * ISOLATED[1];
                        }
                        if popcount(our_pawns & fb) > 1 {
                            a.mg += sign * DOUBLED[0];
                            a.eg += sign * DOUBLED[1];
                        }
                        if their_pawns & passed_mask(c, sq) == 0 {
                            let r = if c == WHITE { rank_of(sq) } else { 7 - rank_of(sq) };
                            a.mg += sign * PASSED_MG[r];
                            a.eg += sign * PASSED_EG[r];
                        }
                    }
                    _ => {}
                }
            }
        }

        if more_than_one(pos.pieces(c, BISHOP_P)) {
            a.mg += sign * BISHOP_PAIR[0];
            a.eg += sign * BISHOP_PAIR[1];
        }
        // A lone attacker is noise; pressure only counts when it is coordinated.
        if attackers >= 2 {
            a.mg += sign * (danger * attackers.min(6)) / 16;
        }
        // Pawn shelter in front of our own king.
        let ksq = pos.king_sq(c);
        let shelter = king_attacks(ksq) & push(bit(ksq) | king_attacks(ksq), c);
        a.mg += sign * 6 * popcount(shelter & our_pawns) as i32;
    }

    let phase = phase.min(MAX_PHASE);
    let score = (a.mg * phase + a.eg * (MAX_PHASE - phase)) / MAX_PHASE;
    let score = if pos.stm == WHITE { score + TEMPO } else { -score + TEMPO };
    scale_drawish(pos, score)
}

/// Squares attacked by `side`'s pawn set.
#[inline(always)]
fn pawn_shield_attacks(pawns: Bb, side: usize) -> Bb {
    push(east(pawns), side) | push(west(pawns), side)
}

/// Squares ahead of `sq` on its own and adjacent files, from `c`'s view.
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

/// Pull scores toward zero when the stronger side has no pawns to convert with.
fn scale_drawish(pos: &Position, score: i32) -> i32 {
    let strong = if score > 0 { WHITE } else { BLACK };
    if pos.pieces(strong, PAWN_P) == 0 {
        let mat: i32 = (1..5)
            .map(|p| EG_VAL[p] * popcount(pos.pieces(strong, p)) as i32)
            .sum();
        let their: i32 = (1..5)
            .map(|p| EG_VAL[p] * popcount(pos.pieces(strong ^ 1, p)) as i32)
            .sum();
        if mat - their < EG_VAL[ROOK_P] {
            return score / 4;
        }
    }
    score
}
