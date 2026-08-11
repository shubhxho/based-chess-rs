//! Fully legal move generation.
//!
//! Nothing pseudo-legal escapes this module: pins, check evasions and the
//! en-passant discovered-check case are resolved during generation, so the
//! search never makes a move it has to take back.

use crate::bb::*;
use crate::pos::*;

use core::mem::MaybeUninit;

pub const MAX_MOVES: usize = 256;

pub struct MoveList {
    pub mv: [Move; MAX_MOVES],
    pub sc: [i32; MAX_MOVES],
    pub n: usize,
}

/// The moves already tried at a node, kept only so that a cutoff can penalise
/// them in the history tables afterwards.
///
/// This was a `MoveList`, which is the wrong shape for the job twice over: it
/// carries a 1 KB score array nothing here ever reads, and constructing one
/// zeroes all 1544 bytes. Two of them per interior node came to 4.6 KB of
/// memset and memcpy per node, which is what put `_platform_memmove` in the
/// profile. Nothing is initialised here -- `push` writes slot `n` before `n`
/// grows, and `get` is only ever called below `len`.
pub struct Tried {
    mv: [MaybeUninit<Move>; MAX_MOVES],
    n: usize,
}

impl Tried {
    pub const fn new() -> Tried {
        Tried { mv: [MaybeUninit::uninit(); MAX_MOVES], n: 0 }
    }
    #[inline(always)]
    pub fn push(&mut self, m: Move) {
        unsafe {
            *self.mv.get_unchecked_mut(self.n) = MaybeUninit::new(m);
        }
        self.n += 1;
    }
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.n
    }
    /// Caller must stay below `len`; every slot under it was written by `push`.
    #[inline(always)]
    pub fn get(&self, i: usize) -> Move {
        unsafe { self.mv.get_unchecked(i).assume_init() }
    }
}

impl MoveList {
    pub const fn new() -> MoveList {
        MoveList { mv: [Move::NULL; MAX_MOVES], sc: [0; MAX_MOVES], n: 0 }
    }
    #[inline(always)]
    pub fn push(&mut self, m: Move) {
        unsafe {
            *self.mv.get_unchecked_mut(self.n) = m;
            *self.sc.get_unchecked_mut(self.n) = 0;
        }
        self.n += 1;
    }
    #[inline(always)]
    pub fn clear(&mut self) {
        self.n = 0;
    }
    /// Selection sort one slot at a time: the search usually gets a cutoff
    /// after a handful of moves, so sorting the whole list up front is wasted.
    #[inline(always)]
    pub fn pick(&mut self, idx: usize) -> Move {
        let mut best = idx;
        for i in idx + 1..self.n {
            if self.sc[i] > self.sc[best] {
                best = i;
            }
        }
        self.mv.swap(idx, best);
        self.sc.swap(idx, best);
        self.mv[idx]
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GenKind {
    All,
    /// Captures, en passant and queen promotions: the quiescence set.
    Noisy,
}

pub fn generate(pos: &Position, list: &mut MoveList, kind: GenKind) {
    list.clear();
    let us = pos.stm;
    let them = us ^ 1;
    let occ = pos.occ();
    let own = pos.color[us];
    let enemy = pos.color[them];
    let ksq = pos.king_sq(us);
    let checkers = pos.checkers;

    // --- king moves; always generated, always the only option in double check
    {
        let occ_no_king = occ ^ bit(ksq);
        let mut t = king_attacks(ksq) & !own;
        if kind == GenKind::Noisy {
            t &= enemy;
        }
        while t != 0 {
            let to = pop_lsb(&mut t);
            if !pos.attacked_by(them, to, occ_no_king) {
                list.push(Move::new(ksq, to, if enemy & bit(to) != 0 { F_CAPTURE } else { F_QUIET }));
            }
        }
    }
    if more_than_one(checkers) {
        return;
    }

    // --- target masks
    let (quiet_t, cap_t) = if checkers != 0 {
        let csq = lsb(checkers);
        (between(ksq, csq) & !occ, checkers)
    } else {
        (!occ, enemy)
    };
    // `land_t` is where a non-capturing move may land regardless of generation
    // kind — promotions are noisy but still land on empty squares.
    let land_t = quiet_t;
    let quiet_t = if kind == GenKind::Noisy { 0 } else { quiet_t };

    let pinned = pos.pinned(us);
    // Destinations a pinned piece may still use: along the pin ray only.
    macro_rules! ok {
        ($from:expr, $to:expr) => {
            pinned & bit($from) == 0 || line(ksq, $from) & bit($to) != 0
        };
    }

    // --- pawns
    {
        let pawns = pos.pieces(us, PAWN_P);
        let promo_rank = if us == WHITE { RANK_8 } else { RANK_1 };

        // Single and double pushes.
        {
            let one = push(pawns, us) & !occ;
            let two = push(one & if us == WHITE { RANK_A3 } else { RANK_A6 }, us) & !occ;

            let mut promo_push = one & promo_rank & land_t;
            while promo_push != 0 {
                let to = pop_lsb(&mut promo_push);
                let from = if us == WHITE { to - 8 } else { to + 8 };
                if ok!(from, to) {
                    push_promos(list, from, to, false, kind);
                }
            }

            let mut p = one & !promo_rank & quiet_t;
            while p != 0 {
                let to = pop_lsb(&mut p);
                let from = if us == WHITE { to - 8 } else { to + 8 };
                if ok!(from, to) {
                    list.push(Move::new(from, to, F_QUIET));
                }
            }

            let mut d = two & quiet_t;
            while d != 0 {
                let to = pop_lsb(&mut d);
                let from = if us == WHITE { to - 16 } else { to + 16 };
                if ok!(from, to) {
                    list.push(Move::new(from, to, F_DOUBLE));
                }
            }
        }

        // Captures.
        for shift_e in [true, false] {
            let targets = if shift_e { push(east(pawns), us) & cap_t } else { push(west(pawns), us) & cap_t };
            let mut t = targets;
            while t != 0 {
                let to = pop_lsb(&mut t);
                let back = if us == WHITE { to - 8 } else { to + 8 };
                let from = if shift_e { back - 1 } else { back + 1 };
                if !ok!(from, to) {
                    continue;
                }
                if bit(to) & promo_rank != 0 {
                    push_promos(list, from, to, true, kind);
                } else {
                    list.push(Move::new(from, to, F_CAPTURE));
                }
            }
        }

        // En passant: rare and full of edge cases, so it gets an explicit
        // occupancy simulation instead of the pin shortcut. Two pawns leave
        // the board on the same rank, which can expose the king sideways.
        if pos.ep != 64 {
            let epsq = pos.ep as usize;
            let cap_sq = if us == WHITE { epsq - 8 } else { epsq + 8 };
            let mut from_bb = pawn_attacks(them, epsq) & pawns;
            while from_bb != 0 {
                let from = pop_lsb(&mut from_bb);
                let after = (occ ^ bit(from) ^ bit(cap_sq)) | bit(epsq);
                let enemy_after = enemy ^ bit(cap_sq);
                if !attacked_with(pos, them, ksq, after, enemy_after) {
                    list.push(Move::new(from, epsq, F_EP));
                }
            }
        }
    }

    // --- knights
    {
        let mut b = pos.pieces(us, KNIGHT_P) & !pinned; // a pinned knight never moves
        while b != 0 {
            let from = pop_lsb(&mut b);
            let mut t = knight_attacks(from) & (quiet_t | cap_t);
            while t != 0 {
                let to = pop_lsb(&mut t);
                list.push(Move::new(from, to, if enemy & bit(to) != 0 { F_CAPTURE } else { F_QUIET }));
            }
        }
    }

    // --- sliders
    for pt in [BISHOP_P, ROOK_P, QUEEN_P] {
        let mut b = pos.pieces(us, pt);
        while b != 0 {
            let from = pop_lsb(&mut b);
            let att = match pt {
                BISHOP_P => bishop_attacks(from, occ),
                ROOK_P => rook_attacks(from, occ),
                _ => queen_attacks(from, occ),
            };
            let mut t = att & (quiet_t | cap_t);
            if pinned & bit(from) != 0 {
                t &= line(ksq, from);
            }
            while t != 0 {
                let to = pop_lsb(&mut t);
                list.push(Move::new(from, to, if enemy & bit(to) != 0 { F_CAPTURE } else { F_QUIET }));
            }
        }
    }

    // --- castling
    if kind == GenKind::All && checkers == 0 {
        let (kc, qc, e, f, g, d, c, b_sq) = if us == WHITE {
            (WK, WQ, 4usize, 5usize, 6usize, 3usize, 2usize, 1usize)
        } else {
            (BK, BQ, 60usize, 61usize, 62usize, 59usize, 58usize, 57usize)
        };
        if pos.castle & kc != 0
            && occ & (bit(f) | bit(g)) == 0
            && !pos.attacked_by(them, f, occ)
            && !pos.attacked_by(them, g, occ)
        {
            list.push(Move::new(e, g, F_KCASTLE));
        }
        if pos.castle & qc != 0
            && occ & (bit(d) | bit(c) | bit(b_sq)) == 0
            && !pos.attacked_by(them, d, occ)
            && !pos.attacked_by(them, c, occ)
        {
            list.push(Move::new(e, c, F_QCASTLE));
        }
    }
}

const RANK_A3: Bb = RANK_1 << 16;
const RANK_A6: Bb = RANK_1 << 40;

#[inline(always)]
fn push_promos(list: &mut MoveList, from: usize, to: usize, cap: bool, kind: GenKind) {
    let base = F_PROMO | if cap { F_CAPTURE } else { 0 };
    // Queen first; it is right often enough that the rest are searched late.
    list.push(Move::new(from, to, base | 3));
    if kind == GenKind::All {
        list.push(Move::new(from, to, base | 2));
        list.push(Move::new(from, to, base | 1));
        list.push(Move::new(from, to, base));
    }
}

/// `attacked_by` against a hypothetical occupancy *and* a hypothetical enemy
/// set, which the en-passant test needs since the captured pawn vanishes.
fn attacked_with(pos: &Position, c: usize, sq: usize, occ: Bb, them: Bb) -> bool {
    if pawn_attacks(c ^ 1, sq) & pos.piece[PAWN_P] & them != 0 {
        return true;
    }
    if knight_attacks(sq) & pos.piece[KNIGHT_P] & them != 0 {
        return true;
    }
    if king_attacks(sq) & pos.piece[KING_P] & them != 0 {
        return true;
    }
    if bishop_attacks(sq, occ) & (pos.piece[BISHOP_P] | pos.piece[QUEEN_P]) & them != 0 {
        return true;
    }
    rook_attacks(sq, occ) & (pos.piece[ROOK_P] | pos.piece[QUEEN_P]) & them != 0
}

/// Does `m` give check? Computed before the move is made, without touching the
/// position, so the search can extend on checks for free.
pub fn gives_check(pos: &Position, m: Move) -> bool {
    let us = pos.stm;
    let them = us ^ 1;
    let ksq = pos.king_sq(them);
    let from = m.from();
    let to = m.to();
    let mut occ = pos.occ() ^ bit(from) ^ bit(to);
    let pt = if m.is_promo() { m.promo() } else { pos.piece_at(from) as usize };

    // Direct check.
    let direct = match pt {
        PAWN_P => pawn_attacks(us, to) & bit(ksq) != 0,
        KNIGHT_P => knight_attacks(to) & bit(ksq) != 0,
        BISHOP_P => bishop_attacks(to, occ) & bit(ksq) != 0,
        ROOK_P => rook_attacks(to, occ) & bit(ksq) != 0,
        QUEEN_P => queen_attacks(to, occ) & bit(ksq) != 0,
        _ => false,
    };
    if direct {
        return true;
    }

    // Discovered check: the mover vacated a line to the enemy king.
    if m.is_ep() {
        let cap = if us == WHITE { to - 8 } else { to + 8 };
        occ ^= bit(cap);
    } else if m.is_castle() {
        let (rf, rt) = match (m.flag(), us) {
            (F_KCASTLE, WHITE) => (7, 5),
            (F_KCASTLE, _) => (63, 61),
            (_, WHITE) => (0, 3),
            (_, _) => (56, 59),
        };
        occ ^= bit(rf) | bit(rt);
        return rook_attacks(rt, occ) & bit(ksq) != 0;
    }
    let ours = pos.color[us] ^ bit(from) | bit(to);
    (bishop_attacks(ksq, occ) & (pos.piece[BISHOP_P] | pos.piece[QUEEN_P]) & ours != 0)
        || (rook_attacks(ksq, occ) & (pos.piece[ROOK_P] | pos.piece[QUEEN_P]) & ours != 0)
}
