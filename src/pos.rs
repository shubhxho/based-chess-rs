//! Position: bitboards + mailbox, incremental Zobrist, make/unmake.

use crate::bb::*;
use crate::sys::SyncCell;

pub const PAWN_P: usize = 0;
pub const KNIGHT_P: usize = 1;
pub const BISHOP_P: usize = 2;
pub const ROOK_P: usize = 3;
pub const QUEEN_P: usize = 4;
pub const KING_P: usize = 5;
pub const NONE: u8 = 6;

pub const WHITE: usize = 0;
pub const BLACK: usize = 1;

// Castle-rights bits.
pub const WK: u8 = 1;
pub const WQ: u8 = 2;
pub const BK: u8 = 4;
pub const BQ: u8 = 8;

// ---------------------------------------------------------------------------
// Moves. 16 bits: from(6) | to(6) | flag(4).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(test, derive(Debug))]
pub struct Move(pub u16);

pub const F_QUIET: u16 = 0;
pub const F_DOUBLE: u16 = 1;
pub const F_KCASTLE: u16 = 2;
pub const F_QCASTLE: u16 = 3;
pub const F_CAPTURE: u16 = 4;
pub const F_EP: u16 = 5;
pub const F_PROMO: u16 = 8; // +0..3 = N,B,R,Q ; |F_CAPTURE for promo-capture

impl Move {
    pub const NULL: Move = Move(0);
    #[inline(always)]
    pub const fn new(from: usize, to: usize, flag: u16) -> Move {
        Move(from as u16 | ((to as u16) << 6) | (flag << 12))
    }
    #[inline(always)]
    pub const fn from(self) -> usize {
        (self.0 & 63) as usize
    }
    #[inline(always)]
    pub const fn to(self) -> usize {
        ((self.0 >> 6) & 63) as usize
    }
    #[inline(always)]
    pub const fn flag(self) -> u16 {
        self.0 >> 12
    }
    #[inline(always)]
    pub const fn is_capture(self) -> bool {
        self.flag() & F_CAPTURE != 0
    }
    #[inline(always)]
    pub const fn is_promo(self) -> bool {
        self.flag() & F_PROMO != 0
    }
    #[inline(always)]
    pub const fn is_ep(self) -> bool {
        self.flag() == F_EP
    }
    #[inline(always)]
    pub const fn is_castle(self) -> bool {
        matches!(self.flag(), F_KCASTLE | F_QCASTLE)
    }
    /// Promotion piece type; only meaningful when `is_promo()`.
    #[inline(always)]
    pub const fn promo(self) -> usize {
        ((self.flag() & 3) + 1) as usize
    }
    #[inline(always)]
    pub const fn is_null(self) -> bool {
        self.0 == 0
    }
}

// ---------------------------------------------------------------------------
// Zobrist
// ---------------------------------------------------------------------------

pub struct Zobrist {
    pub psq: [[[u64; 64]; 6]; 2],
    pub castle: [u64; 16],
    pub ep: [u64; 8],
    pub side: u64,
}
static ZOB: SyncCell<Zobrist> = SyncCell::new(Zobrist {
    psq: [[[0; 64]; 6]; 2],
    castle: [0; 16],
    ep: [0; 8],
    side: 0,
});

#[inline(always)]
fn zob() -> &'static Zobrist {
    unsafe { ZOB.as_ref() }
}

pub fn init_zobrist() {
    // splitmix64, fixed seed.
    let mut s: u64 = 0x1234_5678_9ABC_DEF0;
    let mut next = || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    unsafe {
        let z = ZOB.as_mut();
        for c in 0..2 {
            for p in 0..6 {
                for s in 0..64 {
                    z.psq[c][p][s] = next();
                }
            }
        }
        for i in 0..16 {
            z.castle[i] = next();
        }
        for i in 0..8 {
            z.ep[i] = next();
        }
        z.side = next();
    }
}

// ---------------------------------------------------------------------------
// Position
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct Undo {
    pub castle: u8,
    pub ep: u8, // 64 = none
    pub halfmove: u16,
    pub captured: u8,
    pub key: u64,
    pub pawn_key: u64,
    pub checkers: Bb,
}

pub const MAX_PLY: usize = 128;
const HIST: usize = 4096;

#[derive(Clone)]
pub struct Position {
    pub piece: [Bb; 6],
    pub color: [Bb; 2],
    pub mailbox: [u8; 64],
    pub stm: usize,
    pub castle: u8,
    pub ep: u8,
    pub halfmove: u16,
    pub ply: usize,
    pub key: u64,
    pub pawn_key: u64,
    pub checkers: Bb,
    pub stack: [Undo; MAX_PLY + 8],
    /// Position keys of the whole game, for repetition detection.
    pub hist: [u64; HIST],
    pub hist_len: usize,
    /// Index in `hist` of the last irreversible move.
    pub root_ply: usize,
}

impl Position {
    pub const fn empty() -> Position {
        Position {
            piece: [0; 6],
            color: [0; 2],
            mailbox: [NONE; 64],
            stm: WHITE,
            castle: 0,
            ep: 64,
            halfmove: 0,
            ply: 0,
            key: 0,
            pawn_key: 0,
            checkers: 0,
            stack: [Undo {
                castle: 0,
                ep: 64,
                halfmove: 0,
                captured: NONE,
                key: 0,
                pawn_key: 0,
                checkers: 0,
            }; MAX_PLY + 8],
            hist: [0; HIST],
            hist_len: 0,
            root_ply: 0,
        }
    }

    #[inline(always)]
    pub fn occ(&self) -> Bb {
        self.color[0] | self.color[1]
    }
    #[inline(always)]
    pub fn pieces(&self, c: usize, p: usize) -> Bb {
        self.piece[p] & self.color[c]
    }
    #[inline(always)]
    pub fn king_sq(&self, c: usize) -> usize {
        lsb(self.piece[KING_P] & self.color[c])
    }
    #[inline(always)]
    pub fn piece_at(&self, sq: usize) -> u8 {
        unsafe { *self.mailbox.get_unchecked(sq) }
    }
    #[inline(always)]
    pub fn color_at(&self, sq: usize) -> usize {
        (self.color[BLACK] >> sq) as usize & 1
    }
    /// Non-pawn material present for `c`; gates null-move pruning.
    #[inline(always)]
    pub fn has_big_pieces(&self, c: usize) -> bool {
        self.color[c] & !(self.piece[PAWN_P] | self.piece[KING_P]) != 0
    }

    #[inline(always)]
    fn put(&mut self, c: usize, p: usize, sq: usize) {
        let b = bit(sq);
        self.piece[p] |= b;
        self.color[c] |= b;
        self.mailbox[sq] = p as u8;
        self.key ^= zob().psq[c][p][sq];
        if p == PAWN_P {
            self.pawn_key ^= zob().psq[c][p][sq];
        }
    }
    #[inline(always)]
    fn remove(&mut self, c: usize, p: usize, sq: usize) {
        let b = bit(sq);
        self.piece[p] ^= b;
        self.color[c] ^= b;
        self.mailbox[sq] = NONE;
        self.key ^= zob().psq[c][p][sq];
        if p == PAWN_P {
            self.pawn_key ^= zob().psq[c][p][sq];
        }
    }
    #[inline(always)]
    fn shift(&mut self, c: usize, p: usize, from: usize, to: usize) {
        let b = bit(from) | bit(to);
        self.piece[p] ^= b;
        self.color[c] ^= b;
        self.mailbox[from] = NONE;
        self.mailbox[to] = p as u8;
        let k = zob().psq[c][p][from] ^ zob().psq[c][p][to];
        self.key ^= k;
        if p == PAWN_P {
            self.pawn_key ^= k;
        }
    }

    /// Every piece of `c` that attacks `sq`, given `occ`.
    #[inline(always)]
    pub fn attackers_to(&self, sq: usize, occ: Bb) -> Bb {
        (pawn_attacks(WHITE, sq) & self.pieces(BLACK, PAWN_P))
            | (pawn_attacks(BLACK, sq) & self.pieces(WHITE, PAWN_P))
            | (knight_attacks(sq) & self.piece[KNIGHT_P])
            | (king_attacks(sq) & self.piece[KING_P])
            | (bishop_attacks(sq, occ) & (self.piece[BISHOP_P] | self.piece[QUEEN_P]))
            | (rook_attacks(sq, occ) & (self.piece[ROOK_P] | self.piece[QUEEN_P]))
    }

    #[inline(always)]
    pub fn attacked_by(&self, c: usize, sq: usize, occ: Bb) -> bool {
        let them = self.color[c];
        if pawn_attacks(c ^ 1, sq) & self.piece[PAWN_P] & them != 0 {
            return true;
        }
        if knight_attacks(sq) & self.piece[KNIGHT_P] & them != 0 {
            return true;
        }
        if king_attacks(sq) & self.piece[KING_P] & them != 0 {
            return true;
        }
        if bishop_attacks(sq, occ) & (self.piece[BISHOP_P] | self.piece[QUEEN_P]) & them != 0 {
            return true;
        }
        rook_attacks(sq, occ) & (self.piece[ROOK_P] | self.piece[QUEEN_P]) & them != 0
    }

    #[inline(always)]
    pub fn in_check(&self) -> bool {
        self.checkers != 0
    }

    pub fn compute_checkers(&mut self) {
        let ksq = self.king_sq(self.stm);
        self.checkers = self.attackers_to(ksq, self.occ()) & self.color[self.stm ^ 1];
    }

    /// Absolutely pinned pieces of the side to move.
    pub fn pinned(&self, c: usize) -> Bb {
        let ksq = self.king_sq(c);
        let them = self.color[c ^ 1];
        let occ = self.occ();
        let mut pinners = (rook_attacks(ksq, them) & (self.piece[ROOK_P] | self.piece[QUEEN_P]) & them)
            | (bishop_attacks(ksq, them) & (self.piece[BISHOP_P] | self.piece[QUEEN_P]) & them);
        let mut pin = 0u64;
        while pinners != 0 {
            let s = pop_lsb(&mut pinners);
            let b = between(ksq, s) & occ;
            if b != 0 && !more_than_one(b) {
                pin |= b & self.color[c];
            }
        }
        pin
    }

    // -----------------------------------------------------------------------

    pub fn make(&mut self, m: Move) {
        let us = self.stm;
        let them = us ^ 1;
        let from = m.from();
        let to = m.to();
        let pt = self.piece_at(from) as usize;
        let flag = m.flag();

        let u = &mut self.stack[self.ply];
        u.castle = self.castle;
        u.ep = self.ep;
        u.halfmove = self.halfmove;
        u.key = self.key;
        u.pawn_key = self.pawn_key;
        u.checkers = self.checkers;
        u.captured = NONE;

        if self.ep != 64 {
            self.key ^= zob().ep[file_of(self.ep as usize)];
            self.ep = 64;
        }
        self.key ^= zob().castle[self.castle as usize];
        self.halfmove += 1;

        match flag {
            F_KCASTLE => {
                let (rf, rt) = if us == WHITE { (7, 5) } else { (63, 61) };
                self.shift(us, KING_P, from, to);
                self.shift(us, ROOK_P, rf, rt);
            }
            F_QCASTLE => {
                let (rf, rt) = if us == WHITE { (0, 3) } else { (56, 59) };
                self.shift(us, KING_P, from, to);
                self.shift(us, ROOK_P, rf, rt);
            }
            F_EP => {
                let cap = if us == WHITE { to - 8 } else { to + 8 };
                self.remove(them, PAWN_P, cap);
                self.shift(us, PAWN_P, from, to);
                self.stack[self.ply].captured = PAWN_P as u8;
                self.halfmove = 0;
            }
            _ => {
                if m.is_capture() {
                    let cp = self.piece_at(to) as usize;
                    self.stack[self.ply].captured = cp as u8;
                    self.remove(them, cp, to);
                    self.halfmove = 0;
                }
                if m.is_promo() {
                    self.remove(us, PAWN_P, from);
                    self.put(us, m.promo(), to);
                    self.halfmove = 0;
                } else {
                    self.shift(us, pt, from, to);
                    if pt == PAWN_P {
                        self.halfmove = 0;
                        if flag == F_DOUBLE {
                            // Only record the ep square if a capture is actually
                            // available; otherwise identical positions would get
                            // different keys and repetition detection would miss.
                            let epsq = if us == WHITE { to - 8 } else { to + 8 };
                            if pawn_attacks(us, epsq) & self.pieces(them, PAWN_P) != 0 {
                                self.ep = epsq as u8;
                                self.key ^= zob().ep[file_of(epsq)];
                            }
                        }
                    }
                }
            }
        }

        // Castling rights die when a king or rook leaves, or a rook is captured.
        self.castle &= CASTLE_MASK[from] & CASTLE_MASK[to];
        self.key ^= zob().castle[self.castle as usize];

        self.stm = them;
        self.key ^= zob().side;
        self.ply += 1;

        self.compute_checkers();

        self.hist[self.hist_len] = self.key;
        self.hist_len += 1;
    }

    pub fn unmake(&mut self, m: Move) {
        self.hist_len -= 1;
        self.ply -= 1;
        let them = self.stm;
        let us = them ^ 1;
        self.stm = us;

        let from = m.from();
        let to = m.to();
        let flag = m.flag();
        let u = self.stack[self.ply];

        match flag {
            F_KCASTLE => {
                let (rf, rt) = if us == WHITE { (7, 5) } else { (63, 61) };
                self.move_back(us, KING_P, to, from);
                self.move_back(us, ROOK_P, rt, rf);
            }
            F_QCASTLE => {
                let (rf, rt) = if us == WHITE { (0, 3) } else { (56, 59) };
                self.move_back(us, KING_P, to, from);
                self.move_back(us, ROOK_P, rt, rf);
            }
            F_EP => {
                self.move_back(us, PAWN_P, to, from);
                let cap = if us == WHITE { to - 8 } else { to + 8 };
                self.put_back(them, PAWN_P, cap);
            }
            _ => {
                if m.is_promo() {
                    let pp = m.promo();
                    let b = bit(to);
                    self.piece[pp] ^= b;
                    self.color[us] ^= b;
                    self.mailbox[to] = NONE;
                    self.put_back(us, PAWN_P, from);
                } else {
                    self.move_back(us, self.piece_at(to) as usize, to, from);
                }
                if u.captured != NONE {
                    self.put_back(them, u.captured as usize, to);
                }
            }
        }

        self.castle = u.castle;
        self.ep = u.ep;
        self.halfmove = u.halfmove;
        self.key = u.key;
        self.pawn_key = u.pawn_key;
        self.checkers = u.checkers;
    }

    /// Board-only piece motion. Hash keys are restored wholesale in `unmake`,
    /// so the incremental xor is deliberately skipped here.
    #[inline(always)]
    fn move_back(&mut self, c: usize, p: usize, from: usize, to: usize) {
        let b = bit(from) | bit(to);
        self.piece[p] ^= b;
        self.color[c] ^= b;
        self.mailbox[from] = NONE;
        self.mailbox[to] = p as u8;
    }
    #[inline(always)]
    fn put_back(&mut self, c: usize, p: usize, sq: usize) {
        let b = bit(sq);
        self.piece[p] |= b;
        self.color[c] |= b;
        self.mailbox[sq] = p as u8;
    }

    pub fn make_null(&mut self) {
        let u = &mut self.stack[self.ply];
        u.castle = self.castle;
        u.ep = self.ep;
        u.halfmove = self.halfmove;
        u.key = self.key;
        u.pawn_key = self.pawn_key;
        u.checkers = self.checkers;
        u.captured = NONE;

        if self.ep != 64 {
            self.key ^= zob().ep[file_of(self.ep as usize)];
            self.ep = 64;
        }
        self.stm ^= 1;
        self.key ^= zob().side;
        self.halfmove += 1;
        self.ply += 1;
        self.checkers = 0;
        self.hist[self.hist_len] = self.key;
        self.hist_len += 1;
    }

    pub fn unmake_null(&mut self) {
        self.hist_len -= 1;
        self.ply -= 1;
        self.stm ^= 1;
        let u = self.stack[self.ply];
        self.ep = u.ep;
        self.halfmove = u.halfmove;
        self.key = u.key;
        self.checkers = u.checkers;
    }

    /// Draw by repetition or the fifty-move rule.
    ///
    /// A single repetition inside the search tree is treated as a draw: the
    /// side that can force it once can force it again, so searching further is
    /// wasted work. Repetitions before the root need two hits.
    pub fn is_draw(&self, ply: usize) -> bool {
        if self.halfmove >= 100 {
            return true;
        }
        if self.halfmove < 4 || self.hist_len < 5 {
            return false;
        }
        let end = self.hist_len - 1;
        let limit = (self.halfmove as usize).min(end);
        let mut count = 0;
        let mut i = 4;
        while i <= limit {
            if self.hist[end - i] == self.key {
                // A repetition that happened *inside* the current search tree
                // (distance smaller than the ply we are at) counts once: the
                // side to move can simply repeat again.
                if i < ply {
                    return true;
                }
                count += 1;
                if count >= 2 {
                    return true;
                }
            }
            i += 2;
        }
        false
    }

    /// Insufficient material: K vs K, K+minor vs K.
    pub fn is_material_draw(&self) -> bool {
        if self.piece[PAWN_P] | self.piece[ROOK_P] | self.piece[QUEEN_P] != 0 {
            return false;
        }
        let minors = self.piece[KNIGHT_P] | self.piece[BISHOP_P];
        popcount(minors) <= 1
    }

    // -----------------------------------------------------------------------

    pub fn set_startpos(&mut self) {
        self.set_fen(b"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    }

    pub fn set_fen(&mut self, fen: &[u8]) {
        self.piece = [0; 6];
        self.color = [0; 2];
        self.mailbox = [NONE; 64];
        self.castle = 0;
        self.ep = 64;
        self.halfmove = 0;
        self.ply = 0;
        self.key = 0;
        self.pawn_key = 0;
        self.hist_len = 0;
        self.root_ply = 0;

        let mut i = 0usize;
        let mut sq: i32 = 56;
        while i < fen.len() && fen[i] != b' ' {
            let c = fen[i];
            match c {
                b'/' => sq -= 16,
                b'1'..=b'8' => sq += (c - b'0') as i32,
                _ => {
                    let (col, pt) = match c {
                        b'P' => (WHITE, PAWN_P),
                        b'N' => (WHITE, KNIGHT_P),
                        b'B' => (WHITE, BISHOP_P),
                        b'R' => (WHITE, ROOK_P),
                        b'Q' => (WHITE, QUEEN_P),
                        b'K' => (WHITE, KING_P),
                        b'p' => (BLACK, PAWN_P),
                        b'n' => (BLACK, KNIGHT_P),
                        b'b' => (BLACK, BISHOP_P),
                        b'r' => (BLACK, ROOK_P),
                        b'q' => (BLACK, QUEEN_P),
                        b'k' => (BLACK, KING_P),
                        _ => {
                            i += 1;
                            continue;
                        }
                    };
                    if (0..64).contains(&sq) {
                        self.put(col, pt, sq as usize);
                    }
                    sq += 1;
                }
            }
            i += 1;
        }
        i += 1;
        self.stm = if i < fen.len() && fen[i] == b'b' { BLACK } else { WHITE };
        i += 2;
        while i < fen.len() && fen[i] != b' ' {
            match fen[i] {
                b'K' => self.castle |= WK,
                b'Q' => self.castle |= WQ,
                b'k' => self.castle |= BK,
                b'q' => self.castle |= BQ,
                _ => {}
            }
            i += 1;
        }
        i += 1;
        if i < fen.len() && fen[i] != b'-' && i + 1 < fen.len() {
            let f = (fen[i] - b'a') as usize;
            let r = (fen[i + 1] - b'1') as usize;
            if f < 8 && r < 8 {
                let epsq = r * 8 + f;
                // Same rule as in `make`: only keep it when capturable. The
                // capturing pawns belong to the side to move and stand on the
                // squares from which they attack `epsq`.
                if pawn_attacks(self.stm ^ 1, epsq) & self.pieces(self.stm, PAWN_P) != 0 {
                    self.ep = epsq as u8;
                }
            }
            i += 2;
        } else {
            i += 2;
        }
        while i < fen.len() && fen[i] == b' ' {
            i += 1;
        }
        let mut hm = 0u16;
        while i < fen.len() && fen[i].is_ascii_digit() {
            hm = hm.wrapping_mul(10) + (fen[i] - b'0') as u16;
            i += 1;
        }
        self.halfmove = hm;

        if self.stm == BLACK {
            self.key ^= zob().side;
        }
        self.key ^= zob().castle[self.castle as usize];
        if self.ep != 64 {
            self.key ^= zob().ep[file_of(self.ep as usize)];
        }
        self.compute_checkers();
        self.hist[0] = self.key;
        self.hist_len = 1;
    }
}

/// `castle & CASTLE_MASK[sq]` clears exactly the rights that a move touching
/// `sq` invalidates.
static CASTLE_MASK: [u8; 64] = {
    let mut m = [0xFu8; 64];
    m[0] = 0xF & !WQ;
    m[4] = 0xF & !(WK | WQ);
    m[7] = 0xF & !WK;
    m[56] = 0xF & !BQ;
    m[60] = 0xF & !(BK | BQ);
    m[63] = 0xF & !BK;
    m
};
