//! Bitboard primitives and magic sliding-attack tables.
//!
//! Attack tables live in BSS and are filled once at startup, so they cost
//! nothing in the binary image. Magics are *searched* at init rather than
//! baked in as constants: the search validates itself, which removes a whole
//! class of transcription bugs and keeps ~1 KB out of the executable.

use crate::sys::SyncCell;

pub type Bb = u64;

pub const FILE_A: Bb = 0x0101_0101_0101_0101;
pub const FILE_H: Bb = FILE_A << 7;
pub const RANK_1: Bb = 0xFF;
pub const RANK_8: Bb = RANK_1 << 56;

#[inline(always)]
pub const fn bit(sq: usize) -> Bb {
    1u64 << sq
}
#[inline(always)]
pub const fn file_of(sq: usize) -> usize {
    sq & 7
}
#[inline(always)]
pub const fn rank_of(sq: usize) -> usize {
    sq >> 3
}
/// Lowest set bit index. `rbit`+`clz` on arm64.
#[inline(always)]
pub fn lsb(b: Bb) -> usize {
    b.trailing_zeros() as usize
}
/// Pop the lowest set bit and return its index.
#[inline(always)]
pub fn pop_lsb(b: &mut Bb) -> usize {
    let s = lsb(*b);
    *b &= *b - 1;
    s
}
/// `cnt` + `addv` on arm64.
#[inline(always)]
pub fn popcount(b: Bb) -> u32 {
    b.count_ones()
}
#[inline(always)]
pub fn more_than_one(b: Bb) -> bool {
    b & b.wrapping_sub(1) != 0
}

#[inline(always)]
pub fn east(b: Bb) -> Bb {
    (b & !FILE_H) << 1
}
#[inline(always)]
pub fn west(b: Bb) -> Bb {
    (b & !FILE_A) >> 1
}

/// Forward by one rank from `side`'s point of view (0 = white).
#[inline(always)]
pub fn push(b: Bb, side: usize) -> Bb {
    if side == 0 {
        b << 8
    } else {
        b >> 8
    }
}

// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------

pub static KNIGHT: SyncCell<[Bb; 64]> = SyncCell::new([0; 64]);
pub static KING: SyncCell<[Bb; 64]> = SyncCell::new([0; 64]);
pub static PAWN: SyncCell<[[Bb; 64]; 2]> = SyncCell::new([[0; 64]; 2]);
/// Squares strictly between two aligned squares; 0 when not aligned.
pub static BETWEEN: SyncCell<[[Bb; 64]; 64]> = SyncCell::new([[0; 64]; 64]);
/// Full line through two aligned squares (both endpoints included); 0 otherwise.
pub static LINE: SyncCell<[[Bb; 64]; 64]> = SyncCell::new([[0; 64]; 64]);

const ROOK_TABLE: usize = 102_400;
const BISHOP_TABLE: usize = 5_248;
const TABLE: usize = ROOK_TABLE + BISHOP_TABLE;

static ATTACKS: SyncCell<[Bb; TABLE]> = SyncCell::new([0; TABLE]);

#[derive(Clone, Copy)]
pub struct Magic {
    pub mask: Bb,
    pub magic: u64,
    pub shift: u32,
    pub offset: u32,
}
impl Magic {
    const ZERO: Magic = Magic { mask: 0, magic: 0, shift: 0, offset: 0 };
    #[inline(always)]
    fn index(&self, occ: Bb) -> usize {
        (((occ & self.mask).wrapping_mul(self.magic)) >> self.shift) as usize + self.offset as usize
    }
}

static ROOK_M: SyncCell<[Magic; 64]> = SyncCell::new([Magic::ZERO; 64]);
static BISHOP_M: SyncCell<[Magic; 64]> = SyncCell::new([Magic::ZERO; 64]);

#[inline(always)]
pub fn knight_attacks(sq: usize) -> Bb {
    unsafe { *KNIGHT.as_ref().get_unchecked(sq) }
}
#[inline(always)]
pub fn king_attacks(sq: usize) -> Bb {
    unsafe { *KING.as_ref().get_unchecked(sq) }
}
#[inline(always)]
pub fn pawn_attacks(side: usize, sq: usize) -> Bb {
    unsafe { *PAWN.as_ref().get_unchecked(side).get_unchecked(sq) }
}
#[inline(always)]
pub fn between(a: usize, b: usize) -> Bb {
    unsafe { *BETWEEN.as_ref().get_unchecked(a).get_unchecked(b) }
}
#[inline(always)]
pub fn line(a: usize, b: usize) -> Bb {
    unsafe { *LINE.as_ref().get_unchecked(a).get_unchecked(b) }
}

#[inline(always)]
pub fn rook_attacks(sq: usize, occ: Bb) -> Bb {
    unsafe {
        let m = ROOK_M.as_ref().get_unchecked(sq);
        *ATTACKS.as_ref().get_unchecked(m.index(occ))
    }
}
#[inline(always)]
pub fn bishop_attacks(sq: usize, occ: Bb) -> Bb {
    unsafe {
        let m = BISHOP_M.as_ref().get_unchecked(sq);
        *ATTACKS.as_ref().get_unchecked(m.index(occ))
    }
}
#[inline(always)]
pub fn queen_attacks(sq: usize, occ: Bb) -> Bb {
    rook_attacks(sq, occ) | bishop_attacks(sq, occ)
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

const ROOK_DIR: [(i32, i32); 4] = [(0, 1), (0, -1), (1, 0), (-1, 0)];
const BISHOP_DIR: [(i32, i32); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

/// Ray attacks computed the slow, obviously-correct way. Only used at init and
/// as the oracle the magic search is validated against.
fn slide(sq: usize, occ: Bb, dirs: &[(i32, i32); 4]) -> Bb {
    let (f0, r0) = (file_of(sq) as i32, rank_of(sq) as i32);
    let mut out = 0u64;
    for &(df, dr) in dirs {
        let (mut f, mut r) = (f0 + df, r0 + dr);
        while (0..8).contains(&f) && (0..8).contains(&r) {
            let s = (r * 8 + f) as usize;
            out |= bit(s);
            if occ & bit(s) != 0 {
                break;
            }
            f += df;
            r += dr;
        }
    }
    out
}

#[inline(always)]
pub const fn file_bb(f: usize) -> Bb {
    FILE_A << f
}
#[inline(always)]
pub const fn rank_bb(r: usize) -> Bb {
    RANK_1 << (r * 8)
}

/// Relevant-occupancy mask: the ray squares minus the board edge. A blocker
/// sitting on the edge cannot change what is attacked beyond it, so those bits
/// carry no information and dropping them shrinks the table.
fn mask(sq: usize, dirs: &[(i32, i32); 4]) -> Bb {
    let edges = ((FILE_A | FILE_H) & !file_bb(file_of(sq))) | ((RANK_1 | RANK_8) & !rank_bb(rank_of(sq)));
    slide(sq, 0, dirs) & !edges
}

/// Deposit the bits of `idx` into the set bits of `m`, low to high.
fn scatter(idx: usize, m: Bb) -> Bb {
    let mut m = m;
    let mut out = 0u64;
    let mut i = 0;
    while m != 0 {
        let s = pop_lsb(&mut m);
        if idx & (1 << i) != 0 {
            out |= bit(s);
        }
        i += 1;
    }
    out
}

/// xorshift64*, fixed seed. Determinism matters: the same binary must produce
/// the same tables on every run.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Sparse candidate: good magics have few set bits.
    fn sparse(&mut self) -> u64 {
        self.next() & self.next() & self.next()
    }
}

fn init_magics(rook: bool, base: usize) -> usize {
    let dirs = if rook { &ROOK_DIR } else { &BISHOP_DIR };
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut offset = base;
    let mut occs = [0u64; 4096];
    let mut refs = [0u64; 4096];
    let mut used = [0u64; 4096];
    let mut epoch = [0u32; 4096];
    let mut cur_epoch = 0u32;

    for sq in 0..64 {
        let msk = mask(sq, dirs);
        let bits = popcount(msk);
        let size = 1usize << bits;
        for i in 0..size {
            occs[i] = scatter(i, msk);
            refs[i] = slide(sq, occs[i], dirs);
        }
        let shift = 64 - bits;
        loop {
            let magic = rng.sparse();
            // Cheap rejection: the top byte of mask*magic should be well spread.
            if popcount(msk.wrapping_mul(magic) >> 56) < 6 {
                continue;
            }
            cur_epoch += 1;
            let mut ok = true;
            for i in 0..size {
                let j = ((occs[i].wrapping_mul(magic)) >> shift) as usize;
                if epoch[j] != cur_epoch {
                    epoch[j] = cur_epoch;
                    used[j] = refs[i];
                } else if used[j] != refs[i] {
                    ok = false;
                    break;
                }
            }
            if ok {
                let m = Magic { mask: msk, magic, shift, offset: offset as u32 };
                unsafe {
                    if rook {
                        ROOK_M.as_mut()[sq] = m;
                    } else {
                        BISHOP_M.as_mut()[sq] = m;
                    }
                    let att = ATTACKS.as_mut();
                    for i in 0..size {
                        att[m.index(occs[i])] = refs[i];
                    }
                }
                offset += size;
                break;
            }
        }
    }
    offset
}

pub fn init() {
    unsafe {
        let kn = KNIGHT.as_mut();
        let kg = KING.as_mut();
        let pw = PAWN.as_mut();
        for sq in 0..64usize {
            let (f, r) = (file_of(sq) as i32, rank_of(sq) as i32);
            let mut n = 0u64;
            for &(df, dr) in &[(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)] {
                let (nf, nr) = (f + df, r + dr);
                if (0..8).contains(&nf) && (0..8).contains(&nr) {
                    n |= bit((nr * 8 + nf) as usize);
                }
            }
            kn[sq] = n;

            let mut k = 0u64;
            for df in -1..=1i32 {
                for dr in -1..=1i32 {
                    if df == 0 && dr == 0 {
                        continue;
                    }
                    let (nf, nr) = (f + df, r + dr);
                    if (0..8).contains(&nf) && (0..8).contains(&nr) {
                        k |= bit((nr * 8 + nf) as usize);
                    }
                }
            }
            kg[sq] = k;

            let b = bit(sq);
            pw[0][sq] = east(b) << 8 | west(b) << 8;
            pw[1][sq] = east(b) >> 8 | west(b) >> 8;
        }
    }

    let after_rooks = init_magics(true, 0);
    let end = init_magics(false, after_rooks);
    debug_assert!(end <= TABLE);
    let _ = end;

    unsafe {
        let bt = BETWEEN.as_mut();
        let ln = LINE.as_mut();
        for a in 0..64usize {
            for dirs in [&ROOK_DIR, &BISHOP_DIR] {
                let full = slide(a, 0, dirs);
                let mut t = full;
                while t != 0 {
                    let b = pop_lsb(&mut t);
                    // Squares between = both rays restricted by the other endpoint.
                    bt[a][b] = slide(a, bit(b), dirs) & slide(b, bit(a), dirs);
                    ln[a][b] = (full & slide(b, 0, dirs)) | bit(a) | bit(b);
                }
            }
        }
    }
}
