//! Transposition table.
//!
//! Backing memory comes straight from `mmap`, so the table is page-aligned and
//! the engine still needs no allocator. Entries are 16 bytes and clusters are
//! 4 entries, which makes a cluster exactly one 64-byte cache line: a probe
//! touches memory once.
//!
//! An entry is two atomic words rather than a struct of fields, because the
//! table is the one thing every search thread shares. The scheme is the classic
//! lockless one: everything an entry says is packed into `data`, and `key`
//! holds the position key *exclusive-ored* with that same word. A reader
//! recovers the key as `key ^ data`, so a pair torn by a concurrent write —
//! one word from the old entry, one from the new — reconstructs a key that
//! matches nothing and is simply treated as a miss. No lock, no torn entry ever
//! believed.
//!
//! It is also strictly better than what it replaces even with one thread. The
//! old entry spent 32 bits on the key and 32 on padding; this spends the full
//! 64, so a false hit from a key collision went from one in four billion to one
//! in eighteen quintillion.

use core::sync::atomic::{AtomicU64, Ordering};

use crate::eval::{MATE, MATE_IN_MAX};
use crate::pos::Move;
use crate::sys::{mmap, munmap, SyncCell};

pub const BOUND_NONE: u8 = 0;
pub const BOUND_UPPER: u8 = 1;
pub const BOUND_LOWER: u8 = 2;
pub const BOUND_EXACT: u8 = 3;

/// `mv:16 | score:16 | eval:16 | depth:8 | gen:6 bound:2`, which is exactly
/// sixty-four bits with nothing left over.
#[inline(always)]
fn pack(mv: u16, score: i32, eval: i32, depth: i32, gen_bound: u8) -> u64 {
    (mv as u64)
        | ((score as i16 as u16 as u64) << 16)
        | ((eval as i16 as u16 as u64) << 32)
        | ((depth as i8 as u8 as u64) << 48)
        | ((gen_bound as u64) << 56)
}

#[inline(always)]
fn d_mv(d: u64) -> u16 {
    d as u16
}
#[inline(always)]
fn d_score(d: u64) -> i32 {
    (d >> 16) as u16 as i16 as i32
}
#[inline(always)]
fn d_eval(d: u64) -> i32 {
    (d >> 32) as u16 as i16 as i32
}
#[inline(always)]
fn d_depth(d: u64) -> i32 {
    (d >> 48) as u8 as i8 as i32
}
#[inline(always)]
fn d_gen_bound(d: u64) -> u8 {
    (d >> 56) as u8
}

#[repr(C)]
struct Entry {
    /// The position key, exclusive-ored with `data`.
    key: AtomicU64,
    data: AtomicU64,
}

const CLUSTER: usize = 4;
const ENTRY_BYTES: usize = 16;

pub struct Tt {
    mem: *mut u8,
    bytes: usize,
    clusters: u64,
    gen: u8,
}

static TT: SyncCell<Tt> = SyncCell::new(Tt { mem: core::ptr::null_mut(), bytes: 0, clusters: 0, gen: 0 });

#[inline(always)]
pub fn tt() -> &'static mut Tt {
    unsafe { TT.as_mut() }
}

pub struct Hit {
    pub mv: Move,
    pub score: i32,
    pub eval: i32,
    pub depth: i32,
    pub bound: u8,
}

impl Tt {
    pub fn resize(&mut self, mb: usize) {
        if !self.mem.is_null() {
            munmap(self.mem, self.bytes);
        }
        let mut bytes = mb.max(1) * 1024 * 1024;
        // Round down to a power-of-two cluster count so indexing is a mask.
        let mut clusters = bytes / (CLUSTER * ENTRY_BYTES);
        clusters = 1usize << (usize::BITS - 1 - clusters.max(1).leading_zeros()) as usize;
        bytes = clusters * CLUSTER * ENTRY_BYTES;
        self.mem = mmap(bytes);
        self.bytes = if self.mem.is_null() { 0 } else { bytes };
        self.clusters = if self.mem.is_null() { 0 } else { clusters as u64 };
        self.gen = 0;
    }

    /// Fresh mapping from the kernel is already zeroed; this is for `ucinewgame`
    /// when the mapping is reused.
    pub fn clear(&mut self) {
        if self.mem.is_null() {
            return;
        }
        unsafe {
            core::ptr::write_bytes(self.mem, 0, self.bytes);
        }
        self.gen = 0;
    }

    pub fn new_search(&mut self) {
        self.gen = self.gen.wrapping_add(1) & 0x3F;
    }

    #[inline(always)]
    fn cluster(&self, key: u64) -> *const Entry {
        // High-bit multiply: uses the *whole* key, unlike a low-bit mask, so it
        // stays well distributed when the low bits are correlated.
        let idx = ((key as u128 * self.clusters as u128) >> 64) as usize;
        unsafe { self.mem.add(idx * CLUSTER * ENTRY_BYTES) as *const Entry }
    }

    #[inline(always)]
    pub fn prefetch(&self, key: u64) {
        if self.clusters == 0 {
            return;
        }
        let p = self.cluster(key) as *const u8;
        // One hint per architecture, and nothing at all on the ones we have no
        // instruction for. The probe that follows is correct either way; this
        // only decides whether the cache line is already on its way.
        #[cfg(target_arch = "aarch64")]
        unsafe {
            core::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, readonly));
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            core::arch::asm!("prefetcht0 [{0}]", in(reg) p, options(nostack, readonly));
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let _ = p;
    }

    pub fn probe(&self, key: u64, ply: usize) -> Option<Hit> {
        if self.clusters == 0 {
            return None;
        }
        let c = self.cluster(key);
        for i in 0..CLUSTER {
            let e = unsafe { &*c.add(i) };
            let data = e.data.load(Ordering::Relaxed);
            // `key ^ data` is the whole check: it proves the position matches
            // *and* that these two words belong to the same write.
            if e.key.load(Ordering::Relaxed) ^ data == key && d_gen_bound(data) & 3 != BOUND_NONE {
                return Some(Hit {
                    mv: Move(d_mv(data)),
                    score: from_tt(d_score(data), ply),
                    eval: d_eval(data),
                    depth: d_depth(data),
                    bound: d_gen_bound(data) & 3,
                });
            }
        }
        None
    }

    /// Clippy objects to the arity; the alternative is a parameter struct that
    /// exists only to be destructured at the single call site. The names are
    /// unambiguous and this is the hot path, so it stays as it is.
    #[allow(clippy::too_many_arguments)]
    pub fn store(&self, key: u64, mv: Move, score: i32, eval: i32, depth: i32, bound: u8, ply: usize) {
        if self.clusters == 0 {
            return;
        }
        let c = self.cluster(key);
        let mut slot = 0usize;
        let mut slot_data = 0u64;
        let mut same = false;
        let mut best_val = i32::MAX;
        for i in 0..CLUSTER {
            let e = unsafe { &*c.add(i) };
            let data = e.data.load(Ordering::Relaxed);
            let hit = e.key.load(Ordering::Relaxed) ^ data == key;
            if hit || d_gen_bound(data) & 3 == BOUND_NONE {
                // Same position, or a free slot: nothing to weigh up.
                slot = i;
                slot_data = data;
                same = hit;
                break;
            }
            // Depth is the main currency; entries from older searches are
            // discounted so a long game does not fossilise the table.
            let age = (self.gen as i32 + 64 - (d_gen_bound(data) >> 2) as i32) & 0x3F;
            let val = d_depth(data) - age * 4;
            if val < best_val {
                best_val = val;
                slot = i;
                slot_data = data;
            }
        }

        // Keep the old move when the new probe has none: a stale hint still
        // orders better than nothing.
        let keep = if mv.is_null() && same { d_mv(slot_data) } else { mv.0 };
        if !same || bound == BOUND_EXACT || depth + 4 > d_depth(slot_data) {
            let data = pack(keep, to_tt(score, ply), eval.clamp(-32_000, 32_000), depth.clamp(-8, 127), (self.gen << 2) | bound);
            let e = unsafe { &*c.add(slot) };
            // Data first, then the key that authenticates it. A reader that
            // catches the pair mid-write sees a key that does not reconstruct
            // and treats the slot as a miss, which is the whole point.
            e.data.store(data, Ordering::Relaxed);
            e.key.store(key ^ data, Ordering::Relaxed);
        } else if keep != d_mv(slot_data) {
            // Only the move improved. Rewrite the pair so the two words stay
            // consistent with each other.
            let data = (slot_data & !0xFFFF) | keep as u64;
            let e = unsafe { &*c.add(slot) };
            e.data.store(data, Ordering::Relaxed);
            e.key.store(key ^ data, Ordering::Relaxed);
        }
    }

    /// Rough fill estimate over the first 1000 clusters, for `info hashfull`.
    pub fn hashfull(&self) -> usize {
        if self.clusters == 0 {
            return 0;
        }
        let n = 1000usize.min(self.clusters as usize);
        let mut used = 0;
        for i in 0..n {
            let c = unsafe { self.mem.add(i * CLUSTER * ENTRY_BYTES) as *const Entry };
            for j in 0..CLUSTER {
                let data = unsafe { (*c.add(j)).data.load(Ordering::Relaxed) };
                let gb = d_gen_bound(data);
                if gb & 3 != BOUND_NONE && (gb >> 2) == self.gen {
                    used += 1;
                }
            }
        }
        used * 1000 / (n * CLUSTER)
    }
}

/// Mate scores are stored relative to the node, not the root, or a transposed
/// position would report the wrong distance to mate.
#[inline(always)]
fn to_tt(score: i32, ply: usize) -> i32 {
    if score >= MATE_IN_MAX {
        score + ply as i32
    } else if score <= -MATE_IN_MAX {
        score - ply as i32
    } else {
        score
    }
}
#[inline(always)]
fn from_tt(score: i32, ply: usize) -> i32 {
    if score >= MATE_IN_MAX {
        (score - ply as i32).min(MATE)
    } else if score <= -MATE_IN_MAX {
        (score + ply as i32).max(-MATE)
    } else {
        score
    }
}
