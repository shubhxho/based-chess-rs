//! Transposition table.
//!
//! Backing memory comes straight from `mmap`, so the table is page-aligned and
//! the engine still needs no allocator. Entries are 16 bytes and clusters are
//! 4 entries, which makes a cluster exactly one 64-byte cache line: a probe
//! touches memory once.

use crate::eval::{MATE, MATE_IN_MAX};
use crate::pos::Move;
use crate::sys::{mmap, munmap, SyncCell};

pub const BOUND_NONE: u8 = 0;
pub const BOUND_UPPER: u8 = 1;
pub const BOUND_LOWER: u8 = 2;
pub const BOUND_EXACT: u8 = 3;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Entry {
    key: u32,
    mv: u16,
    score: i16,
    eval: i16,
    depth: i8,
    /// generation in the high 6 bits, bound in the low 2
    gen_bound: u8,
    _pad: [u8; 4],
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
    fn cluster(&self, key: u64) -> *mut Entry {
        // High-bit multiply: uses the *whole* key, unlike a low-bit mask, so it
        // stays well distributed when the low bits are correlated.
        let idx = ((key as u128 * self.clusters as u128) >> 64) as usize;
        unsafe { self.mem.add(idx * CLUSTER * ENTRY_BYTES) as *mut Entry }
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
        let k32 = (key >> 32) as u32;
        let c = self.cluster(key);
        for i in 0..CLUSTER {
            let e = unsafe { &*c.add(i) };
            if e.key == k32 && e.gen_bound & 3 != BOUND_NONE {
                return Some(Hit {
                    mv: Move(e.mv),
                    score: from_tt(e.score as i32, ply),
                    eval: e.eval as i32,
                    depth: e.depth as i32,
                    bound: e.gen_bound & 3,
                });
            }
        }
        None
    }

    /// Clippy objects to the arity; the alternative is a parameter struct that
    /// exists only to be destructured at the single call site. The names are
    /// unambiguous and this is the hot path, so it stays as it is.
    #[allow(clippy::too_many_arguments)]
    pub fn store(&mut self, key: u64, mv: Move, score: i32, eval: i32, depth: i32, bound: u8, ply: usize) {
        if self.clusters == 0 {
            return;
        }
        let k32 = (key >> 32) as u32;
        let c = self.cluster(key);
        let mut best = 0usize;
        let mut best_val = i32::MAX;
        for i in 0..CLUSTER {
            let e = unsafe { &*c.add(i) };
            if e.key == k32 || e.gen_bound & 3 == BOUND_NONE {
                // Same position, or a free slot: nothing to weigh up.
                best = i;
                break;
            }
            // Depth is the main currency; entries from older searches are
            // discounted so a long game does not fossilise the table.
            let age = (self.gen as i32 + 64 - (e.gen_bound >> 2) as i32) & 0x3F;
            let val = e.depth as i32 - age * 4;
            if val < best_val {
                best_val = val;
                best = i;
            }
        }
        let e = unsafe { &mut *c.add(best) };
        // Keep the old move when the new probe has none: a stale hint still
        // orders better than nothing.
        if !mv.is_null() || e.key != k32 {
            e.mv = mv.0;
        }
        if e.key != k32 || bound == BOUND_EXACT || depth + 4 > e.depth as i32 {
            e.key = k32;
            e.score = to_tt(score, ply) as i16;
            e.eval = eval.clamp(-32_000, 32_000) as i16;
            e.depth = depth.clamp(-8, 127) as i8;
            e.gen_bound = (self.gen << 2) | bound;
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
                let e = unsafe { &*c.add(j) };
                if e.gen_bound & 3 != BOUND_NONE && (e.gen_bound >> 2) == self.gen {
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
