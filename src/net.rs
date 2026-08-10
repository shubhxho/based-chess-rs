//! Quantised neural evaluation.
//!
//! Architecture: `768 -> H` perspective accumulator, clipped ReLU, `2H -> 1`.
//! Both sides get their own accumulator over the same weight matrix with the
//! board flipped, and the output layer reads them side-to-move first, so the
//! network learns one function of "my position" rather than two of "white's".
//!
//! Weights are int8 and biases int16, which keeps a 32-neuron net at roughly
//! 24 KB — small enough to sit in L1 and be embedded in the binary.
//!
//! The blob is loaded from `net.bin` at compile time. If the magic does not
//! match, evaluation silently falls back to the hand-crafted function, so the
//! engine is always playable even before the network is trained.

use crate::bb::*;
use crate::pos::*;
use crate::sys::SyncCell;

const BLOB: &[u8] = include_bytes!("../net.bin");
const MAGIC: u32 = 0x4E4C_4253; // "SBLN" little-endian

/// Hidden neurons per perspective. Must match the trainer.
pub const H: usize = 32;
const IN: usize = 768;

/// Quantisation scales; the trainer applies the same ones.
const QA: i32 = 127; // feature-transformer / activation range
const QB: i32 = 64; // output weights
const SCALE: i32 = 400; // internal units -> centipawns

struct Net {
    ft_w: [i8; IN * H],
    ft_b: [i16; H],
    out_w: [i8; 2 * H],
    out_b: i32,
    loaded: bool,
}

static NET: SyncCell<Net> = SyncCell::new(Net {
    ft_w: [0; IN * H],
    ft_b: [0; H],
    out_w: [0; 2 * H],
    out_b: 0,
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
///   magic u32 | hidden u32 | ft_w i8[768*H] | ft_b i16[H] | out_w i8[2H] | out_b i32
pub fn init() {
    let need = 8 + IN * H + 2 * H + 2 * H + 4;
    if BLOB.len() < need {
        return;
    }
    let magic = u32::from_le_bytes([BLOB[0], BLOB[1], BLOB[2], BLOB[3]]);
    let hidden = u32::from_le_bytes([BLOB[4], BLOB[5], BLOB[6], BLOB[7]]) as usize;
    if magic != MAGIC || hidden != H {
        return;
    }
    let n = unsafe { NET.as_mut() };
    let mut o = 8;
    for i in 0..IN * H {
        n.ft_w[i] = BLOB[o + i] as i8;
    }
    o += IN * H;
    for i in 0..H {
        n.ft_b[i] = i16::from_le_bytes([BLOB[o + 2 * i], BLOB[o + 2 * i + 1]]);
    }
    o += 2 * H;
    for i in 0..2 * H {
        n.out_w[i] = BLOB[o + i] as i8;
    }
    o += 2 * H;
    n.out_b = i32::from_le_bytes([BLOB[o], BLOB[o + 1], BLOB[o + 2], BLOB[o + 3]]);
    n.loaded = true;
}

/// Feature index for `(colour, piece, square)` seen from `persp`'s side.
///
/// Flipping the board vertically for black — and swapping which colour counts
/// as "ours" — is what lets one weight matrix serve both perspectives.
#[inline(always)]
fn feature(persp: usize, c: usize, pt: usize, sq: usize) -> usize {
    let rel_c = if c == persp { 0 } else { 1 };
    let rel_sq = if persp == WHITE { sq } else { sq ^ 56 };
    (rel_c * 6 + pt) * 64 + rel_sq
}

/// Full accumulator refresh. At 32 neurons the whole matrix row is four NEON
/// registers, so a refresh costs about as much as an incremental update would
/// once the bookkeeping is counted.
fn accumulate(pos: &Position, persp: usize, acc: &mut [i16; H]) {
    let n = net();
    acc.copy_from_slice(&n.ft_b);
    for c in 0..2 {
        for pt in 0..6 {
            let mut b = pos.pieces(c, pt);
            while b != 0 {
                let sq = pop_lsb(&mut b);
                let f = feature(persp, c, pt, sq);
                let row = &n.ft_w[f * H..f * H + H];
                add_row(acc, row);
            }
        }
    }
}

/// `acc += row`, widening int8 to int16. Written against the NEON intrinsics
/// directly; the scalar path is only there for non-aarch64 builds.
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

/// The learned correction, in centipawns, from the side to move's point of
/// view. Added to the hand-crafted evaluation rather than replacing it.
pub fn evaluate(pos: &Position) -> i32 {
    let n = net();
    let mut us = [0i16; H];
    let mut them = [0i16; H];
    accumulate(pos, pos.stm, &mut us);
    accumulate(pos, pos.stm ^ 1, &mut them);
    let out = propagate(&us, &n.out_w[..H]) + propagate(&them, &n.out_w[H..]) + n.out_b;
    (out * SCALE / (QA * QB)).clamp(-20_000, 20_000)
}
