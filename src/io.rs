//! Buffered stdout and integer formatting, built from nothing.
//!
//! `core` has no formatting machinery worth using here and `format!` needs an
//! allocator, so numbers are rendered by hand into a fixed buffer and flushed
//! with a single `write` syscall per line.

use crate::sys;

const CAP: usize = 8192;

pub struct Out {
    buf: [u8; CAP],
    len: usize,
}

impl Out {
    pub const fn new() -> Out {
        Out { buf: [0; CAP], len: 0 }
    }
    #[inline]
    pub fn s(&mut self, b: &[u8]) -> &mut Self {
        if self.len + b.len() > CAP {
            self.flush();
        }
        let n = b.len().min(CAP - self.len);
        self.buf[self.len..self.len + n].copy_from_slice(&b[..n]);
        self.len += n;
        self
    }
    #[inline]
    pub fn c(&mut self, ch: u8) -> &mut Self {
        if self.len + 1 > CAP {
            self.flush();
        }
        self.buf[self.len] = ch;
        self.len += 1;
        self
    }
    pub fn u(&mut self, mut v: u64) -> &mut Self {
        let mut tmp = [0u8; 20];
        let mut i = tmp.len();
        if v == 0 {
            return self.c(b'0');
        }
        while v > 0 {
            i -= 1;
            tmp[i] = b'0' + (v % 10) as u8;
            v /= 10;
        }
        self.s(&tmp[i..])
    }
    pub fn i(&mut self, v: i64) -> &mut Self {
        if v < 0 {
            self.c(b'-');
            self.u(v.unsigned_abs())
        } else {
            self.u(v as u64)
        }
    }
    pub fn nl(&mut self) -> &mut Self {
        self.c(b'\n');
        self.flush();
        self
    }
    pub fn flush(&mut self) {
        if self.len > 0 {
            sys::write(1, &self.buf[..self.len]);
            self.len = 0;
        }
    }
}

/// Long-algebraic move text: `e2e4`, `e7e8q`.
pub fn move_str(m: crate::pos::Move, out: &mut [u8; 6]) -> usize {
    let f = m.from();
    let t = m.to();
    out[0] = b'a' + (f & 7) as u8;
    out[1] = b'1' + (f >> 3) as u8;
    out[2] = b'a' + (t & 7) as u8;
    out[3] = b'1' + (t >> 3) as u8;
    if m.is_promo() {
        out[4] = b"nbrq"[(m.flag() & 3) as usize];
        5
    } else {
        4
    }
}
