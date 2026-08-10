//! Raw Darwin/arm64 kernel interface.
//!
//! Every kernel entry here is a hand-written `svc #0x80`. Nothing in this file
//! calls libc; libSystem is linked only because Mach-O requires it for the
//! process entry stub.
//!
//! arm64 Darwin convention: args in x0..x5, BSD syscall number in x16,
//! carry flag set on error with the errno in x0.

use core::arch::asm;

pub const SYS_EXIT: u64 = 1;
pub const SYS_READ: u64 = 3;
pub const SYS_WRITE: u64 = 4;
pub const SYS_MUNMAP: u64 = 73;
pub const SYS_MMAP: u64 = 197;
pub const SYS_POLL: u64 = 230;

const EINTR: i64 = 4;

/// Returns (value, is_error). `is_error` mirrors the carry flag.
#[inline(always)]
unsafe fn sc6(n: u64, a0: u64, a1: u64, a2: u64, a3: u64, a4: u64, a5: u64) -> (i64, bool) {
    let ret: i64;
    let err: u64;
    asm!(
        "svc #0x80",
        "cset {err}, cs",
        err = out(reg) err,
        inlateout("x0") a0 => ret,
        in("x1") a1,
        in("x2") a2,
        in("x3") a3,
        in("x4") a4,
        in("x5") a5,
        in("x16") n,
        options(nostack)
    );
    (ret, err != 0)
}

#[inline(always)]
unsafe fn sc3(n: u64, a0: u64, a1: u64, a2: u64) -> (i64, bool) {
    sc6(n, a0, a1, a2, 0, 0, 0)
}

/// Write the whole buffer to `fd`, retrying short writes and EINTR.
pub fn write(fd: i32, buf: &[u8]) {
    let mut off = 0usize;
    while off < buf.len() {
        let (n, err) =
            unsafe { sc3(SYS_WRITE, fd as u64, buf.as_ptr().add(off) as u64, (buf.len() - off) as u64) };
        if err {
            if n == EINTR {
                continue;
            }
            return;
        }
        if n <= 0 {
            return;
        }
        off += n as usize;
    }
}

/// Read once, retrying EINTR. `None` on EOF or error.
pub fn read(fd: i32, buf: &mut [u8]) -> Option<usize> {
    loop {
        let (n, err) = unsafe { sc3(SYS_READ, fd as u64, buf.as_mut_ptr() as u64, buf.len() as u64) };
        if err {
            if n == EINTR {
                continue;
            }
            return None;
        }
        return if n <= 0 { None } else { Some(n as usize) };
    }
}

pub fn exit(code: i32) -> ! {
    unsafe {
        sc3(SYS_EXIT, code as u64, 0, 0);
        core::hint::unreachable_unchecked()
    }
}

const PROT_READ_WRITE: u64 = 0x1 | 0x2;
const MAP_PRIVATE_ANON: u64 = 0x0002 | 0x1000;

/// Anonymous private mapping. Null on failure.
pub fn mmap(len: usize) -> *mut u8 {
    let (p, err) =
        unsafe { sc6(SYS_MMAP, 0, len as u64, PROT_READ_WRITE, MAP_PRIVATE_ANON, (-1i64) as u64, 0) };
    if err {
        core::ptr::null_mut()
    } else {
        p as *mut u8
    }
}

pub fn munmap(p: *mut u8, len: usize) {
    if !p.is_null() {
        unsafe {
            sc3(SYS_MUNMAP, p as u64, len as u64, 0);
        }
    }
}

/// Milliseconds from an arbitrary monotonic origin.
///
/// Reads the arm64 generic-timer registers directly: `cntvct_el0` ticking at
/// `cntfrq_el0` Hz (24 MHz on Apple silicon), both EL0-readable on Darwin.
/// No kernel trap — the search polls this in its time check — and immune to
/// wall-clock steps (NTP) mid-search. The `isb` orders the counter read
/// against surrounding instructions, as `mach_absolute_time` does.
pub fn now_ms() -> u64 {
    let cnt: u64;
    let frq: u64;
    unsafe {
        asm!(
            "isb",
            "mrs {cnt}, cntvct_el0",
            "mrs {frq}, cntfrq_el0",
            cnt = out(reg) cnt,
            frq = out(reg) frq,
            options(nomem, nostack)
        );
    }
    cnt / (frq / 1000)
}

/// Non-destructive check for pending bytes on `fd`. Used to notice `stop` /
/// `quit` mid-search without ever blocking the search thread.
pub fn readable(fd: i32) -> bool {
    #[repr(C)]
    struct PollFd {
        fd: i32,
        events: i16,
        revents: i16,
    }
    const POLLIN: i16 = 0x0001;
    let mut p = PollFd { fd, events: POLLIN, revents: 0 };
    loop {
        let (n, err) = unsafe { sc3(SYS_POLL, &mut p as *mut PollFd as u64, 1, 0) };
        if err && n == EINTR {
            continue;
        }
        return !err && n > 0 && (p.revents & POLLIN) != 0;
    }
}

/// `static mut` with the unsafety made explicit and the aliasing rules left to
/// the caller. The engine is single-threaded; every table below is written once
/// during init and read-only thereafter.
#[repr(transparent)]
pub struct SyncCell<T>(core::cell::UnsafeCell<T>);
unsafe impl<T> Sync for SyncCell<T> {}
impl<T> SyncCell<T> {
    pub const fn new(v: T) -> Self {
        Self(core::cell::UnsafeCell::new(v))
    }
    #[inline(always)]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn as_mut(&self) -> &mut T {
        &mut *self.0.get()
    }
    #[inline(always)]
    pub unsafe fn as_ref(&self) -> &T {
        &*self.0.get()
    }
}
