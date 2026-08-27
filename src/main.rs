//! Sable — a UCI chess engine.
//!
//! `no_std`, no allocator, no third-party crates. Kernel access is by raw
//! `svc` trap; libSystem is linked only to satisfy the Mach-O entry stub.
//!
//! The `no_std` / `no_main` attributes are conditional on not building tests.
//! A `no_main` binary has nowhere to put a test harness, so `cargo test` could
//! never run against this crate otherwise — the shipped binary is unaffected,
//! since a test build is a different binary entirely.

#![cfg_attr(not(test), no_std)]
#![cfg_attr(not(test), no_main)]
// A test build replaces `main` with the harness, so the whole UCI entry path is
// unreachable from the test target and every function it drives looks dead.
// The real binary builds warning-free.
#![cfg_attr(test, allow(dead_code))]

mod bb;
mod datagen;
mod eval;
mod io;
mod movegen;
mod net;
mod pos;
mod search;
mod sys;
mod tt;
mod uci;

#[cfg(test)]
mod tests;

#[cfg(not(test))]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    sys::write(2, b"panic\n");
    sys::exit(101)
}

// libcore on recent rustc still references the unwinding personality even with
// panic=abort. Provide a no-op so `cargo build` / `cargo run` (dev) link on
// Darwin; release already succeeded via LTO eliminating the call.
#[cfg(not(test))]
#[no_mangle]
extern "C" fn rust_eh_personality() {}

#[cfg(not(test))]
#[no_mangle]
pub extern "C" fn main(_argc: i32, _argv: *const *const u8) -> i32 {
    init();
    uci::run()
}

/// One-time table construction. Magic bitboards, Zobrist keys, the network and
/// the reduction table all have to exist before anything touches them.
pub fn init() {
    bb::init();
    pos::init_zobrist();
    net::init();
    // Every search thread has its own reduction table and its own copy of the
    // defaults the array initialiser had to leave zero.
    for i in 0..search::MAX_THREADS {
        let s = search::searcher_at(i);
        s.init_defaults(i);
        s.init_tables();
    }
}
