//! Sable — a UCI chess engine.
//!
//! `no_std`, no allocator, no third-party crates. Kernel access is by raw
//! `svc` trap; libSystem is linked only to satisfy the Mach-O entry stub.

#![no_std]
#![no_main]

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

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    sys::write(2, b"panic\n");
    sys::exit(101)
}

#[no_mangle]
pub extern "C" fn main(_argc: i32, _argv: *const *const u8) -> i32 {
    bb::init();
    pos::init_zobrist();
    net::init();
    search::searcher().init_tables();
    uci::run()
}
