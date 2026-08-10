//! UCI protocol, line editing and the test entry points (`perft`, `bench`).
//!
//! stdin is read with the raw `read` syscall into one buffer. The same buffer
//! is polled mid-search so `stop` and `quit` are noticed without ever blocking
//! the search thread.

use crate::bb::*;
use crate::eval::*;
use crate::io::{move_str, Out};
use crate::movegen::*;
use crate::pos::*;
use crate::search::*;
use crate::sys::{self, SyncCell};
use crate::tt::tt;

const IN_CAP: usize = 1 << 16;

struct InBuf {
    buf: [u8; IN_CAP],
    len: usize,
    scanned: usize,
    eof: bool,
    stop: bool,
}
static IN: SyncCell<InBuf> = SyncCell::new(InBuf {
    buf: [0; IN_CAP],
    len: 0,
    scanned: 0,
    eof: false,
    stop: false,
});

fn inbuf() -> &'static mut InBuf {
    unsafe { IN.as_mut() }
}

impl InBuf {
    fn fill(&mut self, blocking: bool) {
        if self.eof || (!blocking && !sys::readable(0)) {
            return;
        }
        if self.len == IN_CAP {
            // Pathological input with no newline; drop it rather than wedge.
            self.len = 0;
            self.scanned = 0;
        }
        match sys::read(0, &mut self.buf[self.len..]) {
            Some(n) => self.len += n,
            None => self.eof = true,
        }
    }

    fn line_end(&self) -> Option<usize> {
        self.buf[..self.len].iter().position(|&c| c == b'\n')
    }

    fn consume(&mut self, upto: usize) {
        self.buf.copy_within(upto..self.len, 0);
        self.len -= upto;
        self.scanned = self.scanned.saturating_sub(upto);
    }
}

/// True once `stop` or `quit` has appeared on stdin. Called from the search's
/// node counter, so it must stay cheap: the `poll` syscall only runs when the
/// caller has already decided enough nodes have passed.
pub fn interrupted() -> bool {
    let i = inbuf();
    if i.stop {
        return true;
    }
    i.fill(false);
    // Only look at bytes that have arrived since the last scan.
    while i.scanned < i.len {
        let s = i.scanned;
        if i.buf[s..].starts_with(b"stop") || i.buf[s..].starts_with(b"quit") {
            i.stop = true;
            return true;
        }
        i.scanned += 1;
    }
    false
}

/// Blocking line read. `None` at EOF.
fn read_line(out: &mut Out) -> Option<[u8; 4096]> {
    out.flush();
    let i = inbuf();
    loop {
        if let Some(e) = i.line_end() {
            let mut line = [0u8; 4096];
            let n = e.min(4095);
            line[..n].copy_from_slice(&i.buf[..n]);
            i.consume(e + 1);
            i.scanned = 0;
            return Some(line);
        }
        if i.eof {
            return None;
        }
        i.fill(true);
    }
}

// ---------------------------------------------------------------------------

struct Tokens<'a> {
    s: &'a [u8],
    i: usize,
}
impl<'a> Tokens<'a> {
    fn new(s: &'a [u8]) -> Self {
        Tokens { s, i: 0 }
    }
    fn next(&mut self) -> Option<&'a [u8]> {
        while self.i < self.s.len() && (self.s[self.i] == b' ' || self.s[self.i] == b'\r') {
            self.i += 1;
        }
        if self.i >= self.s.len() || self.s[self.i] == 0 {
            return None;
        }
        let start = self.i;
        while self.i < self.s.len() && self.s[self.i] != b' ' && self.s[self.i] != b'\r' && self.s[self.i] != 0 {
            self.i += 1;
        }
        Some(&self.s[start..self.i])
    }
}

fn parse_u64(t: &[u8]) -> u64 {
    let mut v = 0u64;
    for &c in t {
        if c.is_ascii_digit() {
            v = v.wrapping_mul(10) + (c - b'0') as u64;
        }
    }
    v
}

static POS: SyncCell<Position> = SyncCell::new(Position::empty());
fn position() -> &'static mut Position {
    unsafe { POS.as_mut() }
}

/// Match a UCI move string against the legal moves; this is how promotion
/// suffixes and castling get resolved without a second parser.
fn find_move(pos: &Position, t: &[u8]) -> Move {
    let mut list = MoveList::new();
    generate(pos, &mut list, GenKind::All);
    for i in 0..list.n {
        let mut buf = [0u8; 6];
        let n = move_str(list.mv[i], &mut buf);
        if &buf[..n] == t {
            return list.mv[i];
        }
    }
    Move::NULL
}

pub fn perft(pos: &mut Position, depth: i32) -> u64 {
    let mut list = MoveList::new();
    generate(pos, &mut list, GenKind::All);
    if depth <= 1 {
        return list.n as u64;
    }
    let mut total = 0;
    for i in 0..list.n {
        let m = list.mv[i];
        pos.make(m);
        total += perft(pos, depth - 1);
        pos.unmake(m);
    }
    total
}

fn perft_divide(pos: &mut Position, depth: i32, out: &mut Out) {
    let t0 = sys::now_ms();
    let mut list = MoveList::new();
    generate(pos, &mut list, GenKind::All);
    let mut total = 0u64;
    for i in 0..list.n {
        let m = list.mv[i];
        pos.make(m);
        let n = if depth <= 1 { 1 } else { perft(pos, depth - 1) };
        pos.unmake(m);
        total += n;
        let mut buf = [0u8; 6];
        let k = move_str(m, &mut buf);
        out.s(&buf[..k]).s(b": ").u(n).nl();
    }
    let ms = sys::now_ms().saturating_sub(t0);
    out.s(b"nodes ").u(total).s(b" time ").u(ms).s(b" nps ").u(total * 1000 / ms.max(1)).nl();
}

const BENCH_FENS: [&[u8]; 12] = [
    b"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    b"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    b"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    b"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    b"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    b"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    b"2rq1rk1/pp1bppbp/2np1np1/8/3NP3/1BN1BP2/PPPQ2PP/2KR3R w - - 0 1",
    b"8/8/8/8/8/6k1/6p1/6K1 w - - 0 1",
    b"4rrk1/pp1n1pp1/2pb1q1p/3p4/3P1B2/2NBP2P/PPQ2PP1/3R1RK1 w - - 0 1",
    b"r1bq1rk1/pp2ppbp/2np1np1/8/2BNP3/2N1BP2/PPPQ2PP/R3K2R w KQ - 0 1",
    b"8/3k4/8/8/3K4/8/3P4/8 w - - 0 1",
    b"6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
];

fn bench(depth: i32, out: &mut Out) {
    let s = searcher();
    let mut total = 0u64;
    let t0 = sys::now_ms();
    for fen in BENCH_FENS {
        tt().clear();
        s.clear();
        let p = position();
        p.set_fen(fen);
        s.limits = Limits::new();
        s.limits.depth = depth;
        s.go(p, out);
        total += s.nodes;
    }
    let ms = sys::now_ms().saturating_sub(t0);
    out.s(b"===========================").nl();
    out.s(b"Total nodes  : ").u(total).nl();
    out.s(b"Time (ms)    : ").u(ms).nl();
    out.s(b"Nodes/second : ").u(total * 1000 / ms.max(1)).nl();
}

// ---------------------------------------------------------------------------

pub fn run() -> ! {
    let mut out = Out::new();
    let p = position();
    p.set_startpos();
    tt().resize(64);

    while let Some(line) = read_line(&mut out) {
        let mut t = Tokens::new(&line);
        let cmd = match t.next() {
            Some(c) => c,
            None => continue,
        };

        match cmd {
            b"uci" => {
                out.s(b"id name Sable 1.0").nl();
                out.s(b"id author built with Claude Code").nl();
                out.s(b"option name Hash type spin default 64 min 1 max 4096").nl();
                out.s(b"option name Threads type spin default 1 min 1 max 1").nl();
                out.s(b"uciok").nl();
            }
            b"isready" => {
                out.s(b"readyok").nl();
            }
            b"ucinewgame" => {
                tt().clear();
                searcher().clear();
                position().set_startpos();
            }
            b"setoption" => {
                // setoption name <id> value <x>
                let mut name: &[u8] = b"";
                let mut val: u64 = 0;
                while let Some(tok) = t.next() {
                    match tok {
                        b"name" => name = t.next().unwrap_or(b""),
                        b"value" => val = parse_u64(t.next().unwrap_or(b"0")),
                        _ => {}
                    }
                }
                if name == b"Hash" {
                    tt().resize(val.clamp(1, 4096) as usize);
                }
            }
            b"position" => {
                let p = position();
                match t.next() {
                    Some(b"startpos") => {
                        p.set_startpos();
                        let _ = t.next(); // consume "moves" if present
                    }
                    Some(b"fen") => {
                        // Collect the six FEN fields, then look for "moves".
                        let mut fen = [0u8; 128];
                        let mut n = 0;
                        for _ in 0..6 {
                            let save = t.i;
                            match t.next() {
                                Some(b"moves") | None => {
                                    t.i = save;
                                    break;
                                }
                                Some(f) => {
                                    if n > 0 && n < 128 {
                                        fen[n] = b' ';
                                        n += 1;
                                    }
                                    let k = f.len().min(128 - n);
                                    fen[n..n + k].copy_from_slice(&f[..k]);
                                    n += k;
                                }
                            }
                        }
                        p.set_fen(&fen[..n]);
                        let save = t.i;
                        if t.next() != Some(b"moves") {
                            t.i = save;
                        }
                    }
                    _ => {}
                }
                while let Some(mt) = t.next() {
                    let m = find_move(p, mt);
                    if m.is_null() {
                        break;
                    }
                    p.make(m);
                    // Keep the search stack shallow; game history lives in `hist`.
                    p.ply = 0;
                }
                p.root_ply = p.hist_len;
            }
            b"go" => {
                let s = searcher();
                s.limits = Limits::new();
                let mut do_perft = 0;
                while let Some(tok) = t.next() {
                    match tok {
                        b"depth" => s.limits.depth = parse_u64(t.next().unwrap_or(b"1")) as i32,
                        b"nodes" => s.limits.nodes = parse_u64(t.next().unwrap_or(b"0")),
                        b"movetime" => s.limits.movetime = parse_u64(t.next().unwrap_or(b"0")),
                        b"wtime" => s.limits.time[WHITE] = parse_u64(t.next().unwrap_or(b"0")),
                        b"btime" => s.limits.time[BLACK] = parse_u64(t.next().unwrap_or(b"0")),
                        b"winc" => s.limits.inc[WHITE] = parse_u64(t.next().unwrap_or(b"0")),
                        b"binc" => s.limits.inc[BLACK] = parse_u64(t.next().unwrap_or(b"0")),
                        b"movestogo" => s.limits.movestogo = parse_u64(t.next().unwrap_or(b"0")),
                        b"infinite" => s.limits.infinite = true,
                        b"perft" => do_perft = parse_u64(t.next().unwrap_or(b"1")) as i32,
                        _ => {}
                    }
                }
                inbuf().stop = false;
                if do_perft > 0 {
                    perft_divide(position(), do_perft, &mut out);
                } else {
                    s.go(position(), &mut out);
                }
            }
            b"perft" => {
                let d = parse_u64(t.next().unwrap_or(b"1")) as i32;
                perft_divide(position(), d, &mut out);
            }
            b"datagen" => {
                // datagen <positions> <nodes-per-move> <seed>
                let n = parse_u64(t.next().unwrap_or(b"100000"));
                let nodes = parse_u64(t.next().unwrap_or(b"5000")).max(100);
                let seed = parse_u64(t.next().unwrap_or(b"1"));
                tt().resize(8);
                crate::datagen::run(n, nodes, seed, &mut out);
            }
            b"featdump" => {
                // Every remaining line of stdin is a FEN. For each one, emit a
                // fixed-size binary record of the active feature indices for
                // both perspectives plus the output bucket.
                //
                // The trainer reads these rather than deriving features itself.
                // Two implementations of the same feature map is one of the few
                // bugs here that would produce a network which loads, runs, and
                // is silently wrong.
                use crate::net::{bucket_of, features, IN as NET_IN, MAX_F};
                const REC: usize = MAX_F * 4 + 2;
                while let Some(line) = read_line(&mut out) {
                    let end = line
                        .iter()
                        .position(|&c| c == 0 || c == b'\r')
                        .unwrap_or(line.len());
                    if end == 0 {
                        continue;
                    }
                    let p = position();
                    p.set_fen(&line[..end]);
                    let mut rec = [0u8; REC];
                    let mut f = [0u16; MAX_F];
                    for (side, half) in [(p.stm, 0usize), (p.stm ^ 1, 1usize)] {
                        let n = features(p, side, &mut f);
                        for (i, slot) in f.iter().enumerate() {
                            // Unused slots point at the zero row, so every
                            // record is the same width.
                            let v = if i < n { *slot } else { NET_IN as u16 };
                            let o = (half * MAX_F + i) * 2;
                            rec[o] = v as u8;
                            rec[o + 1] = (v >> 8) as u8;
                        }
                    }
                    let b = bucket_of(popcount(p.occ()) as usize) as u16;
                    rec[MAX_F * 4] = b as u8;
                    rec[MAX_F * 4 + 1] = (b >> 8) as u8;
                    out.s(&rec);
                }
                out.flush();
            }
            b"bench" => {
                let d = t.next().map(parse_u64).unwrap_or(12) as i32;
                bench(d.max(1), &mut out);
            }
            b"eval" => {
                out.s(b"eval ").i(evaluate(position()) as i64).nl();
            }
            b"evalhce" => {
                // The hand-crafted term alone. Training needs this to learn the
                // residual between it and what the search actually returns.
                out.s(b"eval ").i(hand_crafted(position()) as i64).nl();
            }
            b"d" => print_board(position(), &mut out),
            b"stop" => {}
            b"quit" => break,
            _ => {}
        }
    }
    out.flush();
    sys::exit(0)
}

fn print_board(pos: &Position, out: &mut Out) {
    const CH: &[u8; 12] = b"PNBRQKpnbrqk";
    for r in (0..8).rev() {
        out.s(b" +---+---+---+---+---+---+---+---+").nl();
        for f in 0..8 {
            let sq = r * 8 + f;
            let p = pos.piece_at(sq);
            out.s(b" | ");
            if p == NONE {
                out.c(b' ');
            } else {
                out.c(CH[pos.color_at(sq) * 6 + p as usize]);
            }
        }
        out.s(b" | ").u(r as u64 + 1).nl();
    }
    out.s(b" +---+---+---+---+---+---+---+---+").nl();
    out.s(b"   a   b   c   d   e   f   g   h").nl();
    out.s(b"key ").u(pos.key).nl();
}
