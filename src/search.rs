//! Search.
//!
//! Fail-soft principal-variation search over an iteratively deepened tree.
//!
//! The shape of the thing, outermost first:
//!
//!   iterative deepening
//!     └ aspiration window, widened on each fail
//!         └ negamax (PVS)
//!             ├ transposition cutoff
//!             ├ static eval, corrected by the search's own past residuals
//!             ├ whole-node pruning: reverse futility, razoring, null move
//!             ├ move loop
//!             │   ├ per-move pruning: late-move, futility, SEE, history
//!             │   ├ extensions: check, singular
//!             │   └ late-move reduction + re-search ladder
//!             └ quiescence at the horizon
//!
//! Everything below the move loop is ordered so the cheapest test that can
//! reject a move runs first. Ordering quality is what makes the pruning safe,
//! so the history tables get as much attention as the pruning rules do.

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::bb::*;
use crate::eval::*;
use crate::io::{move_str, Out};
use crate::movegen::*;
use crate::pos::*;
use crate::sys::{self, SyncCell};
use crate::tt::*;

#[derive(Clone, Copy)]
pub struct Limits {
    pub depth: i32,
    pub nodes: u64,
    pub movetime: u64,
    pub time: [u64; 2],
    pub inc: [u64; 2],
    pub movestogo: u64,
    pub infinite: bool,
}
impl Limits {
    /// All-zero, which is not a usable set of limits -- it is the initialiser
    /// for the searcher array. A static only reaches BSS when every byte of its
    /// initialiser is zero, and `MAX_THREADS` searchers carrying `MAX_DEPTH`
    /// and `u64::MAX` would otherwise be half a megabyte of the binary image.
    /// Every entry point sets real limits before searching.
    pub const fn zeroed() -> Limits {
        Limits {
            depth: 0,
            nodes: 0,
            movetime: 0,
            time: [0, 0],
            inc: [0, 0],
            movestogo: 0,
            infinite: false,
        }
    }

    pub const fn new() -> Limits {
        Limits {
            depth: MAX_DEPTH,
            nodes: u64::MAX,
            movetime: 0,
            time: [0, 0],
            inc: [0, 0],
            movestogo: 0,
            infinite: false,
        }
    }
}

pub const MAX_DEPTH: i32 = 100;

#[derive(Clone, Copy)]
struct Node {
    mv: Move,
    /// `(colour*6 + piece_type)*64 + to`, biased by one so that zero means
    /// "no move". The bias matters: it keeps every static table all-zero, which
    /// puts 2.4 MB of continuation history in BSS instead of the binary image.
    piece_to: usize,
    eval: i32,
    excluded: Move,
    in_check: bool,
}
const NO_PIECE_TO: usize = 0;
const CONT_N: usize = 769;

pub struct Searcher {
    /// Index into every per-thread table below. Thread 0 is the one that owns
    /// the clock, the node budget, stdin and the output.
    pub id: usize,
    /// How many threads this search is running with, so the single-threaded
    /// path can keep its exact node accounting.
    pub threads: usize,
    /// First iteration this thread runs. Helpers start a little deeper than
    /// each other so that they diverge immediately instead of all re-deriving
    /// the same first few plies before the shared table pulls them apart.
    pub start_depth: i32,
    pub nodes: u64,
    pub seldepth: usize,
    pub stop: bool,
    pub limits: Limits,
    start: u64,
    soft: u64,
    hard: u64,
    /// Set once a move exists that we are willing to return.
    pub best: Move,
    pub best_score: i32,
    /// Suppresses `info`/`bestmove` output; data generation runs millions of
    /// searches and the protocol chatter would dominate the run time.
    pub silent: bool,
    /// Set for searches that are not driven by the GUI: `bench` runs a fixed
    /// workload, so a `quit` sitting further down the pipe is not an order to
    /// abandon it. Without this, `printf 'bench\nquit\n' | sable` reports zero
    /// nodes, which reads as a benchmark result rather than a cancelled run.
    pub ignore_stdin: bool,
    /// Milliseconds held back from every time budget to cover the trip through
    /// the GUI and the pipe. Set by the `Move Overhead` UCI option: the default
    /// suits a local match runner, a network game wants more.
    pub move_overhead: u64,

    killers: [[Move; 2]; MAX_PLY + 4],
    /// `[side][from][to]`, the classic butterfly table.
    history: [[[i16; 64]; 64]; 2],
    /// `[piece][to][captured]`
    capt_hist: [[[i16; 7]; 64]; 12],
    counter: [[Move; 64]; 12],
    stack: [Node; MAX_PLY + 8],
    pv: [[Move; MAX_PLY]; MAX_PLY],
    pv_len: [usize; MAX_PLY],
    /// Reduction table in 1/1024 units.
    lmr: [[i32; 64]; 64],
}

/// The most search threads that can be asked for. Every per-thread table below
/// is sized by this, and they are all zero-initialised, so the cost of the
/// ceiling is address space rather than binary.
pub const MAX_THREADS: usize = 8;

const NEW_SEARCHER: Searcher = Searcher {
    id: 0,
    threads: 0,
    start_depth: 0,
    nodes: 0,
    seldepth: 0,
    stop: false,
    limits: Limits::new(),
    start: 0,
    soft: 0,
    hard: 0,
    best: Move::NULL,
    best_score: 0,
    silent: false,
    ignore_stdin: false,
    move_overhead: 0,
    killers: [[Move::NULL; 2]; MAX_PLY + 4],
    history: [[[0; 64]; 64]; 2],
    capt_hist: [[[0; 7]; 64]; 12],
    counter: [[Move::NULL; 64]; 12],
    stack: [Node { mv: Move::NULL, piece_to: NO_PIECE_TO, eval: 0, excluded: Move::NULL, in_check: false };
        MAX_PLY + 8],
    pv: [[Move::NULL; MAX_PLY]; MAX_PLY],
    pv_len: [0; MAX_PLY],
    lmr: [[0; 64]; 64],
};

static SEARCHERS: SyncCell<[Searcher; MAX_THREADS]> = SyncCell::new([NEW_SEARCHER; MAX_THREADS]);

/// Set once the search should wind up, by whichever thread decides it. Helpers
/// watch it; the main thread is the only one that sets it from a real limit.
static STOP: AtomicBool = AtomicBool::new(false);

/// Each thread's node count, republished every thousand nodes so the main
/// thread can hold the whole search to one budget without reading another
/// thread's fields directly.
#[allow(clippy::declare_interior_mutable_const)]
const ZERO_U64: AtomicU64 = AtomicU64::new(0);
static NODE_PUB: [AtomicU64; MAX_THREADS] = [ZERO_U64; MAX_THREADS];

/// Total nodes across every thread taking part in the current search.
pub fn total_nodes(threads: usize) -> u64 {
    let mut t = 0;
    for n in NODE_PUB.iter().take(threads.clamp(1, MAX_THREADS)) {
        t += n.load(Ordering::Relaxed);
    }
    t
}

/// Continuation history: indexed by the previous move's `piece*64+to`, then
/// this move's. Two plies back, which is where most of the signal is.
///
/// Deliberately a static of its own rather than a field of `Searcher`: a global
/// only lands in BSS when its *entire* initialiser is zero, and `Searcher` has
/// non-zero fields. Keeping these 2.4 MB separate is the difference between a
/// 200 KB binary and a 2.6 MB one.
static CONT: SyncCell<[[[[i16; CONT_N]; CONT_N]; 2]; MAX_THREADS]> =
    SyncCell::new([[[[0; CONT_N]; CONT_N]; 2]; MAX_THREADS]);

/// One move list per ply, rather than one per node on the stack.
///
/// A `MoveList` is 1544 bytes and `negamax` builds one every node, which is
/// what gives the frame its size: at 128 plies that is most of a megabyte of
/// stack, and every node dirties fresh cache lines that the previous node had
/// just warmed. Indexed by ply, so a node and its children never share a slot,
/// and the recursion writes to the same 1544 bytes each time it returns to a
/// depth. All-zero initialiser, so this sits in BSS and adds nothing to the
/// binary.
/// Two banks, because a singular search re-enters `negamax` at the *same* ply
/// with one move excluded. Sharing a slot there lets the child's `generate`
/// overwrite the list its parent is still looping over -- which does not crash,
/// it just quietly searches a different tree. A singular search cannot nest at
/// one ply (it only triggers when nothing is excluded yet), so the second bank
/// is enough.
const LIST_SLOTS: usize = MAX_PLY + 8;
static LISTS: SyncCell<[[MoveList; LIST_SLOTS * 2]; MAX_THREADS]> =
    SyncCell::new([[MoveList::new(); LIST_SLOTS * 2]; MAX_THREADS]);

/// Same treatment for the two lists of moves already tried at a node: 520 bytes
/// each, per node, for the same reason and with the same fix.
static TRIED: SyncCell<[[[Tried; 2]; LIST_SLOTS * 2]; MAX_THREADS]> =
    SyncCell::new([[[Tried::new(); 2]; LIST_SLOTS * 2]; MAX_THREADS]);

#[inline(always)]
fn tried_at(id: usize, ply: usize, excluded: bool) -> &'static mut [Tried; 2] {
    let slot = ply.min(LIST_SLOTS - 1) + if excluded { LIST_SLOTS } else { 0 };
    unsafe { &mut TRIED.as_mut()[id][slot] }
}

#[inline(always)]
fn list_at(id: usize, ply: usize, excluded: bool) -> &'static mut MoveList {
    let slot = ply.min(LIST_SLOTS - 1) + if excluded { LIST_SLOTS } else { 0 };
    unsafe { &mut LISTS.as_mut()[id][slot] }
}

#[inline(always)]
fn cont(id: usize) -> &'static mut [[[i16; CONT_N]; CONT_N]; 2] {
    unsafe { &mut CONT.as_mut()[id] }
}

/// Static-eval correction history.
///
/// The search is its own teacher here: whenever a node's true score disagrees
/// with what the static evaluation claimed, the residual is remembered and
/// applied the next time a position indexes the same entry.
///
/// Three tables, three views of the same residual -- pawn structure, the
/// non-pawn material layout, and the move that led here. They disagree often
/// enough to be worth keeping apart: a structure that flatters the evaluation
/// does not flatter it in every piece configuration.
///
/// Units are 1/`CORR_GRAIN` centipawns. Statics of their own rather than
/// `Searcher` fields for the same reason as `CONT`: an all-zero initialiser
/// stays in BSS.
const CORR_N: usize = 16_384;
const CORR_GRAIN: i32 = 256;
/// Ceiling on one table's entry, in centipawns times the grain. Each table is a
/// nudge, never a replacement for the evaluation.
const CORR_MAX: i32 = CORR_GRAIN * 48;
/// Ceiling on the three of them combined, in centipawns.
const CORR_CLAMP: i32 = 72;
static CORR_PAWN: SyncCell<[[[i32; CORR_N]; 2]; MAX_THREADS]> =
    SyncCell::new([[[0; CORR_N]; 2]; MAX_THREADS]);
static CORR_NP: SyncCell<[[[i32; CORR_N]; 2]; MAX_THREADS]> =
    SyncCell::new([[[0; CORR_N]; 2]; MAX_THREADS]);
static CORR_CONT: SyncCell<[[[i32; CONT_N]; 2]; MAX_THREADS]> =
    SyncCell::new([[[0; CONT_N]; 2]; MAX_THREADS]);

#[inline(always)]
fn corr_pawn(id: usize) -> &'static mut [[i32; CORR_N]; 2] {
    unsafe { &mut CORR_PAWN.as_mut()[id] }
}
#[inline(always)]
fn corr_np(id: usize) -> &'static mut [[i32; CORR_N]; 2] {
    unsafe { &mut CORR_NP.as_mut()[id] }
}
#[inline(always)]
fn corr_cont(id: usize) -> &'static mut [[i32; CONT_N]; 2] {
    unsafe { &mut CORR_CONT.as_mut()[id] }
}

#[inline(always)]
pub fn searcher() -> &'static mut Searcher {
    searcher_at(0)
}

#[inline(always)]
pub fn searcher_at(i: usize) -> &'static mut Searcher {
    unsafe { &mut SEARCHERS.as_mut()[i.min(MAX_THREADS - 1)] }
}

/// Scale a millisecond budget by a per-mille factor. `u64::MAX` means "no
/// limit" and has to survive scaling as itself; a saturating multiply would
/// turn it into a large but finite budget and quietly cap an infinite search.
#[inline(always)]
fn scale_ms(ms: u64, per_mille: u64) -> u64 {
    if ms == u64::MAX {
        return ms;
    }
    // Overflow means the budget was already beyond anything a search will
    // reach, so clamping it there loses nothing. (128-bit division is avoided
    // on purpose: it would pull the unwinding personality into a binary that
    // has no unwinder.)
    match ms.checked_mul(per_mille) {
        Some(v) => v / 1000,
        None => u64::MAX,
    }
}

/// `ln(x) * 1024`, integer only. Accurate to a few percent, which is all the
/// reduction formula needs.
fn ln1024(x: u32) -> i32 {
    if x == 0 {
        return 0;
    }
    let lz = x.leading_zeros();
    let int_part = (31 - lz) as i32;
    // Mantissa bits below the leading one, as a fraction of 1024.
    let mant = if int_part > 0 {
        ((x as u64) << (lz + 1) >> 54) as i32 // top 10 bits after the leading 1
    } else {
        0
    };
    // log2(x) ~ int + mant/1024, refined with a quadratic correction so the
    // error at the midpoint (worst case for the linear form) is small.
    let frac = mant - (mant * mant) / 3072;
    let log2 = int_part * 1024 + frac;
    (log2 as i64 * 709 / 1024) as i32
}

/// Scale on every margin that is compared against a static evaluation.
///
/// These margins are in centipawns and they were tuned, over months, against an
/// evaluation that turned out to be understating positions by about a third.
/// The network now ships deliberately quiet to match them, which works but fixes
/// the wrong half of the system: a quieter evaluation is also a less accurate
/// one everywhere else it is used, including the correction tables and the
/// transposition table.
///
/// This is the other half. `MARGIN` scales the thresholds instead, so a network
/// exported at its natural scale can be handed to a search whose margins have
/// been widened to match. 100 leaves the tuned values exactly as they were.
const MARGIN: i32 = 100;

#[inline(always)]
const fn margin(cp: i32) -> i32 {
    cp * MARGIN / 100
}

impl Searcher {
    pub fn init_tables(&mut self) {
        for d in 1..64 {
            for m in 1..64 {
                self.lmr[d][m] = 790 + ln1024(d as u32) * ln1024(m as u32) / 2417;
            }
        }
    }

    /// Fields the array initialiser had to leave zero so that the whole of it
    /// would sit in BSS rather than in the binary.
    pub fn init_defaults(&mut self, id: usize) {
        self.id = id;
        self.threads = 1;
        self.start_depth = 0;
        self.move_overhead = 25;
        self.limits = Limits::new();
    }

    pub fn clear(&mut self) {
        self.killers = [[Move::NULL; 2]; MAX_PLY + 4];
        self.history = [[[0; 64]; 64]; 2];
        self.capt_hist = [[[0; 7]; 64]; 12];
        self.counter = [[Move::NULL; 64]; 12];
        for t in cont(self.id).iter_mut() {
            for r in t.iter_mut() {
                r.fill(0);
            }
        }
        for t in corr_pawn(self.id).iter_mut() {
            t.fill(0);
        }
        for t in corr_np(self.id).iter_mut() {
            t.fill(0);
        }
        for t in corr_cont(self.id).iter_mut() {
            t.fill(0);
        }
        if self.id == 0 {
            crate::net::clear_cache();
        }
    }

    /// The static evaluation after the correction tables have had their say.
    /// Pawn structure carries the most signal, so it carries the most weight.
    #[inline(always)]
    fn corrected_eval(&self, pos: &Position, ply: usize, raw: i32) -> i32 {
        let stm = pos.stm;
        let mut c = corr_pawn(self.id)[stm][pos.pawn_key as usize & (CORR_N - 1)] * 3 / 2;
        c += corr_np(self.id)[stm][pos.np_key as usize & (CORR_N - 1)];
        if ply > 0 {
            let prev = self.stack[ply - 1].piece_to;
            if prev != NO_PIECE_TO {
                c += corr_cont(self.id)[stm][prev];
            }
        }
        let c = (c / (CORR_GRAIN * 2)).clamp(-CORR_CLAMP, CORR_CLAMP);
        (raw + c).clamp(-MATE_IN_MAX + 1, MATE_IN_MAX - 1)
    }

    /// Blend this node's residual into every table that indexed it. Deeper
    /// searches were more certain about their score, so they move entries
    /// further; the gravity term keeps a saturated entry responsive.
    #[inline(always)]
    fn update_corr(&mut self, pos: &Position, ply: usize, diff: i32, depth: i32) {
        let weight = (depth + 1).min(16);
        let scaled = diff.clamp(-256, 256) * CORR_GRAIN;
        let blend = |e: &mut i32| {
            *e = ((*e * (256 - weight) + scaled * weight) / 256).clamp(-CORR_MAX, CORR_MAX);
        };
        let stm = pos.stm;
        blend(&mut corr_pawn(self.id)[stm][pos.pawn_key as usize & (CORR_N - 1)]);
        blend(&mut corr_np(self.id)[stm][pos.np_key as usize & (CORR_N - 1)]);
        if ply > 0 {
            let prev = self.stack[ply - 1].piece_to;
            if prev != NO_PIECE_TO {
                blend(&mut corr_cont(self.id)[stm][prev]);
            }
        }
    }

    #[inline(always)]
    fn elapsed(&self) -> u64 {
        sys::now_ms().saturating_sub(self.start)
    }

    fn set_clocks(&mut self, stm: usize) {
        self.start = sys::now_ms();
        if self.limits.movetime > 0 {
            let t = self.limits.movetime.saturating_sub(self.move_overhead).max(1);
            self.soft = t;
            self.hard = t;
            return;
        }
        if self.limits.time[stm] == 0 {
            self.soft = u64::MAX;
            self.hard = u64::MAX;
            return;
        }
        let avail = self.limits.time[stm].saturating_sub(self.move_overhead);
        let inc = self.limits.inc[stm];
        let mtg = if self.limits.movestogo > 0 {
            self.limits.movestogo.min(50)
        } else {
            // No move count given: assume the game still has this many moves.
            40
        };
        let optimum = avail / mtg + inc * 3 / 4;
        // Never spend more than a fraction of what is left; losing on time is
        // worth less than any evaluation gain.
        self.soft = optimum.min(avail / 2).max(1);
        self.hard = (optimum * 4).min(avail * 3 / 4).max(1);
    }

    /// Only thread 0 owns the clock, the node budget and stdin. Helper threads
    /// run until it tells them to stop, and the flag is the only thing they
    /// look at -- a helper that decided for itself when the search was over
    /// would leave the main thread waiting on a join it could not hurry.
    #[inline(always)]
    fn check_stop(&mut self) {
        if self.stop {
            return;
        }
        // A single-threaded search keeps its exact node accounting: `bench` and
        // `datagen` reproduce node-for-node, and every measurement in this repo
        // was taken against a node limit that lands on the node.
        if self.threads <= 1 && self.nodes >= self.limits.nodes {
            self.stop = true;
            return;
        }
        if self.nodes & 1023 != 0 {
            return;
        }
        NODE_PUB[self.id].store(self.nodes, Ordering::Relaxed);
        if STOP.load(Ordering::Relaxed) {
            self.stop = true;
            return;
        }
        if self.id != 0 {
            return;
        }
        if self.threads > 1 && total_nodes(self.threads) >= self.limits.nodes {
            self.stop = true;
        }
        if !self.limits.infinite && self.elapsed() >= self.hard {
            self.stop = true;
        }
        // Only a UCI search treats stdin as a controller. `datagen` and
        // `relabel` are fed their work on stdin, and polling it there both
        // costs the scan and risks eating input that is not a command.
        if !self.silent && !self.ignore_stdin && crate::uci::interrupted() {
            self.stop = true;
        }
        if self.stop {
            STOP.store(true, Ordering::Relaxed);
        }
    }

    // -----------------------------------------------------------------------
    // Iterative deepening
    // -----------------------------------------------------------------------

    pub fn go(&mut self, pos: &mut Position, out: &mut Out) {
        self.nodes = 0;
        self.stop = false;
        self.best = Move::NULL;
        self.best_score = 0;
        self.set_clocks(pos.stm);
        if self.id == 0 {
            tt().new_search();
        }
        for n in self.stack.iter_mut() {
            n.mv = Move::NULL;
            n.excluded = Move::NULL;
            n.piece_to = NO_PIECE_TO;
        }
        pos.ply = 0;

        // A legal move must exist before anything else; a mated root has none
        // and every later stage assumes one.
        let mut root_moves = MoveList::new();
        generate(pos, &mut root_moves, GenKind::All);
        if root_moves.n == 0 {
            if !self.silent {
                out.s(b"bestmove 0000").nl();
            }
            self.best = Move::NULL;
            return;
        }
        self.best = root_moves.mv[0];

        let mut score = 0i32;
        let max_depth = self.limits.depth.min(MAX_DEPTH);
        // Time is spent where the answer is still in doubt. A root move that has
        // survived several iterations rarely changes on the next one, so the
        // soft budget shrinks as it settles and stretches while it flaps.
        const STABILITY: [u64; 7] = [1500, 1300, 1150, 1050, 1000, 950, 900];
        let mut stability = 0usize;
        let mut prev_best = Move::NULL;
        let mut prev_score = 0i32;

        let first = if self.start_depth > 0 { self.start_depth.min(max_depth) } else { 1 };
        for depth in first..=max_depth {
            let mut delta = 10 + (score * score) / 12_000;
            let (mut alpha, mut beta) =
                if depth < 4 { (-INF, INF) } else { ((score - delta).max(-INF), (score + delta).min(INF)) };
            let mut cur = depth;

            loop {
                let v = self.negamax(pos, cur, alpha, beta, 0, false, true);
                if self.stop {
                    break;
                }
                if v <= alpha {
                    // Fail low: relax alpha and reset depth, the position is
                    // worse than we assumed and needs the full effort again.
                    beta = (alpha + beta) / 2;
                    alpha = (v - delta).max(-INF);
                    cur = depth;
                } else if v >= beta {
                    beta = (v + delta).min(INF);
                    // A fail high usually means the move is simply good; a
                    // slightly shallower verification is enough.
                    if cur > 1 && v.abs() < MATE_IN_MAX {
                        cur -= 1;
                    }
                } else {
                    score = v;
                    break;
                }
                delta += delta / 3;
            }

            if self.stop && self.best.is_null() {
                break;
            }
            if !self.stop {
                self.best_score = score;
                if self.pv_len[0] > 0 {
                    self.best = self.pv[0][0];
                }
                if !self.silent {
                    self.report(depth, score, out);
                }
            }
            if self.stop {
                break;
            }
            if self.best == prev_best {
                stability = (stability + 1).min(STABILITY.len() - 1);
            } else {
                stability = 0;
            }
            prev_best = self.best;

            // A score that just fell is a score that is still moving; buy time
            // to find out where it lands, up to half as much again.
            let fall =
                if depth == 1 { 1000 } else { 1000 + (prev_score - score).clamp(0, 200) as u64 * 5 / 2 };
            prev_score = score;

            // Starting depth d+1 costs roughly 1.5x what depth d did; if that
            // would not fit in the soft budget, stop while we are ahead. Only
            // the main thread may act on this: a helper that stopped early
            // would stop helping while the search was still running.
            if self.id == 0 {
                let soft = scale_ms(scale_ms(self.soft, STABILITY[stability]), fall);
                if !self.limits.infinite && self.limits.movetime == 0 && self.elapsed() * 3 / 2 >= soft {
                    break;
                }
                if self.limits.movetime > 0 && self.elapsed() >= self.soft {
                    break;
                }
                if score.abs() >= MATE_IN_MAX && depth >= 4 {
                    break;
                }
            }
        }

        if !self.silent {
            out.s(b"bestmove ");
            let mut buf = [0u8; 6];
            let n = move_str(self.best, &mut buf);
            out.s(&buf[..n]).nl();
        }
    }

    fn report(&mut self, depth: i32, score: i32, out: &mut Out) {
        let ms = self.elapsed();
        out.s(b"info depth ").u(depth as u64);
        out.s(b" seldepth ").u(self.seldepth as u64);
        if score.abs() >= MATE_IN_MAX {
            let plies = MATE - score.abs();
            let moves = (plies + 1) / 2;
            out.s(b" score mate ").i(if score > 0 { moves as i64 } else { -(moves as i64) });
        } else {
            out.s(b" score cp ").i(score as i64);
        }
        let nodes = if self.threads > 1 { total_nodes(self.threads).max(self.nodes) } else { self.nodes };
        out.s(b" nodes ").u(nodes);
        out.s(b" nps ").u(nodes * 1000 / ms.max(1));
        out.s(b" hashfull ").u(tt().hashfull() as u64);
        out.s(b" time ").u(ms);
        out.s(b" pv");
        for i in 0..self.pv_len[0] {
            let mut buf = [0u8; 6];
            let n = move_str(self.pv[0][i], &mut buf);
            out.c(b' ').s(&buf[..n]);
        }
        out.nl();
    }

    #[inline(always)]
    fn update_pv(&mut self, ply: usize, m: Move) {
        self.pv[ply][0] = m;
        let child = self.pv_len[ply + 1];
        for i in 0..child {
            self.pv[ply][i + 1] = self.pv[ply + 1][i];
        }
        self.pv_len[ply] = child + 1;
    }

    // -----------------------------------------------------------------------
    // Negamax
    // -----------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn negamax(
        &mut self,
        pos: &mut Position,
        mut depth: i32,
        mut alpha: i32,
        mut beta: i32,
        ply: usize,
        cut_node: bool,
        pv_node: bool,
    ) -> i32 {
        if depth <= 0 {
            return self.qsearch(pos, alpha, beta, ply, pv_node);
        }
        self.check_stop();
        if self.stop {
            return 0;
        }

        let root = ply == 0;
        self.pv_len[ply] = 0;
        self.nodes += 1;
        if ply > self.seldepth {
            self.seldepth = ply;
        }

        let in_check = pos.in_check();

        if !root {
            if pos.is_draw(ply) || pos.is_material_draw() {
                return 0;
            }
            if ply >= MAX_PLY - 1 {
                return if in_check { 0 } else { evaluate(pos) };
            }
            // Mate-distance pruning: a faster mate already found elsewhere
            // makes this whole subtree irrelevant.
            alpha = alpha.max(-MATE + ply as i32);
            beta = beta.min(MATE - ply as i32 - 1);
            if alpha >= beta {
                return alpha;
            }
        }

        // Killers two plies down belong to whatever the last visit to this
        // subtree was doing; by the time the search gets back here they refute
        // a position that is no longer on the board.
        self.killers[ply + 2] = [Move::NULL; 2];

        let excluded = self.stack[ply].excluded;
        let hit = if excluded.is_null() { tt().probe(pos.key, ply) } else { None };
        let mut tt_move = Move::NULL;
        let mut tt_score = 0;
        let mut tt_depth = -1;
        let mut tt_bound = BOUND_NONE;
        let mut tt_eval = i32::MIN;
        if let Some(h) = &hit {
            tt_move = h.mv;
            tt_score = h.score;
            tt_depth = h.depth;
            tt_bound = h.bound;
            tt_eval = h.eval;
            if !pv_node
                && tt_depth >= depth
                && (tt_bound == BOUND_EXACT
                    || (tt_bound == BOUND_LOWER && tt_score >= beta)
                    || (tt_bound == BOUND_UPPER && tt_score <= alpha))
            {
                // A cutoff from the table still deserves a history nudge, or
                // ordering degrades in positions the table already knows.
                if tt_score >= beta && !tt_move.is_null() && !pos_is_noisy(pos, tt_move) {
                    self.bonus_quiet(pos, tt_move, ply, depth * 100);
                }
                return tt_score;
            }
        }
        // No legality check on the table move: it is only ever *compared*
        // against generated moves, never made. A colliding entry can therefore
        // only cost a little ordering quality, never correctness -- and the
        // check would mean a second full move generation at every node.

        // --- static evaluation
        let raw_eval = if in_check {
            -INF
        } else if tt_eval != i32::MIN && tt_eval.abs() < MATE_IN_MAX {
            tt_eval
        } else {
            evaluate(pos)
        };
        // What the correction table makes of it. `raw_eval` is what goes into
        // the transposition table -- the correction is re-derived on every
        // probe, so storing it would freeze a stale version of it.
        let corrected = if in_check { -INF } else { self.corrected_eval(pos, ply, raw_eval) };
        // The table's score is a better estimate than the static eval when it
        // is on the right side of it.
        let eval = if !in_check
            && tt_bound != BOUND_NONE
            && tt_score.abs() < MATE_IN_MAX
            && ((tt_bound == BOUND_LOWER && tt_score > corrected)
                || (tt_bound == BOUND_UPPER && tt_score < corrected)
                || tt_bound == BOUND_EXACT)
        {
            tt_score
        } else {
            corrected
        };
        self.stack[ply].eval = corrected;
        self.stack[ply].in_check = in_check;

        // "Improving": is our position better than it was two plies ago? If
        // not, we are on the back foot and should prune less.
        let improving = if in_check {
            false
        } else if ply >= 2 && self.stack[ply - 2].eval != -INF {
            corrected > self.stack[ply - 2].eval
        } else if ply >= 4 && self.stack[ply - 4].eval != -INF {
            corrected > self.stack[ply - 4].eval
        } else {
            true
        };

        // --- whole-node pruning
        if !pv_node && !in_check && excluded.is_null() && beta.abs() < MATE_IN_MAX {
            // Reverse futility: so far ahead that even giving up material
            // repeatedly would not drop us below beta.
            if depth < 9 && eval - margin(75) * depth + margin(60) * improving as i32 >= beta {
                return beta + (eval - beta) / 3;
            }

            // Razoring: so far behind that only a tactic saves us; let the
            // quiescence search look for one.
            if depth <= 5 && eval + margin(180) * depth < alpha {
                let v = self.qsearch(pos, alpha - 1, alpha, ply, false);
                if v < alpha {
                    return v;
                }
            }

            // Null move: give the opponent a free move; if we are still above
            // beta, the real move will be at least as good. Disabled without
            // pieces, where zugzwang makes the assumption false.
            if depth >= 3
                && eval >= beta
                && !self.stack[ply.saturating_sub(1)].mv.is_null()
                && pos.has_big_pieces(pos.stm)
            {
                let r = 4 + depth / 3 + ((eval - beta) / margin(200)).min(3);
                pos.make_null();
                self.stack[ply].mv = Move::NULL;
                self.stack[ply].piece_to = NO_PIECE_TO;
                let v = -self.negamax(pos, depth - r, -beta, -beta + 1, ply + 1, !cut_node, false);
                pos.unmake_null();
                if self.stop {
                    return 0;
                }
                if v >= beta {
                    // Never return an unproven mate from a null move.
                    return if v >= MATE_IN_MAX { beta } else { v };
                }
            }

            // ProbCut: if a capture beats a beta raised by a full pawn and a
            // bit, at a much reduced depth, the full-depth search would almost
            // certainly have beaten the real beta too. Each candidate is sifted
            // by quiescence first, so the reduced search only runs for captures
            // that already look like they clear the raised bar.
            let pc_beta = beta + margin(180);
            if depth >= 5
                && pc_beta.abs() < MATE_IN_MAX
                // A shallower table entry that already failed this test is
                // enough to skip it; nothing here would overturn it.
                && !(tt_depth >= depth - 3 && tt_score < pc_beta)
            {
                let list = list_at(self.id, ply, !excluded.is_null());
                generate(pos, list, GenKind::Noisy);
                self.score_moves::<-20>(pos, list, tt_move, ply);
                for i in 0..list.n {
                    let m = list.pick(i);
                    // The capture has to be able to reach the raised beta on
                    // material alone before it is worth a node.
                    if m == excluded || !see_ge(pos, m, pc_beta - corrected) {
                        continue;
                    }
                    let to = m.to();
                    let moved = pc_index(pos.stm, pos.piece_at(m.from()) as usize);
                    pos.make(m);
                    self.stack[ply].mv = m;
                    self.stack[ply].piece_to = moved * 64 + to + 1;
                    let mut v = -self.qsearch(pos, -pc_beta, -pc_beta + 1, ply + 1, false);
                    if v >= pc_beta {
                        v = -self.negamax(pos, depth - 4, -pc_beta, -pc_beta + 1, ply + 1, !cut_node, false);
                    }
                    pos.unmake(m);
                    if self.stop {
                        return 0;
                    }
                    if v >= pc_beta {
                        tt().store(pos.key, m, v, raw_eval, depth - 3, BOUND_LOWER, ply);
                        return v;
                    }
                }
            }
        }

        // Internal iterative reduction: with no table move, this node is not
        // worth its nominal depth — the ordering will be poor either way.
        if tt_move.is_null() && depth >= 4 && !in_check {
            depth -= 1 + pv_node as i32;
        }

        // --- move loop
        let list = list_at(self.id, ply, !excluded.is_null());
        generate(pos, list, GenKind::All);
        if list.n == 0 {
            return if in_check { -MATE + ply as i32 } else { 0 };
        }
        self.score_moves::<-20>(pos, list, tt_move, ply);

        let mut best = -INF;
        let mut best_move = Move::NULL;
        let mut bound = BOUND_UPPER;
        let tried = tried_at(self.id, ply, !excluded.is_null());
        tried[0].clear();
        tried[1].clear();
        let (searched_quiets, searched_noisy) = tried.split_at_mut(1);
        let searched_quiets = &mut searched_quiets[0];
        let searched_noisy = &mut searched_noisy[0];
        let mut moves_played = 0i32;
        let mut skip_quiets = false;
        let lmp_limit = (3 + depth * depth) / (2 - improving as i32);

        for i in 0..list.n {
            let m = list.pick(i);
            if m == excluded {
                continue;
            }
            let noisy = pos_is_noisy(pos, m);
            let from = m.from();
            let to = m.to();
            let moved = pc_index(pos.stm, pos.piece_at(from) as usize);

            if skip_quiets && !noisy && !m.is_promo() {
                continue;
            }

            // --- move-level pruning, only once a fallback score exists
            if !root && best > -MATE_IN_MAX && !in_check {
                if !noisy {
                    // Late move pruning: deep in the ordered list at low depth,
                    // quiet moves almost never turn out to be best.
                    if moves_played >= lmp_limit {
                        skip_quiets = true;
                        continue;
                    }
                    // Futility: a quiet move cannot swing the score this far.
                    let lmr_depth = (depth
                        - (self.lmr[depth.min(63) as usize][(moves_played as usize).min(63)] >> 10))
                        .max(0);
                    if lmr_depth < 8 && eval + margin(120) + margin(110) * lmr_depth <= alpha {
                        skip_quiets = true;
                        continue;
                    }
                    // Quiet moves that hang material are rarely the answer.
                    if lmr_depth < 7 && !see_ge(pos, m, -25 * lmr_depth * lmr_depth) {
                        continue;
                    }
                    // History pruning: a move the tables have watched fail this
                    // often, this close to the horizon, is not worth a node.
                    if lmr_depth <= 3
                        && (self.history_of(pos, m, ply, moved, to, false) as i32) < -3200 * lmr_depth.max(1)
                    {
                        continue;
                    }
                } else if depth < 8 && !see_ge(pos, m, -95 * depth) {
                    continue;
                }
            }

            // --- extensions
            let mut extension = 0i32;
            if !root && ply < 2 * (self.limits.depth as usize).min(MAX_PLY / 2) {
                if m == tt_move
                    && depth >= 8
                    && excluded.is_null()
                    && tt_depth >= depth - 3
                    && tt_bound & BOUND_LOWER != 0
                    && tt_score.abs() < MATE_IN_MAX
                {
                    // Singular extension: search every *other* move against a
                    // window just below the table score. If they all fail low,
                    // this move stands alone and is worth an extra ply.
                    let sbeta = (tt_score - depth * 2).max(-MATE + 1);
                    self.stack[ply].excluded = m;
                    let v = self.negamax(pos, (depth - 1) / 2, sbeta - 1, sbeta, ply, cut_node, false);
                    self.stack[ply].excluded = Move::NULL;
                    if self.stop {
                        return 0;
                    }
                    if v < sbeta {
                        extension = 1 + (!pv_node && v < sbeta - 30) as i32;
                    } else if sbeta >= beta {
                        // Multi-cut: several moves beat beta, so does this node.
                        return sbeta;
                    } else if tt_score >= beta {
                        extension = -2;
                    }
                } else if in_check {
                    extension = 1;
                }
            }

            // Only needed by the reduction formula, so only computed when a
            // reduction is actually on the table.
            let will_reduce = depth >= 2 && moves_played >= 2 + root as i32;
            let gives = will_reduce && gives_check(pos, m);
            // Read before the move is made, which is the only time it means
            // anything. `history_of` indexes the butterfly table by `pos.stm`
            // and reads the victim off `pos.piece_at(to)`; after `make` the
            // side to move is the opponent and that square holds the piece that
            // just arrived. Asked afterwards it returned the opponent's history
            // for the move, and for a capture the history of capturing the
            // capturer -- so every reduction was being adjusted by a number
            // about a different move. The pruning path a few lines above always
            // asked before the move and was never affected.
            let hist = if will_reduce { self.history_of(pos, m, ply, moved, to, noisy) as i32 } else { 0 };
            pos.make(m);
            tt().prefetch(pos.key);
            self.stack[ply].mv = m;
            self.stack[ply].piece_to = moved * 64 + to + 1;
            moves_played += 1;

            let new_depth = depth - 1 + extension;
            let mut score;

            if moves_played == 1 {
                // The first move gets the full window; PVS assumes it is best
                // and spends its budget proving the rest are not.
                score = -self.negamax(pos, new_depth, -beta, -alpha, ply + 1, false, pv_node);
            } else {
                // --- late move reduction
                let mut r = 0i32;
                if will_reduce {
                    r = self.lmr[depth.min(63) as usize][(moves_played as usize).min(63)];
                    if pv_node {
                        r -= 1024;
                    }
                    if improving {
                        r -= 1024;
                    }
                    if cut_node {
                        r += 2048;
                    }
                    if gives {
                        r -= 1024;
                    }
                    if noisy {
                        r -= 1024;
                    }
                    // No table move means the ordering here is guesswork, so
                    // the tail of the list is worth even less than usual.
                    if tt_move.is_null() {
                        r += 1024;
                    }
                    // Moves with a good history are reduced less, and vice
                    // versa; this is where ordering pays for the pruning.
                    r -= (hist * 1024) / 8192;
                    r = r.clamp(0, (new_depth - 1).max(0) * 1024);
                }
                let rd = new_depth - (r >> 10);

                score = -self.negamax(pos, rd, -alpha - 1, -alpha, ply + 1, true, false);
                if score > alpha && rd < new_depth {
                    // The reduction was wrong; verify at full depth.
                    score = -self.negamax(pos, new_depth, -alpha - 1, -alpha, ply + 1, !cut_node, false);
                }
                if pv_node && score > alpha && score < beta {
                    score = -self.negamax(pos, new_depth, -beta, -alpha, ply + 1, false, true);
                }
            }

            pos.unmake(m);
            if self.stop {
                return 0;
            }

            if score > best {
                best = score;
                if score > alpha {
                    best_move = m;
                    alpha = score;
                    bound = BOUND_EXACT;
                    if pv_node {
                        self.update_pv(ply, m);
                    }
                    if score >= beta {
                        bound = BOUND_LOWER;
                        self.update_histories(pos, m, ply, depth, searched_quiets, searched_noisy, noisy);
                        break;
                    }
                }
            }

            if noisy {
                searched_noisy.push(m);
            } else {
                searched_quiets.push(m);
            }
        }

        if moves_played == 0 {
            // Every move was the excluded one: report the window edge rather
            // than a mate score, which would be a lie.
            return if !excluded.is_null() {
                alpha
            } else if in_check {
                -MATE + ply as i32
            } else {
                0
            };
        }

        if excluded.is_null() {
            tt().store(pos.key, best_move, best, raw_eval, depth, bound, ply);
            // Feed the residual back, but only from nodes that actually measured
            // it: not in check, not decided by a capture (whose score says more
            // about material than about the structure), and not from a bound
            // that points the wrong way to contradict the static eval.
            if !in_check
                && best.abs() < MATE_IN_MAX
                && (best_move.is_null() || !pos_is_noisy(pos, best_move))
                && !(bound == BOUND_LOWER && best <= corrected)
                && !(bound == BOUND_UPPER && best >= corrected)
            {
                self.update_corr(pos, ply, best - corrected, depth);
            }
        }
        best
    }

    // -----------------------------------------------------------------------
    // Quiescence
    // -----------------------------------------------------------------------

    fn qsearch(&mut self, pos: &mut Position, mut alpha: i32, beta: i32, ply: usize, pv_node: bool) -> i32 {
        self.check_stop();
        if self.stop {
            return 0;
        }
        self.nodes += 1;
        if ply > self.seldepth {
            self.seldepth = ply;
        }
        if pos.is_draw(ply) || pos.is_material_draw() {
            return 0;
        }
        let in_check = pos.in_check();
        if ply >= MAX_PLY - 1 {
            return if in_check { 0 } else { evaluate(pos) };
        }

        let hit = tt().probe(pos.key, ply);
        let mut tt_move = Move::NULL;
        let mut tt_eval = i32::MIN;
        if let Some(h) = &hit {
            tt_move = h.mv;
            tt_eval = h.eval;
            if !pv_node
                && (h.bound == BOUND_EXACT
                    || (h.bound == BOUND_LOWER && h.score >= beta)
                    || (h.bound == BOUND_UPPER && h.score <= alpha))
            {
                return h.score;
            }
        }

        let mut best;
        let raw_eval;
        let stand;
        if in_check {
            // In check there is no stand-pat: every evasion must be searched.
            best = -INF;
            raw_eval = -INF;
            stand = -INF;
        } else {
            raw_eval = if tt_eval != i32::MIN { tt_eval } else { evaluate(pos) };
            stand = self.corrected_eval(pos, ply, raw_eval);
            best = stand;
            if best >= beta {
                tt().store(pos.key, Move::NULL, best, raw_eval, 0, BOUND_LOWER, ply);
                return best;
            }
            if best > alpha {
                alpha = best;
            }
        }

        let list = list_at(self.id, ply, false);
        generate(pos, list, if in_check { GenKind::All } else { GenKind::Noisy });
        if list.n == 0 {
            return if in_check { -MATE + ply as i32 } else { best };
        }
        // In check, quiescence generates everything and searches losing
        // captures too, so their order still matters and the main search's
        // threshold stands. Out of check it discards them, which is what makes
        // the cheaper threshold safe.
        if in_check {
            self.score_moves::<-20>(pos, list, tt_move, ply);
        } else {
            self.score_moves::<0>(pos, list, tt_move, ply);
        }

        let mut best_move = Move::NULL;
        let mut bound = BOUND_UPPER;
        for i in 0..list.n {
            let m = list.pick(i);
            if !in_check {
                // Everything left over is a losing capture, so stop rather than
                // ask again. `score_moves` already ran `see_ge(m, -20)` on every
                // noisy move and put the failures below zero; a swap that cannot
                // clear -20 cannot clear the 0 this loop wants, so each of them
                // would be skipped below. `pick` hands out scores in
                // non-increasing order, which makes the rest of the list losing
                // as well. The tt move and queen promotions score in the
                // positive bands, so neither is caught by this.
                if list.sc[i] < 0 {
                    break;
                }
                // Delta pruning: even winning this material would not reach
                // alpha, so the whole branch is pointless.
                let gain = if m.is_ep() { SEE_VAL[PAWN_P] } else { SEE_VAL[pos.piece_at(m.to()) as usize] }
                    + if m.is_promo() { SEE_VAL[m.promo()] - SEE_VAL[PAWN_P] } else { 0 };
                if stand + gain + 150 < alpha {
                    continue;
                }
                // The tt move and queen promotions are scored by what they
                // are rather than what they win, so their sign says nothing
                // about the swap and they still get asked directly. Every other
                // capture was scored against this threshold above.
                if list.sc[i] >= (1 << 23) && !see_ge(pos, m, 0) {
                    continue;
                }
            }
            pos.make(m);
            let score = -self.qsearch(pos, -beta, -alpha, ply + 1, pv_node);
            pos.unmake(m);
            if self.stop {
                return 0;
            }
            if score > best {
                best = score;
                if score > alpha {
                    alpha = score;
                    best_move = m;
                    bound = BOUND_EXACT;
                    if score >= beta {
                        bound = BOUND_LOWER;
                        break;
                    }
                }
            }
        }
        tt().store(pos.key, best_move, best, raw_eval, 0, bound, ply);
        best
    }

    // -----------------------------------------------------------------------
    // Move ordering
    // -----------------------------------------------------------------------

    /// `THRESH` is the swap value a capture must clear to sort above the quiet
    /// moves. The main search wants -20, tolerating a capture that loses a
    /// little. Quiescence wants 0, because it refuses to search anything below
    /// that -- scoring at its own threshold lets the sign of the score stand in
    /// for the test rather than running the swap evaluation twice per capture.
    fn score_moves<const THRESH: i32>(
        &mut self,
        pos: &Position,
        list: &mut MoveList,
        tt_move: Move,
        ply: usize,
    ) {
        let stm = pos.stm;
        let prev = if ply > 0 { self.stack[ply - 1].piece_to } else { NO_PIECE_TO };
        let prev2 = if ply > 1 { self.stack[ply - 2].piece_to } else { NO_PIECE_TO };
        let counter =
            if prev != NO_PIECE_TO { self.counter[(prev - 1) / 64][(prev - 1) % 64] } else { Move::NULL };

        for i in 0..list.n {
            let m = list.mv[i];
            let from = m.from();
            let to = m.to();
            let pt = pos.piece_at(from) as usize;
            let moved = pc_index(stm, pt);

            list.sc[i] = if m == tt_move {
                1 << 24
            } else if m.is_promo() && m.promo() == QUEEN_P {
                (1 << 23) + 100
            } else if pos_is_noisy(pos, m) {
                let victim = if m.is_ep() { PAWN_P } else { pos.piece_at(to) as usize };
                let mvv = SEE_VAL[victim] * 16;
                let ch = self.capt_hist[moved][to][victim.min(6)] as i32;
                // Winning captures ahead of every quiet, losing ones behind.
                if see_ge(pos, m, THRESH) {
                    (1 << 22) + mvv + ch
                } else {
                    -(1 << 22) + mvv + ch
                }
            } else if m == self.killers[ply][0] {
                (1 << 21) + 2
            } else if m == self.killers[ply][1] {
                (1 << 21) + 1
            } else if m == counter {
                1 << 20
            } else {
                let mut s = self.history[stm][from][to] as i32;
                let idx = moved * 64 + to + 1;
                if prev != NO_PIECE_TO {
                    s += cont(self.id)[0][prev][idx] as i32;
                }
                if prev2 != NO_PIECE_TO {
                    s += cont(self.id)[1][prev2][idx] as i32;
                }
                s
            };
        }
    }

    #[inline(always)]
    fn history_of(&self, pos: &Position, m: Move, ply: usize, moved: usize, to: usize, noisy: bool) -> i16 {
        if noisy {
            let victim = if m.is_ep() { PAWN_P } else { pos.piece_at(to) as usize };
            return self.capt_hist[moved][to][victim.min(6)];
        }
        let mut s = self.history[pos.stm][m.from()][to] as i32;
        let idx = moved * 64 + to + 1;
        // The reduction formula reads the same tables the ordering does, or a
        // move can be ordered late and then reduced as though nothing were
        // known about it.
        if ply > 0 && self.stack[ply - 1].piece_to != NO_PIECE_TO {
            s += cont(self.id)[0][self.stack[ply - 1].piece_to][idx] as i32;
        }
        if ply > 1 && self.stack[ply - 2].piece_to != NO_PIECE_TO {
            s += cont(self.id)[1][self.stack[ply - 2].piece_to][idx] as i32;
        }
        s.clamp(-16_384, 16_384) as i16
    }

    /// Gravity update: values saturate toward ±`MAX_HIST` instead of drifting,
    /// so a table that has seen a million updates still responds to new ones.
    #[inline(always)]
    fn gravity(v: &mut i16, bonus: i32) {
        const MAX_HIST: i32 = 16_384;
        let b = bonus.clamp(-1200, 1200);
        let cur = *v as i32;
        *v = (cur + b - cur * b.abs() / MAX_HIST) as i16;
    }

    fn bonus_quiet(&mut self, pos: &Position, m: Move, ply: usize, bonus: i32) {
        let from = m.from();
        let to = m.to();
        let moved = pc_index(pos.stm, pos.piece_at(from) as usize);
        Self::gravity(&mut self.history[pos.stm][from][to], bonus);
        let idx = moved * 64 + to + 1;
        if ply > 0 {
            let p = self.stack[ply - 1].piece_to;
            if p != NO_PIECE_TO {
                Self::gravity(&mut cont(self.id)[0][p][idx], bonus);
            }
        }
        if ply > 1 {
            let p = self.stack[ply - 2].piece_to;
            if p != NO_PIECE_TO {
                Self::gravity(&mut cont(self.id)[1][p][idx], bonus);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn update_histories(
        &mut self,
        pos: &Position,
        best: Move,
        ply: usize,
        depth: i32,
        quiets: &Tried,
        noisy: &Tried,
        best_is_noisy: bool,
    ) {
        let bonus = (155 * depth - 80).clamp(0, 1200);
        let malus = -bonus;

        if best_is_noisy {
            let to = best.to();
            let moved = pc_index(pos.stm, pos.piece_at(best.from()) as usize);
            let victim = if best.is_ep() { PAWN_P } else { pos.piece_at(to) as usize };
            Self::gravity(&mut self.capt_hist[moved][to][victim.min(6)], bonus);
        } else {
            self.bonus_quiet(pos, best, ply, bonus);
            // Killers are per-ply and deliberately not per-position: a refutation
            // usually works against the whole family of sibling positions.
            if self.killers[ply][0] != best {
                self.killers[ply][1] = self.killers[ply][0];
                self.killers[ply][0] = best;
            }
            if ply > 0 {
                let p = self.stack[ply - 1].piece_to;
                if p != NO_PIECE_TO {
                    self.counter[(p - 1) / 64][(p - 1) % 64] = best;
                }
            }
            // Everything tried before the cutoff was, in hindsight, wrong.
            for i in 0..quiets.len() {
                let q = quiets.get(i);
                if q == best {
                    continue;
                }
                self.bonus_quiet(pos, q, ply, malus);
            }
        }
        for i in 0..noisy.len() {
            let q = noisy.get(i);
            if q == best {
                continue;
            }
            let to = q.to();
            let moved = pc_index(pos.stm, pos.piece_at(q.from()) as usize);
            let victim = if q.is_ep() { PAWN_P } else { pos.piece_at(to) as usize };
            Self::gravity(&mut self.capt_hist[moved][to][victim.min(6)], malus);
        }
    }
}

// ---------------------------------------------------------------------------
// Lazy SMP
// ---------------------------------------------------------------------------
//
// Every thread runs the whole iterative deepening loop over its own copy of the
// position, with its own history, killers, correction tables and move lists.
// Nothing is divided up and nothing is handed between them. The only thing they
// share is the transposition table, and that is the whole mechanism: a helper
// that reaches a node first leaves the answer there, and the thread that
// arrives second gets a cutoff it would otherwise have had to search for. The
// gain is not from doing different work on purpose, it is from the table making
// their work different as a side effect.
//
// Which is also why they start at staggered depths. Started together, several
// threads spend the first few iterations deriving identical trees before the
// table has anything in it to tell them apart.

/// One `Position` per helper, from `mmap` rather than a static array.
///
/// A `Position` carries 4096 game keys and a 136-entry undo stack, so it is
/// about forty kilobytes, and `Position::empty()` is not all-zero -- an empty
/// square is 6 and "no en passant" is 64. Eight of them as a static would be a
/// third of a megabyte of non-zero bytes in the binary image. From `mmap` they
/// cost nothing until a thread is actually asked for.
static HELPER_POS: SyncCell<[*mut Position; MAX_THREADS]> =
    SyncCell::new([core::ptr::null_mut(); MAX_THREADS]);

fn helper_pos(id: usize) -> *mut Position {
    let slots = unsafe { HELPER_POS.as_mut() };
    if slots[id].is_null() {
        slots[id] = sys::mmap(core::mem::size_of::<Position>()) as *mut Position;
    }
    slots[id]
}

/// Helper entry point. The argument is the thread id, passed in the pointer
/// itself rather than through a box there is no allocator to make.
extern "C" fn helper_entry(arg: *mut u8) -> *mut u8 {
    let id = arg as usize;
    if id == 0 || id >= MAX_THREADS {
        return core::ptr::null_mut();
    }
    let p = helper_pos(id);
    if p.is_null() {
        return core::ptr::null_mut();
    }
    // Never written to: a helper is silent, so `go` takes no branch that
    // touches it.
    let mut out = Out::new();
    searcher_at(id).go(unsafe { &mut *p }, &mut out);
    core::ptr::null_mut()
}

/// Reset every per-thread table for the threads that are actually in use.
/// Clearing all `MAX_THREADS` would memset twenty megabytes on a `ucinewgame`
/// that is going to be searched with one thread.
pub fn clear_all(threads: usize) {
    for i in 0..threads.clamp(1, MAX_THREADS) {
        searcher_at(i).clear();
    }
}

/// Run one search on `threads` threads and print for the main one.
pub fn go_threaded(pos: &mut Position, out: &mut Out, threads: usize) {
    let n = threads.clamp(1, MAX_THREADS);
    STOP.store(false, Ordering::Relaxed);
    for p in NODE_PUB.iter().take(n) {
        p.store(0, Ordering::Relaxed);
    }

    let main = searcher_at(0);
    main.threads = n;
    main.start_depth = 0;
    if n == 1 {
        main.go(pos, out);
        return;
    }

    let limits = main.limits;
    let overhead = main.move_overhead;
    let mut handles: [Option<sys::Thread>; MAX_THREADS] = [None; MAX_THREADS];
    for (i, h) in handles.iter_mut().enumerate().take(n).skip(1) {
        let dst = helper_pos(i);
        if dst.is_null() {
            continue;
        }
        // Zeroed pages from `mmap` are a valid `Position` -- every field is an
        // integer or an array of them -- so this overwrites rather than reads.
        unsafe { (*dst).clone_from(pos) };
        let hs = searcher_at(i);
        hs.threads = n;
        hs.limits = limits;
        // Only thread 0 owns the budget and the clock. A helper with its own
        // copy of them would stop on its own and stop helping.
        hs.limits.nodes = u64::MAX;
        hs.limits.depth = MAX_DEPTH;
        hs.limits.infinite = true;
        hs.move_overhead = overhead;
        hs.silent = true;
        hs.ignore_stdin = true;
        hs.start_depth = 1 + (i % 4) as i32;
        *h = sys::spawn(helper_entry, i as *mut u8);
    }

    searcher_at(0).go(pos, out);

    // The main thread has its move; everyone else is now doing work nobody will
    // read. Joining before returning is what keeps the next `position` command
    // from landing while a helper still holds a pointer into this one.
    STOP.store(true, Ordering::Relaxed);
    for h in handles.iter().take(n).skip(1).flatten() {
        sys::join(*h);
    }
}

#[inline(always)]
pub fn pc_index(c: usize, pt: usize) -> usize {
    c * 6 + pt.min(5)
}

#[inline(always)]
fn pos_is_noisy(pos: &Position, m: Move) -> bool {
    let _ = pos;
    m.is_capture() || m.is_ep()
}

/// Static exchange evaluation, as a threshold test: "is the material outcome
/// of this capture sequence at least `threshold`?"
///
/// Answering a yes/no question instead of computing the exact swap value lets
/// the loop exit as soon as the sign is decided, which is most of the time.
pub fn see_ge(pos: &Position, m: Move, threshold: i32) -> bool {
    if m.is_castle() {
        return 0 >= threshold;
    }
    let from = m.from();
    let to = m.to();

    let captured = if m.is_ep() {
        PAWN_P
    } else {
        let p = pos.piece_at(to);
        if p == NONE {
            6
        } else {
            p as usize
        }
    };
    let mut balance = SEE_VAL[captured] - threshold;
    if m.is_promo() {
        balance += SEE_VAL[m.promo()] - SEE_VAL[PAWN_P];
    }
    if balance < 0 {
        return false;
    }
    let next = if m.is_promo() { m.promo() } else { pos.piece_at(from) as usize };
    balance -= SEE_VAL[next];
    if balance >= 0 {
        return true;
    }

    let mut occ = pos.occ() ^ bit(from) ^ bit(to);
    if m.is_ep() {
        occ ^= bit(if pos.stm == WHITE { to - 8 } else { to + 8 });
    }
    let diag = pos.piece[BISHOP_P] | pos.piece[QUEEN_P];
    let orth = pos.piece[ROOK_P] | pos.piece[QUEEN_P];
    let mut attackers = pos.attackers_to(to, occ) & occ;
    let mut stm = pos.stm ^ 1;

    loop {
        let mine = attackers & pos.color[stm] & occ;
        if mine == 0 {
            break;
        }
        // Least valuable attacker first; anything else loses material.
        let mut pt = PAWN_P;
        while pt < 6 && mine & pos.piece[pt] == 0 {
            pt += 1;
        }
        let sq = lsb(mine & pos.piece[pt]);
        occ ^= bit(sq);
        // A departing piece can uncover a slider behind it.
        if pt == PAWN_P || pt == BISHOP_P || pt == QUEEN_P {
            attackers |= bishop_attacks(to, occ) & diag;
        }
        if pt == ROOK_P || pt == QUEEN_P {
            attackers |= rook_attacks(to, occ) & orth;
        }
        attackers &= occ;
        stm ^= 1;
        balance = -balance - 1 - SEE_VAL[pt];
        if balance >= 0 {
            // Capturing into a defended square with the king is not legal, so
            // that capture never happened.
            if pt == KING_P && attackers & pos.color[stm] != 0 {
                stm ^= 1;
            }
            break;
        }
    }
    stm != pos.stm
}
