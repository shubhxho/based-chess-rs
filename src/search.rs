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
//!             ├ whole-node pruning: reverse futility, razoring, null move
//!             ├ move loop
//!             │   ├ per-move pruning: late-move, futility, SEE
//!             │   ├ extensions: check, singular
//!             │   └ late-move reduction + re-search ladder
//!             └ quiescence at the horizon
//!
//! Everything below the move loop is ordered so the cheapest test that can
//! reject a move runs first. Ordering quality is what makes the pruning safe,
//! so the history tables get as much attention as the pruning rules do.

use crate::bb::*;
use crate::eval::*;
use crate::io::{move_str, Out};
use crate::movegen::*;
use crate::pos::*;
use crate::sys::{self, SyncCell};
use crate::tt::*;

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
const MOVE_OVERHEAD: u64 = 25;

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

pub static SEARCHER: SyncCell<Searcher> = SyncCell::new(Searcher {
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
    killers: [[Move::NULL; 2]; MAX_PLY + 4],
    history: [[[0; 64]; 64]; 2],
    capt_hist: [[[0; 7]; 64]; 12],
    counter: [[Move::NULL; 64]; 12],
    stack: [Node {
        mv: Move::NULL,
        piece_to: NO_PIECE_TO,
        eval: 0,
        excluded: Move::NULL,
        in_check: false,
    }; MAX_PLY + 8],
    pv: [[Move::NULL; MAX_PLY]; MAX_PLY],
    pv_len: [0; MAX_PLY],
    lmr: [[0; 64]; 64],
});

/// Continuation history: indexed by the previous move's `piece*64+to`, then
/// this move's. Two plies back, which is where most of the signal is.
///
/// Deliberately a static of its own rather than a field of `Searcher`: a global
/// only lands in BSS when its *entire* initialiser is zero, and `Searcher` has
/// non-zero fields. Keeping these 2.4 MB separate is the difference between a
/// 200 KB binary and a 2.6 MB one.
static CONT: SyncCell<[[[i16; CONT_N]; CONT_N]; 2]> = SyncCell::new([[[0; CONT_N]; CONT_N]; 2]);

#[inline(always)]
fn cont() -> &'static mut [[[i16; CONT_N]; CONT_N]; 2] {
    unsafe { CONT.as_mut() }
}

#[inline(always)]
pub fn searcher() -> &'static mut Searcher {
    unsafe { SEARCHER.as_mut() }
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

impl Searcher {
    pub fn init_tables(&mut self) {
        for d in 1..64 {
            for m in 1..64 {
                self.lmr[d][m] = 790 + ln1024(d as u32) * ln1024(m as u32) / 2417;
            }
        }
    }

    pub fn clear(&mut self) {
        self.killers = [[Move::NULL; 2]; MAX_PLY + 4];
        self.history = [[[0; 64]; 64]; 2];
        self.capt_hist = [[[0; 7]; 64]; 12];
        self.counter = [[Move::NULL; 64]; 12];
        for t in cont().iter_mut() {
            for r in t.iter_mut() {
                r.fill(0);
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
            let t = self.limits.movetime.saturating_sub(MOVE_OVERHEAD).max(1);
            self.soft = t;
            self.hard = t;
            return;
        }
        if self.limits.time[stm] == 0 {
            self.soft = u64::MAX;
            self.hard = u64::MAX;
            return;
        }
        let avail = self.limits.time[stm].saturating_sub(MOVE_OVERHEAD);
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

    #[inline(always)]
    fn check_stop(&mut self) {
        if self.stop {
            return;
        }
        if self.nodes >= self.limits.nodes {
            self.stop = true;
            return;
        }
        if self.nodes & 2047 == 0 {
            if !self.limits.infinite && self.elapsed() >= self.hard {
                self.stop = true;
            }
            if crate::uci::interrupted() {
                self.stop = true;
            }
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
        tt().new_search();
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

        for depth in 1..=max_depth {
            let mut delta = 10 + (score * score) / 12_000;
            let (mut alpha, mut beta) = if depth < 4 {
                (-INF, INF)
            } else {
                ((score - delta).max(-INF), (score + delta).min(INF))
            };
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
            // Starting depth d+1 costs roughly 1.5x what depth d did; if that
            // would not fit in the soft budget, stop while we are ahead.
            if !self.limits.infinite && self.limits.movetime == 0 && self.elapsed() * 3 / 2 >= self.soft {
                break;
            }
            if self.limits.movetime > 0 && self.elapsed() >= self.soft {
                break;
            }
            if score.abs() >= MATE_IN_MAX && depth >= 4 {
                break;
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
        out.s(b" nodes ").u(self.nodes);
        out.s(b" nps ").u(self.nodes * 1000 / ms.max(1));
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
        // The table's score is a better estimate than the static eval when it
        // is on the right side of it.
        let eval = if !in_check
            && tt_bound != BOUND_NONE
            && tt_score.abs() < MATE_IN_MAX
            && ((tt_bound == BOUND_LOWER && tt_score > raw_eval)
                || (tt_bound == BOUND_UPPER && tt_score < raw_eval)
                || tt_bound == BOUND_EXACT)
        {
            tt_score
        } else {
            raw_eval
        };
        self.stack[ply].eval = raw_eval;
        self.stack[ply].in_check = in_check;

        // "Improving": is our position better than it was two plies ago? If
        // not, we are on the back foot and should prune less.
        let improving = if in_check {
            false
        } else if ply >= 2 && self.stack[ply - 2].eval != -INF {
            raw_eval > self.stack[ply - 2].eval
        } else if ply >= 4 && self.stack[ply - 4].eval != -INF {
            raw_eval > self.stack[ply - 4].eval
        } else {
            true
        };

        // --- whole-node pruning
        if !pv_node && !in_check && excluded.is_null() && beta.abs() < MATE_IN_MAX {
            // Reverse futility: so far ahead that even giving up material
            // repeatedly would not drop us below beta.
            if depth < 9 && eval - 75 * depth + 60 * improving as i32 >= beta {
                return beta + (eval - beta) / 3;
            }

            // Razoring: so far behind that only a tactic saves us; let the
            // quiescence search look for one.
            if depth <= 5 && eval + 180 * depth < alpha {
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
                let r = 4 + depth / 3 + ((eval - beta) / 200).min(3);
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
        }

        // Internal iterative reduction: with no table move, this node is not
        // worth its nominal depth — the ordering will be poor either way.
        if tt_move.is_null() && depth >= 4 && !in_check {
            depth -= 1 + pv_node as i32;
        }

        // --- move loop
        let mut list = MoveList::new();
        generate(pos, &mut list, GenKind::All);
        if list.n == 0 {
            return if in_check { -MATE + ply as i32 } else { 0 };
        }
        self.score_moves(pos, &mut list, tt_move, ply);

        let mut best = -INF;
        let mut best_move = Move::NULL;
        let mut bound = BOUND_UPPER;
        let mut searched_quiets = MoveList::new();
        let mut searched_noisy = MoveList::new();
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
                    let lmr_depth = (depth - (self.lmr[depth.min(63) as usize][(moves_played as usize).min(63)] >> 10)).max(0);
                    if lmr_depth < 8 && eval + 120 + 110 * lmr_depth <= alpha {
                        skip_quiets = true;
                        continue;
                    }
                    // Quiet moves that hang material are rarely the answer.
                    if lmr_depth < 7 && !see_ge(pos, m, -25 * lmr_depth * lmr_depth) {
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
                    // Moves with a good history are reduced less, and vice
                    // versa; this is where ordering pays for the pruning.
                    let h = self.history_of(pos, m, ply, moved, to, noisy) as i32;
                    r -= (h * 1024) / 8192;
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
                        self.update_histories(pos, m, ply, depth, &searched_quiets, &searched_noisy, noisy);
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
            return if !excluded.is_null() { alpha } else if in_check { -MATE + ply as i32 } else { 0 };
        }

        if excluded.is_null() {
            tt().store(pos.key, best_move, best, raw_eval, depth, bound, ply);
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
        if in_check {
            // In check there is no stand-pat: every evasion must be searched.
            best = -INF;
            raw_eval = -INF;
        } else {
            raw_eval = if tt_eval != i32::MIN { tt_eval } else { evaluate(pos) };
            best = raw_eval;
            if best >= beta {
                tt().store(pos.key, Move::NULL, best, raw_eval, 0, BOUND_LOWER, ply);
                return best;
            }
            if best > alpha {
                alpha = best;
            }
        }

        let mut list = MoveList::new();
        generate(pos, &mut list, if in_check { GenKind::All } else { GenKind::Noisy });
        if list.n == 0 {
            return if in_check { -MATE + ply as i32 } else { best };
        }
        self.score_moves(pos, &mut list, tt_move, ply);

        let mut best_move = Move::NULL;
        let mut bound = BOUND_UPPER;
        for i in 0..list.n {
            let m = list.pick(i);
            if !in_check {
                // Delta pruning: even winning this material would not reach
                // alpha, so the whole branch is pointless.
                let gain = if m.is_ep() {
                    SEE_VAL[PAWN_P]
                } else {
                    SEE_VAL[pos.piece_at(m.to()) as usize]
                } + if m.is_promo() { SEE_VAL[m.promo()] - SEE_VAL[PAWN_P] } else { 0 };
                if raw_eval + gain + 150 < alpha {
                    continue;
                }
                if !see_ge(pos, m, 0) {
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

    fn score_moves(&mut self, pos: &Position, list: &mut MoveList, tt_move: Move, ply: usize) {
        let stm = pos.stm;
        let prev = if ply > 0 { self.stack[ply - 1].piece_to } else { NO_PIECE_TO };
        let prev2 = if ply > 1 { self.stack[ply - 2].piece_to } else { NO_PIECE_TO };
        let counter = if prev != NO_PIECE_TO { self.counter[(prev - 1) / 64][(prev - 1) % 64] } else { Move::NULL };

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
                if see_ge(pos, m, -20) {
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
                    s += cont()[0][prev][idx] as i32;
                }
                if prev2 != NO_PIECE_TO {
                    s += cont()[1][prev2][idx] as i32;
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
        if ply > 0 && self.stack[ply - 1].piece_to != NO_PIECE_TO {
            s += cont()[0][self.stack[ply - 1].piece_to][idx] as i32;
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
                Self::gravity(&mut cont()[0][p][idx], bonus);
            }
        }
        if ply > 1 {
            let p = self.stack[ply - 2].piece_to;
            if p != NO_PIECE_TO {
                Self::gravity(&mut cont()[1][p][idx], bonus);
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
        quiets: &MoveList,
        noisy: &MoveList,
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
            for i in 0..quiets.n {
                let q = quiets.mv[i];
                if q == best {
                    continue;
                }
                self.bonus_quiet(pos, q, ply, malus);
            }
        }
        for i in 0..noisy.n {
            let q = noisy.mv[i];
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
