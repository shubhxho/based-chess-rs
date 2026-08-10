//! Self-play data generation for network distillation.
//!
//! Emits one line per position: `FEN | score | result`, where `score` is the
//! search's evaluation in centipawns from white's point of view and `result`
//! is the eventual game outcome (0, 1 or 2 for black win / draw / white win).
//!
//! The filtering matters more than the volume. Positions are dropped when the
//! side to move is in check or the best move is a capture, because in those
//! positions the static evaluation the network is learning is not what decides
//! the game — the tactic is. Training on them teaches the network to imitate
//! search, which it cannot do.

use crate::eval::*;
use crate::io::Out;
use crate::movegen::*;
use crate::pos::*;
use crate::search::*;
use crate::sys::SyncCell;
use crate::tt::tt;

const MAX_GAME: usize = 400;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// One recorded position: FEN text plus the score, banked until the game ends
/// and the result is known.
struct Rec {
    fen: [u8; 96],
    len: usize,
    score: i32,
}

impl Copy for Rec {}
impl Clone for Rec {
    fn clone(&self) -> Self {
        *self
    }
}

static GAME: SyncCell<[Rec; MAX_GAME]> = SyncCell::new([Rec { fen: [0; 96], len: 0, score: 0 }; MAX_GAME]);

/// Positions already written, as an open-addressed set of Zobrist keys.
///
/// Self-play from randomised openings converges: the same middlegame structures
/// come back game after game, and a duplicate teaches the network nothing while
/// still costing a training slot. A key of zero means the slot is free, so the
/// table starts empty in BSS and needs no initialisation pass.
const SEEN_BITS: usize = 22;
const SEEN_N: usize = 1 << SEEN_BITS;
static SEEN: SyncCell<[u64; SEEN_N]> = SyncCell::new([0; SEEN_N]);

/// True the first time a key is offered, false every time after. Probes a short
/// run of slots and then gives up: a full table should stop deduplicating, not
/// spin looking for room.
fn first_sighting(key: u64) -> bool {
    if key == 0 {
        return true;
    }
    let t = unsafe { SEEN.as_mut() };
    let mut i = (key >> (64 - SEEN_BITS)) as usize;
    for _ in 0..8 {
        if t[i] == key {
            return false;
        }
        if t[i] == 0 {
            t[i] = key;
            return true;
        }
        i = (i + 1) & (SEEN_N - 1);
    }
    true
}

/// Serialise a position as FEN. Written by hand because the trainer needs to
/// read these with an off-the-shelf parser.
pub fn write_fen(pos: &Position, buf: &mut [u8; 96]) -> usize {
    const CH: &[u8; 12] = b"PNBRQKpnbrqk";
    let mut n = 0;
    for r in (0..8).rev() {
        let mut empty = 0;
        for f in 0..8 {
            let sq = r * 8 + f;
            let p = pos.piece_at(sq);
            if p == NONE {
                empty += 1;
            } else {
                if empty > 0 {
                    buf[n] = b'0' + empty;
                    n += 1;
                    empty = 0;
                }
                buf[n] = CH[pos.color_at(sq) * 6 + p as usize];
                n += 1;
            }
        }
        if empty > 0 {
            buf[n] = b'0' + empty;
            n += 1;
        }
        if r > 0 {
            buf[n] = b'/';
            n += 1;
        }
    }
    buf[n] = b' ';
    n += 1;
    buf[n] = if pos.stm == WHITE { b'w' } else { b'b' };
    n += 1;
    buf[n] = b' ';
    n += 1;
    if pos.castle == 0 {
        buf[n] = b'-';
        n += 1;
    } else {
        for (bit, ch) in [(WK, b'K'), (WQ, b'Q'), (BK, b'k'), (BQ, b'q')] {
            if pos.castle & bit != 0 {
                buf[n] = ch;
                n += 1;
            }
        }
    }
    buf[n] = b' ';
    n += 1;
    if pos.ep == 64 {
        buf[n] = b'-';
        n += 1;
    } else {
        buf[n] = b'a' + (pos.ep as usize & 7) as u8;
        buf[n + 1] = b'1' + (pos.ep as usize >> 3) as u8;
        n += 2;
    }
    // Halfmove and fullmove counters; the trainer ignores them but parsers
    // expect the fields to exist.
    buf[n] = b' ';
    buf[n + 1] = b'0';
    buf[n + 2] = b' ';
    buf[n + 3] = b'1';
    n + 4
}

fn random_move(pos: &Position, rng: &mut Rng) -> Move {
    let mut list = MoveList::new();
    generate(pos, &mut list, GenKind::All);
    if list.n == 0 {
        return Move::NULL;
    }
    list.mv[(rng.next() % list.n as u64) as usize]
}

pub fn run(target: u64, nodes: u64, seed: u64, out: &mut Out) {
    let mut rng = Rng(seed | 1);
    let s = searcher();
    s.silent = true;
    let mut pos = Position::empty();
    let mut emitted = 0u64;
    let mut games = 0u64;

    while emitted < target {
        games += 1;
        pos.set_startpos();
        // Wiping a 16 MB table and 2.4 MB of history every game costs more
        // than the games gain in independence. The table's generation counter
        // already ages stale entries out, so a periodic clear is enough to stop
        // one game's tree from steering the next.
        if games % 16 == 1 {
            tt().clear();
            s.clear();
        } else {
            tt().new_search();
        }

        // Random opening. Symmetric play from one book line would give the
        // network a very narrow slice of the position space, and a narrow
        // opening span gives it a narrow slice of game *phases* -- so the
        // length varies as much as the moves do.
        let plies = 8 + (rng.next() % 9) as usize;
        let mut ok = true;
        for _ in 0..plies {
            let m = random_move(&pos, &mut rng);
            if m.is_null() {
                ok = false;
                break;
            }
            pos.make(m);
            pos.ply = 0;
        }
        if !ok {
            continue;
        }
        // Throw away openings that random play has already decided.
        s.limits = Limits::new();
        s.limits.nodes = nodes;
        s.go(&mut pos, out);
        if s.best.is_null() || s.best_score.abs() > 400 {
            continue;
        }

        let mut n_rec = 0usize;
        // 2 = white win, 1 = draw, 0 = black win.
        let mut result = 1u8;
        let mut adjudicate = 0i32;

        for ply in 0..MAX_GAME {
            if pos.is_draw(0) || pos.is_material_draw() {
                result = 1;
                break;
            }
            s.limits = Limits::new();
            s.limits.nodes = nodes;
            s.go(&mut pos, out);
            if s.best.is_null() {
                // No legal move: checkmate or stalemate.
                result = if pos.in_check() {
                    if pos.stm == WHITE {
                        0
                    } else {
                        2
                    }
                } else {
                    1
                };
                break;
            }
            let score = s.best_score;
            let best = s.best;

            // Keep only quiet, non-tactical positions, and only once the
            // random opening has been answered: the first couple of plies are
            // the engine repairing whatever the dice did, not play.
            let quiet = !pos.in_check() && !best.is_capture() && !best.is_promo();
            if quiet && ply >= 2 && score.abs() < 2000 && n_rec < MAX_GAME && first_sighting(pos.key) {
                let white_score = if pos.stm == WHITE { score } else { -score };
                let r = &mut unsafe { GAME.as_mut() }[n_rec];
                r.len = write_fen(&pos, &mut r.fen);
                r.score = white_score;
                n_rec += 1;
            }

            // Adjudicate decisive positions rather than playing out trivially
            // won endgames; those plies teach the network nothing.
            if score.abs() >= 2000 {
                adjudicate += 1;
                if adjudicate >= 4 {
                    let white_ahead = if pos.stm == WHITE { score > 0 } else { score < 0 };
                    result = if white_ahead { 2 } else { 0 };
                    break;
                }
            } else {
                adjudicate = 0;
            }
            if score.abs() >= MATE_IN_MAX {
                let white_ahead = if pos.stm == WHITE { score > 0 } else { score < 0 };
                result = if white_ahead { 2 } else { 0 };
                break;
            }

            pos.make(best);
            pos.ply = 0;
        }

        let recs = unsafe { GAME.as_ref() };
        for r in recs.iter().take(n_rec) {
            out.s(&r.fen[..r.len]);
            out.s(b" | ").i(r.score as i64);
            out.s(b" | ").u(result as u64);
            out.c(b'\n');
            emitted += 1;
        }
        out.flush();
        let _ = games;
    }
    s.silent = false;
}
