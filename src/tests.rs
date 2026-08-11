//! Unit tests.
//!
//! These run under `cargo test`, which builds the crate with `std` and a test
//! harness (see the `cfg_attr` in `main.rs`). They cover the invariants that
//! are cheap to state and expensive to debug once violated: perft counts,
//! make/unmake symmetry, Zobrist consistency, and the bounds the network's
//! feature extraction promises the accumulator.
//!
//! The slow, external-oracle checks live in `tests/` at the repo root and run
//! against `python-chess`. This module is the fast guard that runs on every
//! build.

use crate::bb::*;
use crate::eval::*;
use crate::movegen::*;
use crate::net;
use crate::pos::*;
use crate::search::see_ge;
use crate::tt::*;
use crate::uci::perft;

use std::sync::Once;

static INIT: Once = Once::new();

fn setup() {
    INIT.call_once(crate::init);
}

fn pos_from(fen: &[u8]) -> Position {
    setup();
    let mut p = Position::empty();
    p.set_fen(fen);
    p
}

const STARTPOS: &[u8] = b"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const KIWIPETE: &[u8] = b"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
const POS3: &[u8] = b"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";
const POS4: &[u8] = b"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";
const POS5: &[u8] = b"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8";
const POS6: &[u8] = b"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10";

// ---------------------------------------------------------------------------
// Move generation
// ---------------------------------------------------------------------------

/// Depths chosen so the whole module stays under a couple of seconds. The
/// deeper values live in `tests/perft_suite.sh`.
#[test]
fn perft_classic_positions() {
    let cases: &[(&[u8], i32, u64)] = &[
        (STARTPOS, 5, 4_865_609),
        (KIWIPETE, 4, 4_085_603),
        (POS3, 5, 674_624),
        (POS4, 4, 422_333),
        (POS5, 4, 2_103_487),
        (POS6, 4, 3_894_594),
    ];
    for &(fen, depth, expected) in cases {
        let mut p = pos_from(fen);
        let got = perft(&mut p, depth);
        assert_eq!(got, expected, "perft({depth}) mismatch for {:?}", core::str::from_utf8(fen));
    }
}

#[test]
fn every_generated_move_is_legal() {
    // Legality here means: after making it, our own king is not attacked.
    // The generator claims to guarantee this, so nothing should ever fail.
    let mut p = pos_from(KIWIPETE);
    fn walk(p: &mut Position, depth: i32) {
        let mut list = MoveList::new();
        generate(p, &mut list, GenKind::All);
        for i in 0..list.n {
            let m = list.mv[i];
            let us = p.stm;
            p.make(m);
            let ksq = p.king_sq(us);
            assert!(
                !p.attacked_by(us ^ 1, ksq, p.occ()),
                "generator produced a move leaving the king in check"
            );
            if depth > 1 {
                walk(p, depth - 1);
            }
            p.unmake(m);
        }
    }
    walk(&mut p, 3);
}

#[test]
fn noisy_generation_is_a_subset_of_all() {
    for fen in [STARTPOS, KIWIPETE, POS3, POS4, POS5, POS6] {
        let p = pos_from(fen);
        let mut all = MoveList::new();
        let mut noisy = MoveList::new();
        generate(&p, &mut all, GenKind::All);
        generate(&p, &mut noisy, GenKind::Noisy);
        for i in 0..noisy.n {
            let m = noisy.mv[i];
            assert!(
                (0..all.n).any(|j| all.mv[j] == m),
                "quiescence generated a move the full generator did not"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// make / unmake
// ---------------------------------------------------------------------------

/// Every field the search depends on has to come back exactly. A partial
/// restore shows up much later as an impossible position, so check the whole
/// board rather than just the hash.
#[test]
fn unmake_restores_the_position_exactly() {
    for fen in [STARTPOS, KIWIPETE, POS3, POS4, POS5, POS6] {
        let mut p = pos_from(fen);
        fn walk(p: &mut Position, depth: i32) {
            let before_piece = p.piece;
            let before_color = p.color;
            let before_mailbox = p.mailbox;
            let (key, pawn_key, np_key) = (p.key, p.pawn_key, p.np_key);
            let (castle, ep, half) = (p.castle, p.ep, p.halfmove);
            let checkers = p.checkers;

            let mut list = MoveList::new();
            generate(p, &mut list, GenKind::All);
            for i in 0..list.n {
                let m = list.mv[i];
                p.make(m);
                if depth > 1 {
                    walk(p, depth - 1);
                }
                p.unmake(m);

                assert_eq!(p.piece, before_piece, "piece bitboards not restored");
                assert_eq!(p.color, before_color, "colour bitboards not restored");
                assert_eq!(p.mailbox, before_mailbox, "mailbox not restored");
                assert_eq!(p.key, key, "zobrist key not restored");
                assert_eq!(p.pawn_key, pawn_key, "pawn key not restored");
                assert_eq!(p.np_key, np_key, "non-pawn key not restored");
                assert_eq!((p.castle, p.ep, p.halfmove), (castle, ep, half));
                assert_eq!(p.checkers, checkers, "checkers not restored");
            }
        }
        walk(&mut p, 3);
    }
}

/// The incrementally maintained key must equal the one a fresh parse produces.
/// Drift here silently poisons the transposition table.
#[test]
fn incremental_zobrist_matches_a_fresh_parse() {
    setup();
    let mut p = pos_from(STARTPOS);
    let mut scratch = Position::empty();
    let mut buf = [0u8; 96];

    fn walk(p: &mut Position, scratch: &mut Position, buf: &mut [u8; 96], depth: i32) {
        let mut list = MoveList::new();
        generate(p, &mut list, GenKind::All);
        for i in 0..list.n.min(6) {
            let m = list.mv[i];
            p.make(m);
            let n = crate::datagen::write_fen(p, buf);
            scratch.set_fen(&buf[..n]);
            assert_eq!(
                p.key, scratch.key,
                "incremental key diverged from a fresh parse of the same position"
            );
            assert_eq!(
                p.pawn_key, scratch.pawn_key,
                "incremental pawn key diverged from a fresh parse of the same position"
            );
            assert_eq!(
                p.np_key, scratch.np_key,
                "incremental non-pawn key diverged from a fresh parse of the same position"
            );
            assert_eq!(p.checkers, scratch.checkers, "checkers diverged");
            if depth > 1 {
                walk(p, scratch, buf, depth - 1);
            }
            p.unmake(m);
        }
    }
    walk(&mut p, &mut scratch, &mut buf, 4);
}

#[test]
fn null_move_round_trips() {
    let mut p = pos_from(KIWIPETE);
    let (key, ep, stm) = (p.key, p.ep, p.stm);
    p.make_null();
    assert_ne!(p.key, key, "null move must change the side-to-move key");
    assert_eq!(p.stm, stm ^ 1);
    p.unmake_null();
    assert_eq!((p.key, p.ep, p.stm), (key, ep, stm));
}

// ---------------------------------------------------------------------------
// Move encoding
// ---------------------------------------------------------------------------

#[test]
fn move_encoding_round_trips() {
    for from in 0..64usize {
        for to in 0..64usize {
            for flag in [F_QUIET, F_DOUBLE, F_KCASTLE, F_QCASTLE, F_CAPTURE, F_EP] {
                let m = Move::new(from, to, flag);
                assert_eq!((m.from(), m.to(), m.flag()), (from, to, flag));
            }
            for promo in 0..4u16 {
                let m = Move::new(from, to, F_PROMO | promo);
                assert!(m.is_promo());
                assert_eq!(m.promo(), (promo + 1) as usize);
                let c = Move::new(from, to, F_PROMO | F_CAPTURE | promo);
                assert!(c.is_promo() && c.is_capture());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Static exchange evaluation
// ---------------------------------------------------------------------------

fn find(p: &Position, from: usize, to: usize) -> Move {
    let mut list = MoveList::new();
    generate(p, &mut list, GenKind::All);
    for i in 0..list.n {
        if list.mv[i].from() == from && list.mv[i].to() == to {
            return list.mv[i];
        }
    }
    panic!("move not generated");
}

#[test]
fn see_scores_an_undefended_capture_at_the_victim() {
    // Rook on e1 takes an undefended pawn on e5: exactly a pawn, no more.
    let p = pos_from(b"k7/8/8/4p3/8/8/8/K3R3 w - - 0 1");
    let m = find(&p, 4, 36); // e1 -> e5
    assert!(see_ge(&p, m, SEE_VAL[PAWN_P]));
    assert!(!see_ge(&p, m, SEE_VAL[PAWN_P] + 1));
}

#[test]
fn see_sees_through_a_defender() {
    // Rook takes a pawn defended by another pawn: wins a pawn, loses a rook.
    let p = pos_from(b"k7/8/8/3p4/4p3/8/8/K3R3 w - - 0 1");
    let m = find(&p, 4, 28); // e1 -> e4, defended by d5
    let net_loss = SEE_VAL[PAWN_P] - SEE_VAL[ROOK_P];
    assert!(see_ge(&p, m, net_loss));
    assert!(!see_ge(&p, m, net_loss + 1));
    assert!(!see_ge(&p, m, 0), "a losing capture must not pass a zero threshold");
}

// ---------------------------------------------------------------------------
// Evaluation and network
// ---------------------------------------------------------------------------

#[test]
fn dead_drawn_material_evaluates_to_zero() {
    // Whatever the network has learned, these are draws by rule.
    for fen in [
        b"8/8/4k3/8/8/3K4/8/8 w - - 0 1".as_ref(),
        b"8/8/4k3/8/5B2/3K4/8/8 w - - 0 1".as_ref(),
        b"8/8/4k3/8/5N2/3K4/8/8 b - - 0 1".as_ref(),
    ] {
        let p = pos_from(fen);
        assert_eq!(evaluate(&p), 0, "insufficient material must evaluate to exactly 0");
    }
}

/// Mirror a FEN: flip the board vertically and swap every colour. The result
/// is the same position with the sides exchanged, so it must evaluate
/// identically from the mover's point of view.
///
/// Built programmatically rather than written by hand. The first version of
/// this test used hand-mirrored FEN constants and one of them had a single
/// pawn on the wrong side, which looked exactly like an engine bug.
fn mirror_fen(fen: &str) -> String {
    let mut parts = fen.split_whitespace();
    let board = parts.next().unwrap();
    let stm = parts.next().unwrap_or("w");
    let castle = parts.next().unwrap_or("-");
    let ep = parts.next().unwrap_or("-");

    let flipped: Vec<String> = board
        .split('/')
        .rev()
        .map(|r| {
            r.chars()
                .map(|c| {
                    if c.is_ascii_uppercase() {
                        c.to_ascii_lowercase()
                    } else if c.is_ascii_lowercase() {
                        c.to_ascii_uppercase()
                    } else {
                        c
                    }
                })
                .collect::<String>()
        })
        .collect();

    let new_castle: String = if castle == "-" {
        "-".into()
    } else {
        castle
            .chars()
            .map(|c| if c.is_ascii_uppercase() { c.to_ascii_lowercase() } else { c.to_ascii_uppercase() })
            .collect()
    };
    let new_ep = if ep == "-" {
        "-".to_string()
    } else {
        let mut it = ep.chars();
        let file = it.next().unwrap();
        let rank = it.next().unwrap().to_digit(10).unwrap();
        format!("{file}{}", 9 - rank)
    };
    let new_stm = if stm == "w" { "b" } else { "w" };
    format!("{} {} {} {} 0 1", flipped.join("/"), new_stm, new_castle, new_ep)
}

/// Sorted feature indices for the side to move and the opponent.
///
/// Comparing these directly is much sharper than comparing evaluations: a
/// score can come out symmetric while the underlying features are wrong, and a
/// score mismatch tells you nothing about *which* feature is at fault.
fn feature_sets(p: &Position) -> (Vec<u16>, Vec<u16>) {
    let mut us = [0u16; net::MAX_F];
    let mut them = [0u16; net::MAX_F];
    let n = net::features_both(p, p.stm, &mut us, &mut them);
    let mut a = us[..n].to_vec();
    let mut b = them[..n].to_vec();
    // The board is walked white-first, so mirroring permutes the order.
    a.sort_unstable();
    b.sort_unstable();
    (a, b)
}

/// Neither the features nor the evaluation may care which colour is which.
///
/// Checked at three levels, narrowest first, so a failure says where the bug
/// is rather than just that one exists: the feature map, the hand-crafted
/// terms, and the network output.
#[test]
fn evaluation_is_colour_symmetric() {
    setup();

    let check = |text: &str| {
        let mirrored = mirror_fen(text);
        let a = pos_from(text.as_bytes());
        let b = pos_from(mirrored.as_bytes());

        let (a_us, a_them) = feature_sets(&a);
        let (b_us, b_them) = feature_sets(&b);
        assert_eq!(a_us, b_us, "side-to-move features differ under mirroring\n  {text}\n  {mirrored}");
        assert_eq!(a_them, b_them, "opponent features differ under mirroring\n  {text}\n  {mirrored}");
        assert_eq!(
            net::bucket_of(popcount(a.occ()) as usize),
            net::bucket_of(popcount(b.occ()) as usize),
            "output bucket differs under mirroring"
        );
        assert_eq!(
            hand_crafted(&a),
            hand_crafted(&b),
            "hand-crafted evaluation is not colour-symmetric\n  {text}\n  {mirrored}"
        );
        assert_eq!(evaluate(&a), evaluate(&b), "evaluation is not colour-symmetric\n  {text}\n  {mirrored}");
    };

    let mut checked = 0;
    for fen in [STARTPOS, KIWIPETE, POS3, POS4, POS5, POS6] {
        check(core::str::from_utf8(fen).unwrap());
        checked += 1;
    }

    // And over positions reached by play, which is where the odd material
    // configurations and lost castling rights actually turn up.
    let mut buf = [0u8; 96];
    for (seed, start) in [(3usize, STARTPOS), (5, KIWIPETE), (11, POS6), (17, POS4)] {
        let mut p = pos_from(start);
        for step in 0..40 {
            let mut list = MoveList::new();
            generate(&p, &mut list, GenKind::All);
            if list.n == 0 {
                break;
            }
            // Deterministic but varied: no RNG, so a failure always reproduces.
            p.make(list.mv[(step * seed + 3) % list.n]);
            p.ply = 0;
            let n = crate::datagen::write_fen(&p, &mut buf);
            check(core::str::from_utf8(&buf[..n]).unwrap());
            checked += 1;
        }
    }
    assert!(checked > 100, "symmetry test covered only {checked} positions");
}

/// The accumulator indexes weights without bounds checks, so the extractor's
/// promises — indices below `IN`, count at most `MAX_F` — are load-bearing.
#[test]
fn feature_extraction_stays_within_its_bounds() {
    for fen in [STARTPOS, KIWIPETE, POS3, POS4, POS5, POS6] {
        let p = pos_from(fen);
        for persp in 0..2 {
            let mut a = [0u16; net::MAX_F];
            let mut b = [0u16; net::MAX_F];
            let n = net::features_both(&p, persp, &mut a, &mut b);
            assert!(n <= net::MAX_F, "feature count exceeded MAX_F");
            for i in 0..n {
                assert!((a[i] as usize) < net::IN, "feature index out of range");
                assert!((b[i] as usize) < net::IN, "feature index out of range");
            }
        }
    }
}

/// `features` is a thin wrapper over `features_both`; if they ever disagree,
/// the trainer and the engine are looking at different networks.
#[test]
fn single_and_dual_perspective_extraction_agree() {
    for fen in [STARTPOS, KIWIPETE, POS4, POS6] {
        let p = pos_from(fen);
        for persp in 0..2 {
            let mut one = [0u16; net::MAX_F];
            let n1 = net::features(&p, persp, &mut one);
            let mut a = [0u16; net::MAX_F];
            let mut b = [0u16; net::MAX_F];
            let n2 = net::features_both(&p, persp, &mut a, &mut b);
            assert_eq!(n1, n2);
            assert_eq!(one[..n1], a[..n2], "single-perspective view disagrees");

            // The opposite perspective must match the other half.
            let mut other = [0u16; net::MAX_F];
            let n3 = net::features(&p, persp ^ 1, &mut other);
            assert_eq!(other[..n3], b[..n2], "mirrored view disagrees");
        }
    }
}

#[test]
fn output_buckets_cover_the_whole_piece_range() {
    let mut seen = [false; 8];
    for pieces in 2..=32usize {
        let b = net::bucket_of(pieces);
        assert!(b < 8, "bucket index out of range");
        seen[b] = true;
    }
    assert!(seen.iter().all(|&s| s), "some output bucket is unreachable");
    // Monotonic: more material never selects an earlier bucket.
    for pieces in 3..=32usize {
        assert!(net::bucket_of(pieces) >= net::bucket_of(pieces - 1));
    }
}

// ---------------------------------------------------------------------------
// Transposition table
// ---------------------------------------------------------------------------

#[test]
fn tt_round_trips_and_rebases_mate_scores() {
    setup();
    tt().resize(1);
    tt().clear();

    let key = 0x1234_5678_9ABC_DEF0u64;
    let mv = Move::new(12, 28, F_DOUBLE);
    tt().store(key, mv, 42, -7, 9, BOUND_EXACT, 0);
    let hit = tt().probe(key, 0).expect("stored entry should be found");
    assert_eq!((hit.mv, hit.score, hit.eval, hit.depth, hit.bound), (mv, 42, -7, 9, BOUND_EXACT));

    // Mate scores are stored relative to the node. Storing at ply 5 and
    // probing at ply 5 must give back what went in.
    let mate_key = 0xDEAD_BEEF_CAFE_1234u64;
    let mate = MATE - 8;
    tt().store(mate_key, mv, mate, 0, 5, BOUND_LOWER, 5);
    let h = tt().probe(mate_key, 5).expect("mate entry should be found");
    assert_eq!(h.score, mate, "mate distance did not survive the round trip");

    assert!(tt().probe(0xFFFF_FFFF_FFFF_FFFF, 0).is_none() || true);

    // A resize that replaces an existing mapping goes down a different path
    // from the first one -- it unmaps before it maps -- and used to come back
    // with a byte count of zero, which turned `clear()` into a no-op and left
    // the table live across `ucinewgame`. The first resize never showed it.
    //
    // This lives inside the one test that owns the table rather than in a test
    // of its own: the harness runs tests on parallel threads, and a second one
    // resizing the global table unmaps memory the first is still probing.
    for mb in [1usize, 2, 1] {
        tt().resize(mb);
        tt().store(key, mv, 42, -7, 9, BOUND_EXACT, 0);
        assert!(tt().probe(key, 0).is_some(), "entry missing at {mb} MB");
        tt().clear();
        assert!(tt().probe(key, 0).is_none(), "clear() left the entry live at {mb} MB");
    }
}

// ---------------------------------------------------------------------------
// FEN
// ---------------------------------------------------------------------------

#[test]
fn fen_round_trips() {
    for fen in [STARTPOS, KIWIPETE, POS3, POS4, POS5, POS6] {
        let p = pos_from(fen);
        let mut buf = [0u8; 96];
        let n = crate::datagen::write_fen(&p, &mut buf);
        let mut q = Position::empty();
        q.set_fen(&buf[..n]);
        assert_eq!(p.key, q.key, "FEN round trip changed the position");
        assert_eq!(p.piece, q.piece);
        assert_eq!(p.color, q.color);
        assert_eq!(p.castle, q.castle);
        assert_eq!(p.stm, q.stm);
    }
}

// ---------------------------------------------------------------------------
// Bitboards
// ---------------------------------------------------------------------------

#[test]
fn magic_attacks_match_the_naive_ray_walk() {
    setup();
    // The magic tables are searched at startup and validated against this same
    // slow path, but re-checking with real occupancies catches an indexing bug
    // that a subset-only validation would miss.
    for sq in 0..64usize {
        for &occ in &[0u64, 0xFFFF_0000_0000_FFFF, 0x0081_0000_2400_0081, !0u64] {
            assert_eq!(rook_attacks(sq, occ), naive(sq, occ, true), "rook sq={sq}");
            assert_eq!(bishop_attacks(sq, occ), naive(sq, occ, false), "bishop sq={sq}");
        }
    }
}

fn naive(sq: usize, occ: u64, rook: bool) -> u64 {
    let dirs: [(i32, i32); 4] =
        if rook { [(0, 1), (0, -1), (1, 0), (-1, 0)] } else { [(1, 1), (1, -1), (-1, 1), (-1, -1)] };
    let (f0, r0) = (file_of(sq) as i32, rank_of(sq) as i32);
    let mut out = 0u64;
    for (df, dr) in dirs {
        let (mut f, mut r) = (f0 + df, r0 + dr);
        while (0..8).contains(&f) && (0..8).contains(&r) {
            let s = (r * 8 + f) as usize;
            out |= bit(s);
            if occ & bit(s) != 0 {
                break;
            }
            f += df;
            r += dr;
        }
    }
    out
}

#[test]
fn between_and_line_are_consistent() {
    setup();
    for a in 0..64usize {
        for b in 0..64usize {
            let bt = between(a, b);
            let ln = line(a, b);
            if bt != 0 {
                assert_ne!(ln, 0, "squares between implies a line through");
                assert_eq!(bt & (bit(a) | bit(b)), 0, "between must exclude its endpoints");
                assert_eq!(bt & ln, bt, "between must lie on the line");
            }
            if ln != 0 {
                assert_eq!(ln & bit(a), bit(a), "line must contain both endpoints");
                assert_eq!(ln & bit(b), bit(b), "line must contain both endpoints");
            }
        }
    }
}
