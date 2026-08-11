#!/usr/bin/env bash
# The six classic perft positions, with values that are genuinely standard.
#
# Every other position this suite once contained came from my memory, and five
# of them were wrong. python-chess (tests/perft_oracle.py) is the oracle for
# anything beyond these — never a remembered constant.
#
# pos3 is the one worth being sure about: a bare pawns-and-rooks endgame that
# leans on en passant, pinned pawn captures and rook x-rays, which is where a
# generator that is subtly wrong shows it. Its numbers have since been recomputed
# move for move with python-chess at every depth from 1 to 7 — 14, 191, 2812,
# 43238, 674624, 11030083, 178633661 — and all seven agree with the engine. The
# d8 figure below is still a published constant; three billion nodes is out of
# reach for a pure-Python reference.

cd "$(dirname "$0")/.." || exit 1
ENGINE=${ENGINE:-./target/release/sable}

run() { # fen depth expected label
  got=$(printf "position fen %s\ngo perft %d\nquit\n" "$1" "$2" \
        | "$ENGINE" | grep '^nodes' | awk '{print $2}')
  if [ "$got" = "$3" ]; then
    echo "PASS $4 d$2 = $got"
  else
    echo "FAIL $4 d$2 got=$got want=$3"
    fail=1
  fi
}

fail=0
run "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" 6 119060324 startpos
run "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1" 5 193690690 kiwipete
run "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1" 7 178633661 pos3
run "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1" 6 706045033 pos4
run "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8" 5 89941194 pos5
run "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10" 6 6923051137 pos6

# DEEP=1 adds a ply to four of them: 17 billion nodes against published values
# rather than 8. Roughly a minute, so CI runs the set above and this is for
# when movegen or make/unmake has actually been touched. The depths below are
# where the standard tables stop, which is the point -- past here there is no
# published number to check against and the python-chess oracle is the only
# honest reference.
if [ "${DEEP:-0}" = "1" ]; then
  run "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" 7 3195901860 startpos
  run "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1" 6 8031647685 kiwipete
  run "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1" 8 3009794393 pos3
  run "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8" 6 3048196529 pos5
fi
exit $fail
