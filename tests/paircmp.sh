#!/usr/bin/env bash
# Paired bench comparison. usage: paircmp.sh N binA binB
#
# Comparing two medians taken minutes apart is worthless on a machine that
# drifts -- the same binary has measured 495 ms and 428 ms in one session. So
# measure A and B adjacently and keep the *difference*: whatever the machine is
# doing at that moment hits both halves of a pair about equally and cancels.
# Pair order alternates so neither binary always runs into a cold cache.
set -u
N=$1; A=$2; B=$3
raw=$(mktemp)
for i in $(seq 1 "$N"); do
  if (( i % 2 )); then first=$A; second=$B; else first=$B; second=$A; fi
  fa=$(printf 'bench 13\n' | "$first"  | awk '/Time/{print $4}')
  fb=$(printf 'bench 13\n' | "$second" | awk '/Time/{print $4}')
  if (( i % 2 )); then echo "$fa $fb" >> "$raw"; else echo "$fb $fa" >> "$raw"; fi
done
python3 - "$raw" "$A" "$B" <<'EOF'
import sys
rows = [tuple(map(int, l.split())) for l in open(sys.argv[1])]
a = [r[0] for r in rows]; b = [r[1] for r in rows]
d = [x - y for x, y in rows]          # positive => B faster
n = len(d)
sd = sorted(d)
med = sd[n // 2]
# Bootstrap-free interval: order statistics give an exact sign-test CI for the
# median of the paired differences.
lo, hi = sd[max(0, n // 2 - int(0.98 * (n ** 0.5)))], sd[min(n - 1, n // 2 + int(0.98 * (n ** 0.5)))]
wins = sum(1 for x in d if x > 0); ties = sum(1 for x in d if x == 0)
base = sorted(a)[len(a) // 2]
print(f"pairs {n}   A {sys.argv[2].split('/')[-1]}  B {sys.argv[3].split('/')[-1]}")
print(f"A median {base} ms   min {min(a)}     B median {sorted(b)[len(b)//2]} ms   min {min(b)}")
print(f"paired delta (A-B): median {med:+d} ms  ci[{lo:+d},{hi:+d}]  = {100.0*med/base:+.2f}% for B")
print(f"B faster in {wins}/{n} pairs ({ties} ties)")
EOF
rm -f "$raw"
