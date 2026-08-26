#!/usr/bin/env python3
"""Delete byte-identical aug_hf_*.txt shards, keeping the lowest index.

    python scripts/purge_dup_shards.py data/lichess-sf
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dir", type=Path, help="directory with aug_hf_*.txt")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    files = sorted(args.dir.glob("aug_hf_*.txt"))
    if not files:
        print(f"no shards in {args.dir}", file=sys.stderr)
        return 1
    by_hash: dict[str, list[Path]] = {}
    for f in files:
        by_hash.setdefault(file_md5(f), []).append(f)
    removed = 0
    for dig, group in by_hash.items():
        if len(group) < 2:
            continue
        keep, *dups = group
        print(f"keep {keep.name}; remove {[d.name for d in dups]} ({dig[:8]})")
        for d in dups:
            removed += 1
            if not args.dry_run:
                d.unlink()
    print(f"{'would remove' if args.dry_run else 'removed'} {removed} duplicate shard(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
