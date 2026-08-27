#!/usr/bin/env python3
"""Publish the distilled network and its trainer to the Hugging Face Hub."""

import os
import sys
import time

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

REPO = os.environ.get("HF_REPO", "shubhxho/sable-chess-net")
RETRIES = int(os.environ.get("HF_RETRIES", "5"))
RETRY_SLEEP = float(os.environ.get("HF_RETRY_SLEEP", "8"))


def with_retry(label, fn):
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            last = e
            wait = RETRY_SLEEP * attempt
            print(f"  retry {attempt}/{RETRIES} {label}: {type(e).__name__}: {e}", flush=True)
            if attempt < RETRIES:
                time.sleep(wait)
    raise last


api = HfApi()
who = with_retry("whoami", lambda: api.whoami()["name"])
print(f"authenticated as {who}, target {REPO}")

with_retry(
    "create_repo",
    lambda: api.create_repo(REPO, repo_type="model", exist_ok=True, private=False),
)

for local, remote in [
    ("MODEL_CARD.md", "README.md"),
    ("net.bin", "net.bin"),
    ("train.py", "train.py"),
    ("src/net.rs", "reference/net.rs"),
    ("arena.py", "arena.py"),
    # The card quotes an Elo anchored against Stockfish. Ship the harness that
    # produced it, so the claim can be checked rather than taken on trust.
    ("tests/calibrate.py", "calibrate.py"),
]:
    if not os.path.exists(local):
        print(f"  skip {local} (missing)")
        continue

    def _upload(local=local, remote=remote):
        return api.upload_file(
            path_or_fileobj=local,
            path_in_repo=remote,
            repo_id=REPO,
            repo_type="model",
        )

    with_retry(f"upload {local}", _upload)
    print(f"  {local} -> {remote} ({os.path.getsize(local)} bytes)")

print(f"https://huggingface.co/{REPO}")
