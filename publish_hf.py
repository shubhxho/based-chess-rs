#!/usr/bin/env python3
"""Publish the distilled network and its trainer to the Hugging Face Hub."""

import os
import sys

from huggingface_hub import HfApi

REPO = os.environ.get("HF_REPO", "shubhxho/sable-chess-net")

api = HfApi()
who = api.whoami()["name"]
print(f"authenticated as {who}, target {REPO}")

api.create_repo(REPO, repo_type="model", exist_ok=True, private=False)

for local, remote in [
    ("MODEL_CARD.md", "README.md"),
    ("net.bin", "net.bin"),
    ("train.py", "train.py"),
    ("src/net.rs", "reference/net.rs"),
    ("arena.py", "arena.py"),
]:
    if not os.path.exists(local):
        print(f"  skip {local} (missing)")
        continue
    api.upload_file(
        path_or_fileobj=local,
        path_in_repo=remote,
        repo_id=REPO,
        repo_type="model",
    )
    print(f"  {local} -> {remote} ({os.path.getsize(local)} bytes)")

print(f"https://huggingface.co/{REPO}")
