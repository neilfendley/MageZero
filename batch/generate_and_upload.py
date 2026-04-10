"""Wrapper script for AWS Batch: run generate-data, then upload results to S3.

Required env vars:
    S3_BUCKET       - target S3 bucket
    DECK_PATH       - deck path relative to mage repo (e.g. decks/IzzetElementals.dck)
    VERSION         - data version number

Optional env vars:
    GAMES           - number of games (default 18)
    THREADS         - KrenkoMain thread count (default 8)
    MAX_TURNS       - max turns per game (default 40)
    OPPONENT_DECK   - opponent deck path (omit for mirror)
    OFFLINE         - set to "1" to force offline mode
    S3_PREFIX       - S3 key prefix (default "data")
"""
import os
import subprocess
import sys
from pathlib import Path

import boto3


def required_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"ERROR: {name} env var is required", file=sys.stderr)
        sys.exit(1)
    return val


def main() -> None:
    s3_bucket = required_env("S3_BUCKET")
    deck_path = required_env("DECK_PATH")
    version = required_env("VERSION")

    games = os.environ.get("GAMES", "18")
    threads = os.environ.get("THREADS", "8")
    max_turns = os.environ.get("MAX_TURNS", "40")
    opponent_deck = os.environ.get("OPPONENT_DECK")
    offline = os.environ.get("OFFLINE", "")
    s3_prefix = os.environ.get("S3_PREFIX", "data")

    # Build generate-data command (entry point installed by magezero package)
    cmd = [
        "generate-data",
        "--deck-path", deck_path,
        "--version", version,
        "--games", games,
        "--threads", threads,
        "--max-turns", max_turns,
    ]
    if opponent_deck:
        cmd += ["--opponent-deck", opponent_deck]
    if offline.strip().lower() in {"1", "true", "yes"}:
        cmd += ["--offline"]

    print(f"[batch] Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)

    # Upload generated data to S3
    # MAGEZERO_ROOT is computed from executor.py's __file__ location;
    # data lands relative to it regardless of cwd.
    magezero_root = Path(__file__).resolve().parent.parent
    data_dir = magezero_root / "data" / f"ver{version}" / "training"
    if not data_dir.exists():
        print(f"[batch] No data directory at {data_dir}", file=sys.stderr)
        sys.exit(1)

    s3 = boto3.client("s3")
    files = list(data_dir.rglob("*.hdf5")) + list(data_dir.rglob("*.h5"))
    if not files:
        print("[batch] No .hdf5/.h5 files to upload", file=sys.stderr)
        sys.exit(1)

    for f in sorted(files):
        rel = f.relative_to(data_dir)
        key = f"{s3_prefix}/ver{version}/training/{rel}"
        print(f"[batch] Uploading {f.name} -> s3://{s3_bucket}/{key}", flush=True)
        s3.upload_file(str(f), s3_bucket, key)

    print(f"[batch] Uploaded {len(files)} file(s) to s3://{s3_bucket}/{s3_prefix}/ver{version}/training/")


if __name__ == "__main__":
    main()
