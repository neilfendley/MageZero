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
    MODEL_S3_KEY    - S3 key for model dir (e.g. models/MonoRAggro/ver0/)
                      Downloads model.pt and ignore.roar into the right local path
                      so generate-data runs in online mode.
"""
import os
import subprocess
import sys
from pathlib import Path

import boto3


MAGEZERO_ROOT = Path(__file__).resolve().parent.parent


def required_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"ERROR: {name} env var is required", file=sys.stderr)
        sys.exit(1)
    return val


def download_model(s3, bucket: str, model_key: str) -> None:
    """Download model files from S3 into the local models directory."""
    model_key = model_key.rstrip("/") + "/"
    local_dir = MAGEZERO_ROOT / model_key
    local_dir.mkdir(parents=True, exist_ok=True)

    for filename in ("model.pt", "ignore.roar"):
        s3_key = f"{model_key}{filename}"
        local_path = local_dir / filename
        try:
            print(f"[batch] Downloading s3://{bucket}/{s3_key} -> {local_path}", flush=True)
            s3.download_file(bucket, s3_key, str(local_path))
        except Exception as e:
            if filename == "model.pt":
                print(f"ERROR: Failed to download model: {e}", file=sys.stderr)
                sys.exit(1)
            print(f"[batch] Optional file {filename} not found, skipping", flush=True)


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
    model_s3_key = os.environ.get("MODEL_S3_KEY")
    search_budget = os.environ.get("SEARCH_BUDGET")
    shard_id = os.environ.get("AWS_BATCH_JOB_ID", os.environ.get("SHARD_ID", "local"))

    s3 = boto3.client("s3")

    if model_s3_key:
        download_model(s3, s3_bucket, model_s3_key)

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
    if search_budget:
        cmd += ["--search-budget", search_budget]

    print(f"[batch] Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)

    data_dir = MAGEZERO_ROOT / "data" / f"ver{version}" / "training"
    if not data_dir.exists():
        print(f"[batch] No data directory at {data_dir}", file=sys.stderr)
        sys.exit(1)

    files = list(data_dir.rglob("*.hdf5")) + list(data_dir.rglob("*.h5"))
    if not files:
        print("[batch] No .hdf5/.h5 files to upload", file=sys.stderr)
        sys.exit(1)

    for f in sorted(files):
        rel = f.relative_to(data_dir)
        key = f"{s3_prefix}/ver{version}/training/{shard_id}/{rel}"
        print(f"[batch] Uploading {f.name} -> s3://{s3_bucket}/{key}", flush=True)
        s3.upload_file(str(f), s3_bucket, key)

    print(f"[batch] Uploaded {len(files)} file(s) to s3://{s3_bucket}/{s3_prefix}/ver{version}/training/{shard_id}/")


if __name__ == "__main__":
    main()
