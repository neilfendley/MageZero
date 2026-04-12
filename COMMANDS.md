# MageZero Commands

All commands run via `uv run <cmd>`. One-time setup: `uv sync`.

## Data Generation

```bash
# Offline mirror match (no model needed)
uv run generate-data --deck-path decks/IzzetElementals.dck --version 0 --games 30

# Online mirror (uses models/IzzetElementals/ver0/model.pt to generate ver1 data)
uv run generate-data --deck-path decks/IzzetElementals.dck --version 1

# Asymmetric matchup
uv run generate-data --deck-path decks/IzzetElementals.dck --opponent-deck decks/JeskaiControl.dck --version 0

# Auto-increment picks the next empty data/ver{N}/training/ slot
uv run generate-data --deck-path decks/IzzetElementals.dck --auto-increment

# Force offline even when a model exists
uv run generate-data --deck-path decks/IzzetElementals.dck --version 1 --offline
```

Key flags: `--games` (default 18), `--threads` (default 8), `--max-turns` (default 40),
`--external-server` (skip server startup), `--server-threads` (default 6).

## Training

```bash
# Train from scratch on data/ver0/training/
uv run train --deck-name IzzetElementals --version 0 --epochs 60

# Fine-tune from latest checkpoint (AlphaZero-style iteration)
uv run train --deck-name IzzetElementals --auto-increment --use-previous-model --epochs 20
```

Key flags: `--batch-size` (default 128), `--force` (overwrite existing model),
`--train-opponent-head`. Extra knobs via env vars — see top of `src/magezero/train.py`.

## Inspection

```bash
# Inspect training data with human-readable feature/action names
# Auto-loads ../mage/FeatureTable.txt if present
uv run inspect-data data/ver0/training --samples 5

# Dataset statistics and plots (output to stats_out/)
MAGEZERO_DECK_NAME=IzzetElementals MAGEZERO_VER_NUMBER=0 uv run stats

# Interactive pdb debugger with dataset loaded (edit deck/version in game_exploration.py)
uv run explore
```

## Batch (round-robin, orchestration)

```bash
# Round-robin self-play across all decks (no server, no auto-copy to MageZero/data/)
uv run round --version 0 --games-per-test 30 --threads 8

# Full self-play / train / reload loop per deck (experimental)
uv run onedeck --iterations 5 --games-per-test 30 --threads 8 --train-epochs 20
```

## Inference Server

```bash
# Start manually (server.py's main() is broken, use waitress directly)
MAGEZERO_DECK_NAME=IzzetElementals MAGEZERO_VER_NUMBER=0 \
  uv run waitress-serve --host=127.0.0.1 --port=50052 --threads=6 magezero.server:app

# Hot-swap model weights at runtime
curl -X POST http://127.0.0.1:50052/load -H 'Content-Type: application/json' \
  -d '{"path": "models/IzzetElementals/ver1/model.pt"}'
```

## Docker (AWS Batch)

```bash
# Build shaded KrenkoMain jar (run from mtg/mage/)
mvn package -T 1C -pl Mage.Tests -am -DskipTests

# Update pinned CPU-only Python deps (run from mtg/MageZero/)
uv export --no-dev --no-hashes --no-emit-project 2>/dev/null | grep -v 'nvidia\|cuda-\|triton\|^#\|^$' | sed '/^    #/d' > batch/requirements-cpu.txt

# Build base image, then batch image (run from mtg/)
docker build --platform linux/amd64 -f MageZero/batch/Dockerfile.base -t magezero-base .
docker build --platform linux/amd64 -f MageZero/batch/Dockerfile -t magezero-batch .

# Test locally — offline, 1 game
docker run --rm --platform linux/amd64 --entrypoint sh magezero-batch -c "generate-data --deck-path decks/MonoRAggro.dck --version 99 --games 1 --threads 1 --max-turns 1 --offline"

# Test with S3 upload (mount ~/.aws for credentials)
docker run --rm --platform linux/amd64 -v ~/.aws:/root/.aws:ro \
  -e S3_BUCKET=my-bucket -e S3_PREFIX=batch-output -e DECK_PATH=decks/MonoRAggro.dck \
  -e VERSION=99 -e GAMES=1 -e THREADS=1 -e MAX_TURNS=5 -e OFFLINE=1 magezero-batch
```

## Logging

```bash
# Set Python log level (default INFO)
MAGEZERO_LOG_LEVEL=DEBUG uv run generate-data ...
```
