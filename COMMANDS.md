# MageZero Commands

Reference for every `uv run <command>` exposed by this repo. All commands are
defined as console-script entry points in `pyproject.toml` under `[project.scripts]`
and run inside the project's virtual environment.

If you've never run anything yet, start with [Getting Started](#getting-started).

## Contents

- [Getting Started](#getting-started)
- [Logging](#logging)
- [Commands](#commands)
  - [`uv run generate-data`](#uv-run-generate-data) — produce self-play training data
  - [`uv run train`](#uv-run-train) — train a model on collected data
  - [`uv run round`](#uv-run-round) — round-robin self-play across all decks
  - [`uv run onedeck`](#uv-run-onedeck) — full self-play / train / reload loop
  - [`uv run stats`](#uv-run-stats) — dataset statistics and plots
  - [`uv run explore`](#uv-run-explore) — interactive dataset debugger
  - [`uv run server`](#uv-run-server) — inference server (currently broken — use `waitress-serve`)

---

## Getting Started

### Prerequisites

- Python 3.14+ (uv will install it for you if missing)
- Java 21 + Maven, with the sibling `mage/` repo built (`cd ../mage && make install`)
- `uv` installed (<https://docs.astral.sh/uv/>)
- This repo and `mage/` cloned as siblings:

  ```
  mtg/
  ├── mage/
  └── MageZero/   ← you are here
  ```

### One-time install

```bash
uv sync
```

This creates `.venv/`, installs all dependencies from `pyproject.toml`/`uv.lock`,
and registers the console scripts. You don't need to activate the venv —
`uv run <cmd>` handles that automatically.

### A minimal end-to-end pass

Generate some data with the offline (no-model) MCTS, then train a first model on it:

```bash
uv run generate-data --deck-path decks/IzzetElementals.dck --version 0 --games 10 --threads 4
uv run train --deck-name IzzetElementals --version 0 --epochs 10
```

After training you'll have:

```
data/ver0/training/...hdf5
models/IzzetElementals/ver0/model.pt
models/IzzetElementals/ver0/ignore.roar
```

### The iterative loop

Once you have a `ver0` model, you can keep stepping with `--auto-increment` —
each `generate-data` run uses the most recent model trained at any earlier
version, and each `train` run picks the next free model slot. Pair `train`
with `--use-previous-model` so each new version is initialized from the
latest existing checkpoint instead of training from scratch:

```bash
# v1: generates data using ver0 model (online), then trains a ver1 model
# initialized from ver0's weights
uv run generate-data --deck-path decks/IzzetElementals.dck --auto-increment --games 30
uv run train --deck-name IzzetElementals --auto-increment --use-previous-model --epochs 20

# v2: uses ver1 model, initialized from ver1's weights, ...
uv run generate-data --deck-path decks/IzzetElementals.dck --auto-increment --games 30
uv run train --deck-name IzzetElementals --auto-increment --use-previous-model --epochs 20
```

---

## Logging

The new commands (`generate-data`, `train`) use Python's `logging` module. Set
the level via env var:

```bash
MAGEZERO_LOG_LEVEL=DEBUG uv run generate-data ...
MAGEZERO_LOG_LEVEL=WARNING uv run train ...
```

Defaults to `INFO`. Valid values: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
Invalid values silently fall back to `INFO`. Only the Python side reads this var;
the Java side (XMage) has its own log4j config.

---

## Commands

### `uv run generate-data`

Run a single batch of XMage self-play and copy the resulting `.hdf5` / `.h5`
files into `MageZero/data/ver{N}/training/` so `train` can pick them up.

**Mode auto-detection.** Looks for the most recent trained model strictly *below*
the target version: it walks `models/{deck}/ver{N-1}/model.pt`,
`ver{N-2}/model.pt`, … and uses the first one it finds. If a model exists,
`generate-data` runs in **online** mode: it starts a Flask inference server,
loads that model, and lets MCTS query it during search. If no earlier model
exists, it runs in **offline** mode (heuristic eval, uniform priors). `--offline`
forces offline regardless. This mirrors the AlphaZero loop convention — data at
version `N` is generated with the model trained on version `N-1`.

**Server lifecycle.** In online mode, the inference server is started, awaited
on `/healthz`, and torn down when the run ends. Pass `--external-server` to skip
startup if you're already running one.

The host and ports are read directly from the same `krenko_config.yml` file
that KrenkoMain uses (default: `mage/Mage.MageZero/config/krenko_config.yml`,
overridable via `--config-path`). Both sides stay in sync automatically — to
change the address, edit `server.host`, `server.port`, and `server.opponent_port`
in the YAML and re-run. If the YAML can't be parsed or omits those keys, the
Python side falls back to `127.0.0.1:50052/50053`.

**Mirror vs asymmetric.** Omit `--opponent-deck` for a mirror match (one server,
one model). Pass it for an asymmetric matchup — two servers will be started, one
per deck, on the `server.port` and `server.opponent_port` from the krenko YAML
config (defaults: `50052` and `50053`).

**Auto-increment.** `--auto-increment` picks the lowest version number where
`data/ver{N}/training/` is empty (or doesn't exist), so you never accidentally
mix new data into an old slot. Mutually exclusive with `--version`.

**Progress.** A `tqdm` bar polls the XMage output directory every 2 seconds and
ticks up as new `.hdf5` / `.h5` files appear. Total is set to `--games`, but
the file count may differ from the game count, so the bar is best read as a
throughput indicator rather than an exact percentage.

**Overwrite safety.** If the destination already contains `.hdf5` or `.h5`
files, the command logs a `WARNING` and proceeds (data is added alongside the
existing files in a fresh timestamp subdirectory).

#### Common flags

| Flag | Default | Notes |
|---|---|---|
| `--deck-path PATH` | required | e.g. `decks/IzzetElementals.dck`, relative to the `mage/` repo |
| `--deck-name NAME` | derived from `--deck-path` stem | used for model lookup |
| `--opponent-deck PATH` | none (mirror match) | enables asymmetric mode |
| `--opponent-name NAME` | derived from opponent path stem | |
| `--version N` | `0` | data version to write to |
| `--auto-increment` | off | pick the next empty `data/ver{N}/training/` slot |
| `--games N` | `18` | number of games to generate |
| `--threads N` | `8` | XMage worker thread count |
| `--max-turns N` | `40` | turn cap per game |
| `--offline` | off | force offline mode |
| `--external-server` | off | don't start the inference server (assume one is running) |
| `--server-threads N` | `6` | waitress thread count for the Python inference server |
| `--mage-output-dir DIR` | `data` | folder inside `mage/` where XMage writes hdf5 |
| `--dest-dir DIR` | `MageZero/data/ver{N}/training` | where new hdf5s get copied |

#### Examples

```bash
# Offline mirror match — bootstrap when no model exists yet
uv run generate-data --deck-path decks/IzzetElementals.dck --version 0 --games 30

# Offline asymmetric matchup
uv run generate-data \
  --deck-path decks/IzzetElementals.dck \
  --opponent-deck decks/JeskaiControl.dck \
  --version 0

# Online mirror — uses models/IzzetElementals/ver0/model.pt to generate ver1 data
uv run generate-data --deck-path decks/IzzetElementals.dck --version 1

# Iterative loop — let auto-increment pick the next slot
uv run generate-data --deck-path decks/IzzetElementals.dck --auto-increment

# Force offline even though a previous model exists
uv run generate-data --deck-path decks/IzzetElementals.dck --version 1 --offline
```

---

### `uv run train`

Train (or fine-tune) a model on the HDF5 files in `data/ver{N}/training/`.

This is a thin CLI wrapper around `magezero.train.train`. Flags translate to
`MAGEZERO_*` environment variables, which `train.py` reads at module load.
Omitted flags fall through to existing env-var defaults, so previous workflows
that set the env vars directly still work.

#### Flags

| Flag | Env var equivalent | Default | Notes |
|---|---|---|---|
| `--deck-name NAME` | `MAGEZERO_DECK_NAME` | `IzzetElementals` | filters HDF5s by filename prefix |
| `--version N` | `MAGEZERO_VER_NUMBER` | `0` | reads from `data/ver{N}/training`, writes to `models/{deck}/ver{N}/` |
| `--auto-increment` | — | off | pick the lowest version where `models/{deck}/ver{N}/model.pt` doesn't exist |
| `--epochs N` | `MAGEZERO_EPOCH_COUNT` | `60` | |
| `--batch-size N` | `MAGEZERO_BATCH_SIZE` | `128` | |
| `--use-previous-model` | `MAGEZERO_USE_PREVIOUS_MODEL=1` | off | resume from same-version checkpoint, or if absent, fall back to the latest checkpoint at any earlier version |
| `--train-opponent-head` | `MAGEZERO_TRAIN_OPPONENT_HEAD=1` | off | train the opponent policy head |
| `--force` | — | off | overwrite an existing model checkpoint at the target version |

Other knobs (`MAGEZERO_GLOBAL_MAX`, `MAGEZERO_EMBEDDING_DIM`, `MAGEZERO_HIDDEN_DIM`,
`MAGEZERO_MIXED_PRECISION`, `MAGEZERO_NUM_WORKERS`, …) are env-var only — see
the top of `src/magezero/train.py` for the full list.

**Overwrite safety.** Refuses to start if `models/{deck}/ver{N}/model.pt`
already exists. Three escape hatches:

- `--force` — overwrite intentionally
- `--use-previous-model` — exempt; resumes training. If the same-version
  `model.pt` exists, it fine-tunes in place. If not, it walks back to the
  latest checkpoint at any earlier version, copies it (and `ignore.roar`)
  into the target directory, and trains from there. This is the AlphaZero-style
  init-from-previous-version path.
- bump `--version`, or pass `--auto-increment` to pick the next free slot

`--auto-increment` and `--version` are mutually exclusive (they're two ways
to specify the same thing). `--use-previous-model` is **independent** of both
and is the recommended companion to `--auto-increment` for AlphaZero-style
iteration: pick the next free version slot, and initialize it from the latest
existing checkpoint.

**Progress.** Two nested `tqdm` bars: an outer epoch counter and an inner
batch counter (`leave=False`, so the inner bar disappears between epochs).

**Device handling.** Auto-detects CUDA. On CPU it disables mixed precision and
`pin_memory` automatically. To force CPU on a CUDA box: `CUDA_VISIBLE_DEVICES=`
prefix.

#### Examples

```bash
# Train from scratch
uv run train --deck-name IzzetElementals --version 0 --epochs 60

# Fine-tune from an existing same-version checkpoint
uv run train --deck-name IzzetElementals --version 1 --epochs 20 --use-previous-model

# Auto-pick the next free version slot AND init from the latest existing
# checkpoint (the AlphaZero loop pairing)
uv run train --deck-name IzzetElementals --auto-increment --use-previous-model --epochs 20

# Equivalent env-var-only invocation (everything uv run train accepts is
# also driven by MAGEZERO_* env vars; the CLI flags are just overrides)
MAGEZERO_DECK_NAME=IzzetElementals MAGEZERO_VER_NUMBER=0 MAGEZERO_EPOCH_COUNT=60 uv run train
```

---

### `uv run round`

Run KrenkoMain self-play across every pair of decks in the hardcoded `DECKS`
list at the top of `src/magezero/executor.py`. No inference server is started.
**Generated `.hdf5` files are NOT copied into `MageZero/data/` automatically** —
you'll find them under `mage/data/ver{N}/...` and need to move them yourself if
you want `train` to pick them up.

Useful for batch-generating bootstrap data with the offline MCTS. For more
control, prefer `generate-data`.

#### Notable flags

| Flag | Default |
|---|---|
| `--version` | `0` |
| `--games-per-test` | `18` |
| `--threads` | `18` |
| `--max-turns` | `40` |
| `--output-dir` | `data` (folder inside `mage/`) |

```bash
uv run round --version 0 --games-per-test 30 --threads 8
```

---

### `uv run onedeck`

The full self-play / train / reload loop for the hardcoded deck list. Trains a
separate model per deck, then alternates between generating online self-play
data, retraining each model, and reloading the inference servers. Inherits
`round`'s flag set plus training-specific flags (`--iterations`, `--train-epochs`,
`--train-opponent-head`).

Note: the loop has a few rough edges — the first iteration intentionally skips
self-play (`if iteration == 1: break`), and the file-copy step has occasionally
failed in practice. Treat this as a long-running, somewhat experimental
orchestration script. For day-to-day work, prefer composing `generate-data` and
`train` yourself.

```bash
uv run onedeck --iterations 5 --games-per-test 30 --threads 8 --train-epochs 20
```

---

### `uv run stats`

Compute dataset statistics and write plots to `stats_out/`. Reads the same
`MAGEZERO_DECK_NAME` / `MAGEZERO_VER_NUMBER` env vars as `train`. No CLI flags.

```bash
MAGEZERO_DECK_NAME=IzzetElementals MAGEZERO_VER_NUMBER=0 uv run stats
```

Outputs include feature-count summaries, value-label histograms, and confusion
matrices. Inspect `src/magezero/dataset_stats.py` for the full list.

---

### `uv run explore`

Drops you into a `pdb` breakpoint with an `H5Indexed` dataset already loaded so
you can poke at samples interactively. The deck name and version are currently
**hardcoded** at the top of `src/magezero/game_exploration.py` (`BWBats`, `0`) —
edit those directly to inspect a different dataset.

```bash
uv run explore
```

---

### `uv run server`

**Currently broken.** `pyproject.toml` points the entry at `magezero.server:main`,
but `server.py` doesn't define a `main()` function. To start the inference
server, run waitress directly:

```bash
MAGEZERO_DECK_NAME=IzzetElementals MAGEZERO_VER_NUMBER=0 \
  uv run waitress-serve --host=127.0.0.1 --port=50052 --threads=6 magezero.server:app
```

The server reads `MAGEZERO_DECK_NAME` and `MAGEZERO_VER_NUMBER` to locate the
initial model and ignore-list files under `models/{DECK_NAME}/ver{N}/`. After
startup, push different weights at runtime via:

```bash
curl -X POST http://127.0.0.1:50052/load \
  -H 'Content-Type: application/json' \
  -d '{"path": "models/IzzetElementals/ver1/model.pt"}'
```

Endpoints: `/healthz` (liveness), `/load` (swap weights), and the inference
endpoint queried by the XMage MCTS player.
