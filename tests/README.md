# Tests

Lightweight unit tests for the Python side of MageZero. They cover the
helpers and CLI guards in `magezero.executor` (the surface added by
`generate-data` and `train`), without ever launching Maven, the JVM, the
inference server, or PyTorch training.

## Running

From the repo root:

```bash
uv sync --group dev   # installs pytest into .venv
uv run pytest         # all tests
uv run pytest -v      # verbose
uv run pytest tests/test_executor_helpers.py::test_load_endpoints_happy_path
```

The full suite runs in ~0.3 seconds on a laptop. No GPU, no Java, no network.

## What's covered

| File | What it tests |
|---|---|
| `test_executor_helpers.py` | `_load_krenko_server_endpoints` (YAML parsing + fallbacks), `_latest_model_below` (model walk-back), `gather_hdf5_files` (recursive glob) |
| `test_train_cli.py` | `train_cli` arg validation, data preflight, overwrite guard, `--use-previous-model` walkback, `--auto-increment` slot picking, `--auto-increment + --use-previous-model` composition |
| `test_generate_data.py` | `generate_data` arg validation, online/offline mode auto-detection, asymmetric two-server setup, `--auto-increment` slot picking, `--offline` override |

## What's deliberately NOT covered

- Actual `magezero.train.train()` execution (needs real data + GPU + minutes)
- The Flask inference server (needs a real model checkpoint)
- The XMage / KrenkoMain JVM subprocess (needs Maven + the sibling `mage/` repo)
- `H5Indexed` end-to-end data loading (needs synthetic HDF5 fixtures — possible but skipped for v1)

These are all "integration" surfaces that depend on heavy infra. They're worth
adding eventually, but they don't fit the "fast feedback in CI" goal.

## How the mocking works

Two fixtures in `conftest.py` make the CLI tests possible:

- **`fake_workspace`** — points the module-level `MAGEZERO_ROOT` and `MAGE_ROOT`
  constants in `magezero.executor` at a `tmp_path` directory containing empty
  `data/`, `models/`, and a sibling `mage/`. Each test stages whatever fake
  files it needs into that workspace.

- **`fake_train_module`** — replaces `sys.modules["magezero.train"]` with a
  stub module exposing a no-op `train()`. The lazy `from magezero.train import
  train as run_train` inside `train_cli` picks it up, so happy-path CLI tests
  can run end-to-end without ever importing the real PyTorch-heavy `train.py`.

The `mock_subprocess` fixture (in `test_generate_data.py`) monkeypatches
`run_command`, `start_server`, `wait_for_http`, and `update_model_weights`
on the `executor` module so KrenkoMain never actually launches.

## Adding new tests

For new helpers in `executor.py`, add unit tests to `test_executor_helpers.py`.
For new CLI flags or guards, prefer extending `test_train_cli.py` or
`test_generate_data.py`. Use the existing fixtures rather than reinventing
the workspace setup.
