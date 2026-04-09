"""Shared test fixtures.

Most CLI tests in this suite need to point `MAGEZERO_ROOT` (the constant in
`magezero.executor`) at a temporary directory so they can stage fake `data/`
and `models/` trees without touching the real repo. They also need to mock the
KrenkoMain subprocess and the lazy `magezero.train` import so nothing actually
runs Maven or PyTorch.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def fake_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point `magezero.executor.MAGEZERO_ROOT` and `MAGE_ROOT` at a tmpdir.

    Returns the tmpdir, which contains an empty `data/` and `models/` and a
    sibling `mage/` so the path resolution in `generate_data()` works.
    """
    from magezero import executor

    workspace = tmp_path / "workspace"
    magezero_root = workspace / "MageZero"
    mage_root = workspace / "mage"
    (magezero_root / "data").mkdir(parents=True)
    (magezero_root / "models").mkdir(parents=True)
    mage_root.mkdir(parents=True)

    monkeypatch.setattr(executor, "MAGEZERO_ROOT", magezero_root)
    monkeypatch.setattr(executor, "MAGE_ROOT", mage_root)
    return magezero_root


@pytest.fixture
def fake_train_module(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Replace `magezero.train` in sys.modules with a fake module exposing a
    no-op `train()`. Returns a dict the test can read to assert the fake
    `train()` was invoked.
    """
    state = {"called": 0}

    fake = types.ModuleType("magezero.train")
    fake.train = lambda: state.__setitem__("called", state["called"] + 1)
    monkeypatch.setitem(sys.modules, "magezero.train", fake)
    return state


@pytest.fixture(autouse=True)
def reset_executor_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI entry points call _setup_logging() which configures the root
    logger with `force=True`. That's fine in production but it leaks across
    tests and can swallow caplog. Stub it out for the test session.
    """
    from magezero import executor

    monkeypatch.setattr(executor, "_setup_logging", lambda: None)


@pytest.fixture(autouse=True)
def isolate_magezero_env() -> None:
    """`train_cli` and `generate_data` mutate `os.environ` directly (e.g.
    `os.environ["MAGEZERO_USE_PREVIOUS_MODEL"] = "1"`). pytest's `monkeypatch`
    only auto-reverts variables it set itself, so without this fixture those
    direct mutations leak between tests. Snapshot and restore everything that
    starts with `MAGEZERO_`.
    """
    snapshot = {k: v for k, v in os.environ.items() if k.startswith("MAGEZERO_")}
    # Clear them all so each test starts from a clean slate.
    for k in list(os.environ):
        if k.startswith("MAGEZERO_"):
            del os.environ[k]
    try:
        yield
    finally:
        for k in [k for k in os.environ if k.startswith("MAGEZERO_")]:
            del os.environ[k]
        os.environ.update(snapshot)
