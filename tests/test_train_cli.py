"""Tests for `magezero.executor.train_cli` — argument parsing, safety guards,
and the --use-previous-model walkback. Does NOT actually run training; the
real `magezero.train` module is replaced via the `fake_train_module` fixture.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from magezero import executor


def _stage_data(magezero_root: Path, version: int, deck: str, ext: str = "hdf5") -> Path:
    """Create a fake training shard so the data preflight passes."""
    d = magezero_root / "data" / f"ver{version}" / "training"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{deck}_v{version}.{ext}"
    f.write_bytes(b"")
    return f


def _stage_model(magezero_root: Path, deck: str, version: int, with_ignore: bool = True) -> Path:
    p = magezero_root / "models" / deck / f"ver{version}" / "model.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"weights")
    if with_ignore:
        (p.parent / "ignore.roar").write_bytes(b"ignore")
    return p


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------

def test_auto_increment_and_version_are_mutually_exclusive(
    fake_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["train", "--deck-name", "X", "--auto-increment", "--version", "2"])
    with pytest.raises(SystemExit) as exc:
        executor.train_cli()
    assert exc.value.code == 1


def test_auto_increment_and_use_previous_model_now_compose(
    fake_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_train_module: dict,
) -> None:
    """Regression test for the codex finding: --auto-increment + --use-previous-model
    used to be rejected. They should now compose to enable the AlphaZero loop."""
    _stage_model(fake_workspace, "X", 0)
    _stage_data(fake_workspace, 1, "X")
    monkeypatch.setattr(sys, "argv", ["train", "--deck-name", "X", "--auto-increment", "--use-previous-model"])
    executor.train_cli()
    assert fake_train_module["called"] == 1
    # The auto-incremented version should be 1, and use-previous-model should
    # have copied the ver0 weights into ver1.
    assert (fake_workspace / "models" / "X" / "ver1" / "model.pt").exists()


# ---------------------------------------------------------------------------
# data preflight
# ---------------------------------------------------------------------------

def test_train_refuses_without_data(
    fake_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["train", "--deck-name", "X", "--version", "0"])
    with pytest.raises(SystemExit) as exc:
        executor.train_cli()
    assert exc.value.code == 1


def test_train_accepts_h5_extension(
    fake_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_train_module: dict,
) -> None:
    """Regression test: H5Indexed accepts both .h5 and .hdf5, so the preflight
    should too."""
    _stage_data(fake_workspace, 0, "X", ext="h5")
    monkeypatch.setattr(sys, "argv", ["train", "--deck-name", "X", "--version", "0"])
    executor.train_cli()
    assert fake_train_module["called"] == 1


def test_train_filters_data_by_deck_prefix(
    fake_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shard for a different deck shouldn't satisfy the preflight for this deck."""
    _stage_data(fake_workspace, 0, "OtherDeck")
    monkeypatch.setattr(sys, "argv", ["train", "--deck-name", "X", "--version", "0"])
    with pytest.raises(SystemExit) as exc:
        executor.train_cli()
    assert exc.value.code == 1


# ---------------------------------------------------------------------------
# overwrite guard
# ---------------------------------------------------------------------------

def test_train_refuses_to_overwrite_existing_model(
    fake_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage_data(fake_workspace, 0, "X")
    _stage_model(fake_workspace, "X", 0)
    monkeypatch.setattr(sys, "argv", ["train", "--deck-name", "X", "--version", "0"])
    with pytest.raises(SystemExit) as exc:
        executor.train_cli()
    assert exc.value.code == 1


def test_train_force_allows_overwrite(
    fake_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_train_module: dict,
) -> None:
    _stage_data(fake_workspace, 0, "X")
    _stage_model(fake_workspace, "X", 0)
    monkeypatch.setattr(sys, "argv", ["train", "--deck-name", "X", "--version", "0", "--force"])
    executor.train_cli()
    assert fake_train_module["called"] == 1


def test_train_use_previous_model_exempts_from_overwrite_check(
    fake_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_train_module: dict,
) -> None:
    _stage_data(fake_workspace, 0, "X")
    _stage_model(fake_workspace, "X", 0)
    monkeypatch.setattr(sys, "argv", ["train", "--deck-name", "X", "--version", "0", "--use-previous-model"])
    executor.train_cli()
    assert fake_train_module["called"] == 1


# ---------------------------------------------------------------------------
# data preflight ordering — codex finding regression
# ---------------------------------------------------------------------------

def test_use_previous_model_does_not_stage_files_when_data_missing(
    fake_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the data preflight is going to fail, --use-previous-model must NOT
    leave a staged checkpoint behind in the target version's model dir."""
    _stage_model(fake_workspace, "X", 0)
    # No data at ver1.
    monkeypatch.setattr(sys, "argv", ["train", "--deck-name", "X", "--version", "1", "--use-previous-model"])
    with pytest.raises(SystemExit) as exc:
        executor.train_cli()
    assert exc.value.code == 1
    # Critical: nothing should have been written to models/X/ver1/.
    assert not (fake_workspace / "models" / "X" / "ver1").exists()


# ---------------------------------------------------------------------------
# auto-increment slot picking
# ---------------------------------------------------------------------------

def test_auto_increment_picks_zero_when_no_models(
    fake_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_train_module: dict,
) -> None:
    _stage_data(fake_workspace, 0, "X")
    monkeypatch.setattr(sys, "argv", ["train", "--deck-name", "X", "--auto-increment"])
    executor.train_cli()
    assert fake_train_module["called"] == 1


def test_auto_increment_skips_existing_models(
    fake_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_train_module: dict,
) -> None:
    _stage_model(fake_workspace, "X", 0)
    _stage_model(fake_workspace, "X", 1)
    _stage_data(fake_workspace, 2, "X")
    monkeypatch.setattr(sys, "argv", ["train", "--deck-name", "X", "--auto-increment"])
    executor.train_cli()
    assert fake_train_module["called"] == 1


# ---------------------------------------------------------------------------
# env-var-only resume flow (codex regression)
# ---------------------------------------------------------------------------

def test_env_var_use_previous_model_exempts_from_overwrite_check(
    fake_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_train_module: dict,
) -> None:
    """Regression: setting MAGEZERO_USE_PREVIOUS_MODEL=1 should be equivalent
    to passing --use-previous-model on the CLI for guard purposes."""
    _stage_data(fake_workspace, 0, "X")
    _stage_model(fake_workspace, "X", 0)
    monkeypatch.setenv("MAGEZERO_USE_PREVIOUS_MODEL", "1")
    monkeypatch.setattr(sys, "argv", ["train", "--deck-name", "X", "--version", "0"])
    executor.train_cli()
    assert fake_train_module["called"] == 1


def test_env_var_use_previous_model_triggers_walkback(
    fake_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_train_module: dict,
) -> None:
    """Regression: env-only --use-previous-model should also stage the previous
    version's checkpoint into the target dir."""
    _stage_model(fake_workspace, "X", 0)
    _stage_data(fake_workspace, 1, "X")
    monkeypatch.setenv("MAGEZERO_USE_PREVIOUS_MODEL", "1")
    monkeypatch.setattr(sys, "argv", ["train", "--deck-name", "X", "--version", "1"])
    executor.train_cli()
    assert fake_train_module["called"] == 1
    staged = fake_workspace / "models" / "X" / "ver1" / "model.pt"
    assert staged.exists()
    assert staged.read_bytes() == b"weights"
