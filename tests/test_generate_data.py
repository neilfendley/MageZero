"""Tests for `magezero.executor.generate_data` — argument parsing, mode
auto-detection, and auto-increment slot picking. The KrenkoMain subprocess
and inference server lifecycle are mocked away.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from magezero import executor

from tests.conftest import stage_model, stage_training_shard


@pytest.fixture
def mock_subprocess(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Stub out everything that would touch the JVM or the network so the
    `generate_data` flow runs end-to-end without side effects.

    Returns a dict the test can inspect to assert which mocks were called.
    """
    state = {
        "run_command_calls": [],
        "start_server_calls": [],
        "wait_for_http_calls": [],
        "update_model_calls": [],
    }

    def fake_run_command(cmd, cwd=None, env=None):
        state["run_command_calls"].append(cmd)

    class FakeProc:
        def terminate(self): pass
        def wait(self, timeout=None): pass
        def kill(self): pass

    def fake_start_server(args, env):
        state["start_server_calls"].append((args, env))
        return FakeProc()

    def fake_wait_for_http(url, timeout_s=60):
        state["wait_for_http_calls"].append(url)

    def fake_update_model(url, path):
        state["update_model_calls"].append((url, path))

    monkeypatch.setattr(executor, "run_command", fake_run_command)
    monkeypatch.setattr(executor, "start_server", fake_start_server)
    monkeypatch.setattr(executor, "wait_for_http", fake_wait_for_http)
    monkeypatch.setattr(executor, "update_model_weights", fake_update_model)
    return state


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------

def test_auto_increment_and_version_are_mutually_exclusive(
    fake_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys, "argv",
        ["generate-data", "--deck-path", "decks/X.dck", "--auto-increment", "--version", "2"],
    )
    with pytest.raises(SystemExit) as exc:
        executor.generate_data()
    assert exc.value.code == 1


# ---------------------------------------------------------------------------
# auto-increment slot picking
# ---------------------------------------------------------------------------

def test_auto_increment_picks_zero_when_no_data(
    fake_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_subprocess: dict,
) -> None:
    monkeypatch.setattr(sys, "argv", ["generate-data", "--deck-path", "decks/X.dck", "--auto-increment"])
    executor.generate_data()
    # KrenkoMain "ran" once.
    assert len(mock_subprocess["run_command_calls"]) == 1
    # No model exists, so no inference server should have started.
    assert len(mock_subprocess["start_server_calls"]) == 0


def test_auto_increment_skips_populated_versions(
    fake_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_subprocess: dict,
) -> None:
    stage_training_shard(fake_workspace, 0)
    stage_training_shard(fake_workspace, 1)
    monkeypatch.setattr(sys, "argv", ["generate-data", "--deck-path", "decks/X.dck", "--auto-increment"])
    executor.generate_data()
    # The picked version should be 2 — we can verify by inspecting the krenko
    # command's --version arg.
    cmd = mock_subprocess["run_command_calls"][0]
    flat = " ".join(cmd)
    assert "--version 2" in flat


def test_auto_increment_treats_h5_only_slot_as_populated(
    fake_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_subprocess: dict,
) -> None:
    """Regression: an existing slot containing only `.h5` shards (not `.hdf5`)
    used to look empty to the auto-increment search. Now both extensions count."""
    d = fake_workspace / "data" / "ver0" / "training"
    d.mkdir(parents=True)
    (d / "X_v0.h5").write_bytes(b"")
    monkeypatch.setattr(sys, "argv", ["generate-data", "--deck-path", "decks/X.dck", "--auto-increment"])
    executor.generate_data()
    cmd = mock_subprocess["run_command_calls"][0]
    flat = " ".join(cmd)
    assert "--version 1" in flat


# ---------------------------------------------------------------------------
# online vs offline mode auto-detection
# ---------------------------------------------------------------------------

def test_offline_when_no_earlier_model_exists(
    fake_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_subprocess: dict,
) -> None:
    monkeypatch.setattr(
        sys, "argv",
        ["generate-data", "--deck-path", "decks/X.dck", "--version", "0"],
    )
    executor.generate_data()
    assert len(mock_subprocess["start_server_calls"]) == 0
    assert len(mock_subprocess["update_model_calls"]) == 0


def test_online_when_earlier_model_exists(
    fake_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_subprocess: dict,
) -> None:
    stage_model(fake_workspace, "X", 0)
    monkeypatch.setattr(
        sys, "argv",
        ["generate-data", "--deck-path", "decks/X.dck", "--version", "1"],
    )
    executor.generate_data()
    assert len(mock_subprocess["start_server_calls"]) == 1
    assert len(mock_subprocess["update_model_calls"]) == 1
    # The loaded model should be the ver0 path.
    _, model_path = mock_subprocess["update_model_calls"][0]
    assert "ver0" in model_path


def test_offline_flag_forces_offline_even_with_model(
    fake_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_subprocess: dict,
) -> None:
    stage_model(fake_workspace, "X", 0)
    monkeypatch.setattr(
        sys, "argv",
        ["generate-data", "--deck-path", "decks/X.dck", "--version", "1", "--offline"],
    )
    executor.generate_data()
    assert len(mock_subprocess["start_server_calls"]) == 0


def test_asymmetric_starts_two_servers_when_both_models_exist(
    fake_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_subprocess: dict,
) -> None:
    stage_model(fake_workspace, "X", 0)
    stage_model(fake_workspace, "Y", 0)
    monkeypatch.setattr(
        sys, "argv",
        [
            "generate-data",
            "--deck-path", "decks/X.dck",
            "--opponent-deck", "decks/Y.dck",
            "--version", "1",
        ],
    )
    executor.generate_data()
    assert len(mock_subprocess["start_server_calls"]) == 2
