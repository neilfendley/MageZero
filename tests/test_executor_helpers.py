"""Unit tests for the pure-ish helpers in `magezero.executor`."""
from __future__ import annotations

from pathlib import Path

import pytest

from magezero import executor


# ---------------------------------------------------------------------------
# _load_krenko_server_endpoints
# ---------------------------------------------------------------------------

def test_load_endpoints_happy_path(tmp_path: Path) -> None:
    cfg = tmp_path / "krenko_config.yml"
    cfg.write_text(
        "server:\n"
        "  host: 10.0.0.5\n"
        "  port: 60000\n"
        "  opponent_port: 60001\n"
    )
    host, port, opp = executor._load_krenko_server_endpoints(cfg)
    assert host == "10.0.0.5"
    assert port == 60000
    assert opp == 60001


def test_load_endpoints_localhost_normalized(tmp_path: Path) -> None:
    cfg = tmp_path / "k.yml"
    cfg.write_text("server:\n  host: localhost\n  port: 50052\n  opponent_port: 50053\n")
    host, _, _ = executor._load_krenko_server_endpoints(cfg)
    assert host == "127.0.0.1"


def test_load_endpoints_missing_file_returns_defaults(tmp_path: Path) -> None:
    host, port, opp = executor._load_krenko_server_endpoints(tmp_path / "nope.yml")
    assert (host, port, opp) == ("127.0.0.1", 50052, 50053)


def test_load_endpoints_malformed_yaml_returns_defaults(tmp_path: Path, caplog) -> None:
    cfg = tmp_path / "broken.yml"
    cfg.write_text("server:\n  host: [unclosed\n")
    host, port, opp = executor._load_krenko_server_endpoints(cfg)
    assert (host, port, opp) == ("127.0.0.1", 50052, 50053)


def test_load_endpoints_missing_keys_uses_defaults_per_field(tmp_path: Path) -> None:
    cfg = tmp_path / "partial.yml"
    cfg.write_text("server:\n  port: 9999\n")  # only port set
    host, port, opp = executor._load_krenko_server_endpoints(cfg)
    assert host == "127.0.0.1"
    assert port == 9999
    assert opp == 50053


def test_load_endpoints_no_server_section(tmp_path: Path) -> None:
    cfg = tmp_path / "no_server.yml"
    cfg.write_text("training:\n  games: 5\n")
    host, port, opp = executor._load_krenko_server_endpoints(cfg)
    assert (host, port, opp) == ("127.0.0.1", 50052, 50053)


# ---------------------------------------------------------------------------
# _latest_model_below
# ---------------------------------------------------------------------------

def _stage_model(magezero_root: Path, deck: str, version: int) -> Path:
    p = magezero_root / "models" / deck / f"ver{version}" / "model.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    return p


def test_latest_model_below_no_models_returns_none(fake_workspace: Path) -> None:
    assert executor._latest_model_below("X", 5) is None


def test_latest_model_below_returns_immediately_below(fake_workspace: Path) -> None:
    _stage_model(fake_workspace, "X", 0)
    _stage_model(fake_workspace, "X", 1)
    result = executor._latest_model_below("X", 2)
    assert result is not None
    assert result.parent.name == "ver1"


def test_latest_model_below_tolerates_gaps(fake_workspace: Path) -> None:
    _stage_model(fake_workspace, "X", 0)
    _stage_model(fake_workspace, "X", 3)  # gap at 1, 2
    result = executor._latest_model_below("X", 5)
    assert result is not None
    assert result.parent.name == "ver3"


def test_latest_model_below_version_zero_returns_none(fake_workspace: Path) -> None:
    _stage_model(fake_workspace, "X", 0)  # exists, but we ask for "below 0"
    assert executor._latest_model_below("X", 0) is None


def test_latest_model_below_filters_by_deck(fake_workspace: Path) -> None:
    _stage_model(fake_workspace, "OtherDeck", 5)
    assert executor._latest_model_below("X", 10) is None


# ---------------------------------------------------------------------------
# gather_hdf5_files
# ---------------------------------------------------------------------------

def test_gather_hdf5_files_nonexistent_root(tmp_path: Path) -> None:
    assert executor.gather_hdf5_files(tmp_path / "nope") == set()


def test_gather_hdf5_files_recursive(tmp_path: Path) -> None:
    (tmp_path / "a" / "b").mkdir(parents=True)
    f1 = tmp_path / "x.hdf5"
    f2 = tmp_path / "a" / "y.hdf5"
    f3 = tmp_path / "a" / "b" / "z.hdf5"
    for f in (f1, f2, f3):
        f.write_bytes(b"")
    found = executor.gather_hdf5_files(tmp_path)
    assert found == {f1.resolve(), f2.resolve(), f3.resolve()}


def test_gather_hdf5_files_accepts_both_h5_and_hdf5(tmp_path: Path) -> None:
    """H5Indexed accepts both .h5 and .hdf5, so the auto-increment / overwrite
    guards in executor.py have to too — otherwise a slot containing only .h5
    shards looks empty."""
    (tmp_path / "data.txt").write_bytes(b"")
    short_ext = tmp_path / "data.h5"
    long_ext = tmp_path / "data.hdf5"
    short_ext.write_bytes(b"")
    long_ext.write_bytes(b"")
    assert executor.gather_hdf5_files(tmp_path) == {short_ext.resolve(), long_ext.resolve()}


def test_gather_hdf5_files_multiple_roots(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    fa = a / "x.hdf5"
    fb = b / "y.hdf5"
    fa.write_bytes(b"")
    fb.write_bytes(b"")
    assert executor.gather_hdf5_files(a, b) == {fa.resolve(), fb.resolve()}
