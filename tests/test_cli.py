"""Tests for the CLI commands."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from nex.cli import app

runner = CliRunner()


class TestInit:
    def test_init_creates_nex_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        assert (tmp_path / ".nex").is_dir()
        assert (tmp_path / ".nex" / "memory.md").is_file()
        assert (tmp_path / ".nex" / "decisions.md").is_file()
        assert (tmp_path / ".nex" / "config.toml").is_file()

    def test_init_already_initialized(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".nex").mkdir()
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        assert "already" in result.output.lower()


class TestStatus:
    def test_status_not_initialized(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 1

    def test_status_initialized(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        runner.invoke(app, ["init"])
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0


class TestMemory:
    def test_memory_show(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        runner.invoke(app, ["init"])
        result = runner.invoke(app, ["memory", "show"])
        assert result.exit_code == 0

    def test_memory_not_initialized(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["memory", "show"])
        assert result.exit_code == 1
