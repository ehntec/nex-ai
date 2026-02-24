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


class TestIndex:
    def test_index_not_initialized(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["index"])
        assert result.exit_code == 1

    def test_index_no_source_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".nex").mkdir()
        result = runner.invoke(app, ["index"])
        assert result.exit_code == 0
        assert "no source files" in result.output.lower()

    def test_index_builds_successfully(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".nex").mkdir()
        (tmp_path / "main.py").write_text("def hello():\n    pass\n", encoding="utf-8")
        result = runner.invoke(app, ["index"])
        assert result.exit_code == 0
        assert (tmp_path / ".nex" / "index.json").is_file()

    def test_index_shows_summary(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".nex").mkdir()
        (tmp_path / "app.py").write_text(
            "def foo():\n    pass\n\ndef bar():\n    pass\n", encoding="utf-8"
        )
        result = runner.invoke(app, ["index"])
        assert result.exit_code == 0
        # Should display file/symbol counts
        assert "1" in result.output  # 1 file
        assert "2" in result.output  # 2 symbols


class TestChat:
    def test_chat_not_initialized(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["chat"])
        assert result.exit_code == 1

    def test_chat_no_api_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".nex").mkdir()
        (tmp_path / ".nex" / "memory.md").write_text("# Test\n", encoding="utf-8")
        # Ensure no API key is set
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = runner.invoke(app, ["chat"])
        assert result.exit_code == 1
        assert "api key" in result.output.lower()


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
