"""Tests for test runner detection and execution."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from nex.test_runner import TestRunner
from nex.tools import ToolResult


class TestDetection:
    """Tests for TestRunner.detect() — framework auto-detection."""

    def test_detect_python_pyproject(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text(
            '[tool.pytest.ini_options]\nminversion = "7.0"\n',
            encoding="utf-8",
        )
        runner = TestRunner(tmp_path)
        assert runner.detect() == "python -m pytest"

    def test_detect_python_test_files(self, tmp_path: Path) -> None:
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_app.py").write_text("def test_one(): pass\n", encoding="utf-8")
        runner = TestRunner(tmp_path)
        assert runner.detect() == "python -m pytest"

    def test_detect_javascript_npm(self, tmp_path: Path) -> None:
        pkg = {"name": "my-app", "scripts": {"test": "jest"}}
        (tmp_path / "package.json").write_text(json.dumps(pkg), encoding="utf-8")
        runner = TestRunner(tmp_path)
        assert runner.detect() == "npm test"

    def test_detect_javascript_no_real_script(self, tmp_path: Path) -> None:
        pkg = {
            "name": "my-app",
            "scripts": {"test": 'echo "Error: no test specified" && exit 1'},
        }
        (tmp_path / "package.json").write_text(json.dumps(pkg), encoding="utf-8")
        runner = TestRunner(tmp_path)
        assert runner.detect() is None

    def test_detect_go(self, tmp_path: Path) -> None:
        (tmp_path / "go.mod").write_text("module example.com/app\n", encoding="utf-8")
        (tmp_path / "main_test.go").write_text("package main\n", encoding="utf-8")
        runner = TestRunner(tmp_path)
        assert runner.detect() == "go test ./..."

    def test_detect_rust(self, tmp_path: Path) -> None:
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "app"\n', encoding="utf-8")
        runner = TestRunner(tmp_path)
        assert runner.detect() == "cargo test"

    def test_detect_nothing(self, tmp_path: Path) -> None:
        runner = TestRunner(tmp_path)
        assert runner.detect() is None


class TestRun:
    """Tests for TestRunner.run() — test execution."""

    @pytest.mark.asyncio
    async def test_run_success(self, tmp_path: Path) -> None:
        mock_result = ToolResult(success=True, output="3 passed in 0.5s")
        with patch("nex.test_runner.run_command", new_callable=AsyncMock, return_value=mock_result):
            runner = TestRunner(tmp_path)
            result = await runner.run("python -m pytest")

        assert result.success is True
        assert "3 passed" in result.output
        assert result.command == "python -m pytest"
        assert result.framework == "pytest"

    @pytest.mark.asyncio
    async def test_run_failure(self, tmp_path: Path) -> None:
        mock_result = ToolResult(
            success=False,
            output="FAILED test_app.py::test_one",
            error="Exit code 1",
        )
        with patch("nex.test_runner.run_command", new_callable=AsyncMock, return_value=mock_result):
            runner = TestRunner(tmp_path)
            result = await runner.run("python -m pytest")

        assert result.success is False
        assert "FAILED" in result.output
        assert result.framework == "pytest"

    @pytest.mark.asyncio
    async def test_run_no_runner(self, tmp_path: Path) -> None:
        runner = TestRunner(tmp_path)
        result = await runner.run()

        assert result.success is True
        assert result.output == "No test runner detected"
        assert result.command == ""
        assert result.framework == ""
