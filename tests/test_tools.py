"""Tests for tool execution."""

from __future__ import annotations

from pathlib import Path

import pytest

from nex.tools.file_ops import read_file, write_file
from nex.tools.shell import run_command


class TestReadFile:
    @pytest.mark.asyncio
    async def test_read_existing_file(self, tmp_path: Path) -> None:
        test_file = tmp_path / "hello.txt"
        test_file.write_text("line1\nline2\nline3", encoding="utf-8")

        result = await read_file("hello.txt", tmp_path)
        assert result.success
        assert "line1" in result.output
        assert "line2" in result.output

    @pytest.mark.asyncio
    async def test_read_nonexistent(self, tmp_path: Path) -> None:
        result = await read_file("nope.txt", tmp_path)
        assert not result.success
        assert "not found" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self, tmp_path: Path) -> None:
        result = await read_file("../../etc/passwd", tmp_path)
        assert not result.success
        assert "traversal" in (result.error or "").lower()


class TestWriteFile:
    @pytest.mark.asyncio
    async def test_write_new_file(self, tmp_path: Path) -> None:
        result = await write_file("output.txt", "hello world", tmp_path)
        assert result.success
        assert (tmp_path / "output.txt").read_text() == "hello world"

    @pytest.mark.asyncio
    async def test_write_creates_dirs(self, tmp_path: Path) -> None:
        result = await write_file("sub/dir/file.txt", "content", tmp_path)
        assert result.success
        assert (tmp_path / "sub" / "dir" / "file.txt").exists()

    @pytest.mark.asyncio
    async def test_write_path_traversal(self, tmp_path: Path) -> None:
        result = await write_file("../../escape.txt", "bad", tmp_path)
        assert not result.success


class TestRunCommand:
    @pytest.mark.asyncio
    async def test_echo(self, tmp_path: Path) -> None:
        result = await run_command("echo hello", tmp_path)
        assert result.success
        assert "hello" in result.output

    @pytest.mark.asyncio
    async def test_failing_command(self, tmp_path: Path) -> None:
        result = await run_command("exit 1", tmp_path)
        assert not result.success

    @pytest.mark.asyncio
    async def test_timeout(self, tmp_path: Path) -> None:
        result = await run_command("sleep 10", tmp_path, timeout=1)
        assert not result.success
        assert "timed out" in (result.error or "").lower()
