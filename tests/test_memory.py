"""Tests for the project memory system."""

from __future__ import annotations

from pathlib import Path

import pytest

from nex.memory.project import ProjectMemory


@pytest.fixture
def memory(tmp_path: Path) -> ProjectMemory:
    (tmp_path / ".nex").mkdir()
    return ProjectMemory(project_dir=tmp_path)


class TestProjectMemory:
    def test_exists_false(self, memory: ProjectMemory) -> None:
        assert not memory.exists()

    def test_initialize(self, memory: ProjectMemory) -> None:
        memory.initialize("Test Project", "Python, FastAPI")
        assert memory.exists()
        content = memory.load()
        assert "Test Project" in content
        assert "Python, FastAPI" in content

    def test_initialize_no_overwrite(self, memory: ProjectMemory) -> None:
        memory.initialize("First")
        memory.initialize("Second")
        assert "First" in memory.load()

    def test_load_empty(self, memory: ProjectMemory) -> None:
        assert memory.load() == ""

    def test_save_and_load(self, memory: ProjectMemory) -> None:
        memory.save("# My Project\nHello world")
        assert memory.load() == "# My Project\nHello world"

    def test_append_existing_section(self, memory: ProjectMemory) -> None:
        memory.initialize("Test")
        memory.append("Notes", "Remember to add tests")
        content = memory.load()
        assert "Remember to add tests" in content

    def test_append_new_section(self, memory: ProjectMemory) -> None:
        memory.save("# Project\n")
        memory.append("Custom Section", "Some data")
        content = memory.load()
        assert "## Custom Section" in content
        assert "Some data" in content
