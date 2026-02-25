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

    def test_prune_section_under_limit(self, memory: ProjectMemory) -> None:
        """No pruning when section has fewer lines than max_lines."""
        lines = [f"- Entry {i}" for i in range(10)]
        memory.save("# Project\n\n## Session Log\n\n" + "\n".join(lines) + "\n")
        memory.prune_section("Session Log", max_lines=30, keep_lines=20)
        content = memory.load()
        # All 10 entries should still be present
        for i in range(10):
            assert f"Entry {i}" in content

    def test_prune_section_over_limit(self, memory: ProjectMemory) -> None:
        """Oldest entries removed when section exceeds max_lines."""
        lines = [f"- entry_{i:03d}" for i in range(35)]
        memory.save(
            "# Project\n\n## Session Log\n\n"
            + "\n".join(lines)
            + "\n\n## Other Section\n\nKeep this.\n"
        )
        memory.prune_section("Session Log", max_lines=30, keep_lines=20)
        content = memory.load()
        # Oldest entries (000-014) should be gone, newest (015-034) kept
        for i in range(15):
            assert f"entry_{i:03d}" not in content
        for i in range(15, 35):
            assert f"entry_{i:03d}" in content
        # Other section preserved
        assert "## Other Section" in content
        assert "Keep this." in content

    def test_prune_section_nonexistent(self, memory: ProjectMemory) -> None:
        """Pruning a missing section should not raise."""
        memory.save("# Project\n")
        memory.prune_section("Nonexistent Section")
        assert "# Project" in memory.load()

    def test_prune_section_no_file(self, memory: ProjectMemory) -> None:
        """Pruning when memory file does not exist should not raise."""
        memory.prune_section("Session Log")
