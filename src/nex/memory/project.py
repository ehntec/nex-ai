"""Project memory manager for .nex/memory.md.

The project memory file is a human-readable markdown document that stores
project context (overview, tech stack, architecture, conventions). It is
loaded into the system prompt on every agent invocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from rich.console import Console

from nex.exceptions import NexMemoryError

console: Final[Console] = Console(stderr=True)

_MEMORY_TEMPLATE: Final[str] = """\
# Project Overview

{project_name}

## Tech Stack

{tech_stack}

## Architecture

<!-- Describe the high-level architecture of this project. -->

## Conventions

<!-- List coding conventions, naming rules, and style preferences. -->

## Notes

<!-- Anything else the agent should remember between sessions. -->
"""


@dataclass
class ProjectMemory:
    """Manages persistent project memory stored in .nex/memory.md.

    Attributes:
        project_dir: Root directory of the project containing the .nex folder.
    """

    project_dir: Path

    @property
    def memory_path(self) -> Path:
        """Return the absolute path to the memory file."""
        return self.project_dir / ".nex" / "memory.md"

    def exists(self) -> bool:
        """Check whether the memory file exists on disk."""
        return self.memory_path.is_file()

    def load(self) -> str:
        """Read and return the contents of memory.md.

        Returns:
            The full text of the memory file, or an empty string if not found.

        Raises:
            NexMemoryError: If the file exists but cannot be read.
        """
        if not self.exists():
            return ""
        try:
            return self.memory_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise NexMemoryError(
                f"Failed to read project memory at {self.memory_path}: {exc}"
            ) from exc

    def save(self, content: str) -> None:
        """Write content to memory.md, creating parent dirs if needed.

        Args:
            content: The full markdown text to persist.

        Raises:
            NexMemoryError: If the file cannot be written.
        """
        try:
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
            self.memory_path.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise NexMemoryError(
                f"Failed to save project memory at {self.memory_path}: {exc}"
            ) from exc

    def initialize(self, project_name: str, tech_stack: str = "") -> None:
        """Create an initial memory.md from the built-in template.

        Does not overwrite if already exists.

        Args:
            project_name: Human-readable project name.
            tech_stack: Optional tech stack description.

        Raises:
            NexMemoryError: If the file cannot be written.
        """
        if self.exists():
            console.print("[yellow]Memory file already exists — skipping.[/yellow]")
            return

        rendered = _MEMORY_TEMPLATE.format(
            project_name=project_name,
            tech_stack=tech_stack or "<!-- Add your tech stack here. -->",
        )
        self.save(rendered)
        console.print(f"[green]Initialized project memory[/green] for [bold]{project_name}[/bold]")

    def prune_section(
        self, section: str, max_lines: int = 30, keep_lines: int = 20
    ) -> None:
        """Trim a section to prevent unbounded growth.

        When the section's content lines exceed *max_lines*, the oldest
        entries are removed so that only *keep_lines* remain.

        Args:
            section: Section title without the ``#`` prefix (e.g. "Session Log").
            max_lines: Trigger pruning when content lines exceed this count.
            keep_lines: Number of newest content lines to retain after pruning.
        """
        if not self.exists():
            return

        text = self.load()
        lines = text.splitlines(keepends=True)

        # Locate section heading
        section_idx: int | None = None
        for idx, line in enumerate(lines):
            stripped = line.strip().lstrip("#").strip()
            if stripped.lower() == section.lower():
                section_idx = idx
                break

        if section_idx is None:
            return

        # Find the end of this section (next heading or EOF)
        section_end = len(lines)
        for idx in range(section_idx + 1, len(lines)):
            if lines[idx].lstrip().startswith("#"):
                section_end = idx
                break

        # Extract non-blank content lines within the section
        content_lines: list[int] = []
        for idx in range(section_idx + 1, section_end):
            if lines[idx].strip():
                content_lines.append(idx)

        if len(content_lines) <= max_lines:
            return

        # Remove the oldest lines, keep the newest keep_lines
        remove = set(content_lines[: len(content_lines) - keep_lines])
        pruned = [line for idx, line in enumerate(lines) if idx not in remove]
        self.save("".join(pruned))

    def append(self, section: str, content: str) -> None:
        """Append content under a given section heading.

        If the section is not found, a new section is appended at EOF.

        Args:
            section: Section title without the # prefix (e.g. "Notes").
            content: Markdown text to append.

        Raises:
            NexMemoryError: If the file cannot be read or written.
        """
        existing = self.load()
        lines = existing.splitlines(keepends=True)

        section_idx: int | None = None
        for idx, line in enumerate(lines):
            stripped = line.strip().lstrip("#").strip()
            if stripped.lower() == section.lower():
                section_idx = idx
                break

        if section_idx is not None:
            insert_idx = len(lines)
            for idx in range(section_idx + 1, len(lines)):
                if lines[idx].lstrip().startswith("#"):
                    insert_idx = idx
                    break

            while insert_idx > section_idx + 1 and lines[insert_idx - 1].strip() == "":
                insert_idx -= 1

            lines.insert(insert_idx, f"\n{content}\n")
        else:
            if existing and not existing.endswith("\n"):
                lines.append("\n")
            lines.append(f"\n## {section}\n\n{content}\n")

        self.save("".join(lines))
