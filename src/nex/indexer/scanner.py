"""File discovery engine that walks a project tree respecting .gitignore."""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from rich.console import Console

from nex.exceptions import IndexerError

console = Console(stderr=True)

_MAX_FILE_SIZE = 1_048_576  # 1 MB

_ALWAYS_SKIP: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        "node_modules",
        ".nex",
        ".venv",
        "venv",
        ".env",
        "env",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".eggs",
    }
)


@dataclass(frozen=True, slots=True)
class FileInfo:
    """Metadata for a single discovered source file.

    Attributes:
        path: File path relative to the project root.
        language: Detected programming language identifier.
        size: File size in bytes.
    """

    path: Path
    language: str
    size: int


class FileScanner:
    """Discovers source files in a project, respecting .gitignore.

    Usage::

        scanner = FileScanner(Path("/my/project"))
        files = scanner.scan()
    """

    LANGUAGE_MAP: ClassVar[dict[str, str]] = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".jsx": "javascript",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".rb": "ruby",
        ".php": "php",
        ".cs": "csharp",
        ".cpp": "cpp",
        ".c": "c",
        ".swift": "swift",
        ".kt": "kotlin",
    }

    def __init__(self, project_dir: Path) -> None:
        """Initialize the scanner.

        Args:
            project_dir: Absolute path to the project root.

        Raises:
            IndexerError: If project_dir does not exist.
        """
        self._project_dir = project_dir.resolve()
        if not self._project_dir.is_dir():
            raise IndexerError(f"Project directory does not exist: {self._project_dir}")
        self._gitignore_patterns = self._load_gitignore()

    def scan(self) -> list[FileInfo]:
        """Walk the project tree and return discovered source files.

        Returns:
            A list of FileInfo instances sorted by path.
        """
        results: list[FileInfo] = []
        try:
            for dirpath_str, dirnames, filenames in os.walk(self._project_dir, topdown=True):
                dirpath = Path(dirpath_str)

                dirnames[:] = [
                    d
                    for d in dirnames
                    if not self._should_skip_dir(d) and not self._is_ignored(dirpath / d)
                ]

                for fname in filenames:
                    full = dirpath / fname
                    lang = self._detect_language(full)
                    if lang is None:
                        continue

                    try:
                        size = full.stat().st_size
                    except OSError:
                        continue

                    if size > _MAX_FILE_SIZE:
                        continue

                    rel = full.relative_to(self._project_dir)
                    if self._is_ignored(full):
                        continue

                    results.append(FileInfo(path=rel, language=lang, size=size))
        except OSError as exc:
            raise IndexerError(f"Failed to scan project directory: {exc}") from exc

        results.sort(key=lambda fi: fi.path)
        console.print(f"[green]Scanner[/green] found [bold]{len(results)}[/bold] source files")
        return results

    def _load_gitignore(self) -> list[str]:
        """Load .gitignore patterns, returning empty list if missing."""
        gitignore_path = self._project_dir / ".gitignore"
        if not gitignore_path.is_file():
            return []

        lines: list[str] = []
        text = gitignore_path.read_text(encoding="utf-8", errors="replace")
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
        return lines

    def _is_ignored(self, path: Path) -> bool:
        """Check whether path matches any .gitignore pattern."""
        if not self._gitignore_patterns:
            return False

        try:
            rel = str(path.relative_to(self._project_dir)).replace("\\", "/")
        except ValueError:
            return False

        basename = rel.rsplit("/", 1)[-1]
        for pattern in self._gitignore_patterns:
            negated = pattern.startswith("!")
            pat = pattern.lstrip("!")
            pat = pat.rstrip("/")

            if "/" in pat:
                if fnmatch.fnmatch(rel, pat):
                    return not negated
            elif fnmatch.fnmatch(basename, pat):
                return not negated
        return False

    @staticmethod
    def _should_skip_dir(dirname: str) -> bool:
        """Return True if dirname should always be skipped."""
        if dirname.startswith("."):
            return True
        if dirname in _ALWAYS_SKIP:
            return True
        if dirname.endswith(".egg-info"):
            return True
        return False

    def _detect_language(self, path: Path) -> str | None:
        """Detect programming language from file extension."""
        return self.LANGUAGE_MAP.get(path.suffix.lower())
