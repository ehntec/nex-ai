"""File search tool — ripgrep wrapper with Python fallback."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from nex.tools import ToolResult


async def search_files(
    pattern: str,
    path: str,
    project_dir: Path,
    max_results: int = 50,
) -> ToolResult:
    """Search files using ripgrep or fallback to Python grep.

    Args:
        pattern: Regex pattern to search for.
        path: Directory to search (relative to project root).
        project_dir: Project root directory.
        max_results: Maximum number of matches to return.

    Returns:
        ToolResult with matches in file:line:content format.
    """
    search_dir = (project_dir / path).resolve()
    project_root = project_dir.resolve()
    if not str(search_dir).startswith(str(project_root)):
        return ToolResult(success=False, output="", error="Path traversal blocked")

    if not search_dir.is_dir():
        return ToolResult(success=False, output="", error=f"Directory not found: {path}")

    # Try ripgrep first
    result = await _search_ripgrep(pattern, search_dir, max_results)
    if result is not None:
        return result

    # Fallback to Python search
    return _search_python(pattern, search_dir, project_root, max_results)


async def _search_ripgrep(pattern: str, search_dir: Path, max_results: int) -> ToolResult | None:
    """Attempt search using ripgrep (rg).

    Returns None if rg is not available.
    """
    try:
        process = await asyncio.create_subprocess_exec(
            "rg",
            "--no-heading",
            "--line-number",
            "--context",
            "2",
            "--max-count",
            str(max_results),
            "--color",
            "never",
            pattern,
            str(search_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=15)

        if process.returncode == 2:
            # rg error (bad pattern, etc.)
            return ToolResult(
                success=False,
                output="",
                error=f"Search error: {stderr.decode('utf-8', errors='replace')}",
            )

        output = stdout.decode("utf-8", errors="replace")
        if not output.strip():
            return ToolResult(success=True, output="No matches found.")

        return ToolResult(success=True, output=output)

    except FileNotFoundError:
        return None  # rg not installed, fall back
    except TimeoutError:
        return ToolResult(success=False, output="", error="Search timed out")


def _search_python(
    pattern: str, search_dir: Path, project_root: Path, max_results: int
) -> ToolResult:
    """Fallback search using Python regex."""
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        return ToolResult(success=False, output="", error=f"Invalid regex: {exc}")

    matches: list[str] = []
    source_extensions = {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".go",
        ".rs",
        ".java",
        ".rb",
        ".php",
        ".c",
        ".cpp",
        ".h",
        ".swift",
        ".kt",
        ".md",
        ".txt",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
    }

    for file_path in search_dir.rglob("*"):
        if len(matches) >= max_results:
            break
        if not file_path.is_file():
            continue
        if file_path.suffix not in source_extensions:
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            rel_path = file_path.relative_to(project_root)

            for i, line in enumerate(lines, start=1):
                if compiled.search(line):
                    # Include 2 lines of context
                    start = max(0, i - 3)
                    end = min(len(lines), i + 2)
                    context_lines = []
                    for j in range(start, end):
                        prefix = ">" if j == i - 1 else " "
                        context_lines.append(f"{prefix} {j + 1:4d} | {lines[j]}")
                    matches.append(f"{rel_path}:{i}\n" + "\n".join(context_lines))

                    if len(matches) >= max_results:
                        break
        except (OSError, UnicodeDecodeError):
            continue

    if not matches:
        return ToolResult(success=True, output="No matches found.")

    return ToolResult(success=True, output="\n\n".join(matches))
