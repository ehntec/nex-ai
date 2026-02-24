"""File operations tools — read_file and write_file."""

from __future__ import annotations

from pathlib import Path

from nex.exceptions import ToolError
from nex.tools import ToolResult

_MAX_FILE_SIZE = 1_048_576  # 1 MB


def _resolve_safe_path(path: str, project_dir: Path) -> Path:
    """Resolve a path relative to project_dir, preventing traversal.

    Args:
        path: User-supplied path (relative or absolute).
        project_dir: Project root directory.

    Returns:
        Resolved absolute path within project_dir.

    Raises:
        ToolError: If the resolved path escapes the project root.
    """
    resolved = (project_dir / path).resolve()
    project_root = project_dir.resolve()
    if not str(resolved).startswith(str(project_root)):
        raise ToolError(f"Path traversal blocked: {path} escapes project root")
    return resolved


async def read_file(path: str, project_dir: Path) -> ToolResult:
    """Read a file and return its content with line numbers.

    Args:
        path: File path relative to the project root.
        project_dir: Project root directory.

    Returns:
        ToolResult with the file content or error message.
    """
    try:
        resolved = _resolve_safe_path(path, project_dir)
    except ToolError as exc:
        return ToolResult(success=False, output="", error=str(exc))

    if not resolved.is_file():
        return ToolResult(success=False, output="", error=f"File not found: {path}")

    try:
        size = resolved.stat().st_size
        if size > _MAX_FILE_SIZE:
            return ToolResult(
                success=False,
                output="",
                error=f"File too large ({size} bytes, max {_MAX_FILE_SIZE})",
            )

        content = resolved.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
        numbered = "\n".join(f"{i + 1:4d} | {line}" for i, line in enumerate(lines))
        return ToolResult(success=True, output=numbered)
    except OSError as exc:
        return ToolResult(success=False, output="", error=f"Cannot read {path}: {exc}")


async def write_file(path: str, content: str, project_dir: Path) -> ToolResult:
    """Write content to a file, creating directories if needed.

    Args:
        path: File path relative to the project root.
        content: The content to write.
        project_dir: Project root directory.

    Returns:
        ToolResult with success message or error.
    """
    try:
        resolved = _resolve_safe_path(path, project_dir)
    except ToolError as exc:
        return ToolResult(success=False, output="", error=str(exc))

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return ToolResult(success=True, output=f"Successfully wrote {len(content)} bytes to {path}")
    except OSError as exc:
        return ToolResult(success=False, output="", error=f"Cannot write {path}: {exc}")
