"""Nex tool system — the 5 tools available to the agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["TOOL_DEFINITIONS", "ToolResult"]


@dataclass
class ToolResult:
    """Result from a tool execution.

    Attributes:
        success: Whether the tool executed successfully.
        output: The tool's output text.
        error: Error message if the tool failed.
    """

    success: bool
    output: str
    error: str | None = None


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read the contents of a file. Returns the file content with line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the project root.",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file. Creates parent directories if needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the project root.",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file.",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_command",
        "description": "Execute a shell command and return its output. Subject to safety checks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                }
            },
            "required": ["command"],
        },
    },
    {
        "name": "search_files",
        "description": (
            "Search for a pattern across files using ripgrep or fallback grep. "
            "Returns matches with file path, line number, and surrounding context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: project root).",
                    "default": ".",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "list_directory",
        "description": "List the contents of a directory recursively up to a given depth.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path relative to the project root.",
                    "default": ".",
                },
                "depth": {
                    "type": "integer",
                    "description": "Maximum recursion depth (default: 3).",
                    "default": 3,
                },
            },
        },
    },
]
