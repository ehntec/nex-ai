"""Nex AI exception hierarchy.

All exceptions inherit from NexError so callers can catch the base
class when they want to handle any Nex-specific failure uniformly.
"""

from __future__ import annotations


class NexError(Exception):
    """Base exception for all Nex errors."""


class ConfigError(NexError):
    """Configuration-related errors (missing API key, invalid config, etc.)."""


class APIError(NexError):
    """Errors communicating with the Anthropic API."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ToolError(NexError):
    """Errors during tool execution (file ops, shell commands, etc.)."""


class SafetyError(NexError):
    """Safety layer violations (destructive commands, path traversal, etc.)."""


class IndexerError(NexError):
    """Errors during codebase indexing or AST parsing."""


class NexMemoryError(NexError):
    """Errors in the memory system (project memory, error DB, decision log)."""
