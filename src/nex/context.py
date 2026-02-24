"""Context assembly for API calls with priority-based token budgeting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console

from nex.indexer.index import CodeIndex, IndexBuilder
from nex.memory.errors import ErrorPattern

console = Console(stderr=True)

_SYSTEM_PROMPT_TEMPLATE = """\
You are Nex, an AI coding agent that works on the user's codebase. You have access to tools \
for reading files, writing files, running commands, searching code, and listing directories.

## Project Context
{project_memory}

## Past Errors to Avoid
{error_patterns}

## Relevant Code
{relevant_code}

## Rules
- Always read existing code before modifying it. Match the project's style.
- Run tests after making changes. If tests fail, fix them before reporting success.
- Never execute destructive commands without user approval.
- When you make an architectural decision, explain why briefly.
- If you're unsure about something, ask the user rather than guessing.
- Keep changes minimal — don't refactor code that isn't related to the task.
"""


class ContextAssembler:
    """Assembles the context window for API calls with priority-based budgeting.

    Token budget: 150K of 200K window reserved for context, 50K for response.
    Priority order: system prompt -> project memory -> error patterns -> relevant code.
    """

    TOKEN_BUDGET = 150_000

    def __init__(self, project_dir: Path) -> None:
        """Initialize the context assembler.

        Args:
            project_dir: Project root directory.
        """
        self._project_dir = project_dir

    def build_system_prompt(
        self,
        project_memory: str,
        error_patterns: list[ErrorPattern],
        relevant_code: list[tuple[str, str]],
    ) -> str:
        """Assemble the system prompt from all context sources.

        Args:
            project_memory: Contents of .nex/memory.md.
            error_patterns: Recent similar error patterns.
            relevant_code: (file_path, content) pairs of relevant code.

        Returns:
            The assembled system prompt string.
        """
        # Format error patterns
        if error_patterns:
            error_text = "\n".join(
                f"- [{ep.error_type}] {ep.what_failed} -> Fix: {ep.what_fixed}"
                + (f" (in {ep.file_path})" if ep.file_path else "")
                for ep in error_patterns
            )
        else:
            error_text = "No relevant past errors found."

        # Format relevant code
        if relevant_code:
            code_sections = []
            for file_path, content in relevant_code:
                code_sections.append(f"### {file_path}\n```\n{content}\n```")
            code_text = "\n\n".join(code_sections)
        else:
            code_text = "No relevant code indexed yet."

        return _SYSTEM_PROMPT_TEMPLATE.format(
            project_memory=project_memory or "No project memory found. Run 'nex init' first.",
            error_patterns=error_text,
            relevant_code=code_text,
        )

    def select_relevant_code(
        self,
        task: str,
        index: CodeIndex | None,
        budget_tokens: int = 100_000,
    ) -> list[tuple[str, str]]:
        """Select relevant code files based on task description.

        Uses TF-IDF relevance from the index to select top files,
        filling until the token budget is reached.

        Args:
            task: The user's task description.
            index: Pre-loaded code index.
            budget_tokens: Maximum tokens worth of code to include.

        Returns:
            List of (file_path, content) pairs.
        """
        if index is None:
            return []

        builder = IndexBuilder(self._project_dir)
        try:
            symbols = builder.search_symbols(task, index)
        except Exception:  # noqa: BLE001
            return []

        # Collect unique files ordered by relevance
        seen_files: set[str] = set()
        ordered_files: list[str] = []
        for sym in symbols:
            if sym.file_path not in seen_files:
                seen_files.add(sym.file_path)
                ordered_files.append(sym.file_path)

        # Read files until budget is exhausted
        results: list[tuple[str, str]] = []
        tokens_used = 0

        for file_path in ordered_files:
            abs_path = self._project_dir / file_path
            if not abs_path.is_file():
                continue

            try:
                content = abs_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            file_tokens = self.estimate_tokens(content)
            if tokens_used + file_tokens > budget_tokens:
                # Try to include just the first 200 lines if too large
                lines = content.splitlines()[:200]
                truncated = "\n".join(lines) + "\n... (truncated)"
                trunc_tokens = self.estimate_tokens(truncated)
                if tokens_used + trunc_tokens <= budget_tokens:
                    results.append((file_path, truncated))
                    tokens_used += trunc_tokens
                break

            results.append((file_path, content))
            tokens_used += file_tokens

        return results

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count (~4 chars per token).

        Args:
            text: Input text.

        Returns:
            Estimated number of tokens.
        """
        return len(text) // 4
