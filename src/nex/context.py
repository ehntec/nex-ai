"""Context assembly for API calls with priority-based token budgeting."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from nex.indexer.index import CodeIndex, IndexBuilder
from nex.memory.errors import ErrorPattern

console = Console(stderr=True)

_COMMENT_PREFIXES = ("#", "//", "/*", "*", '"""', "'''")

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

        Uses file-level TF-IDF relevance scoring from the index. Applies a
        relevance threshold to filter noise. Two-phase budget: 60% for full
        file content, 40% for signature-only summaries of remaining files.

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
            ranked_files = builder.search_files(task, index)
        except Exception:
            return []

        if not ranked_files:
            return []

        # Apply relevance threshold
        top_score = ranked_files[0][1]
        threshold = self._relevance_threshold(top_score)
        ranked_files = [(fp, sc) for fp, sc in ranked_files if sc >= threshold]

        # Two-phase budget: 60% full content, 40% signature summaries
        full_budget = int(budget_tokens * 0.60)
        sig_budget = budget_tokens - full_budget

        results: list[tuple[str, str]] = []
        tokens_used = 0
        remaining_files: list[str] = []

        # Phase 1: Full file content
        for file_path, _score in ranked_files:
            abs_path = self._project_dir / file_path
            if not abs_path.is_file():
                continue

            try:
                content = abs_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            file_tokens = self.estimate_tokens(content)
            if tokens_used + file_tokens <= full_budget:
                results.append((file_path, content))
                tokens_used += file_tokens
                continue

            # Try truncation (first 200 lines)
            lines = content.splitlines()[:200]
            truncated = "\n".join(lines) + "\n... (truncated)"
            trunc_tokens = self.estimate_tokens(truncated)
            if tokens_used + trunc_tokens <= full_budget:
                results.append((file_path, truncated))
                tokens_used += trunc_tokens
                continue

            # Defer to signature phase
            remaining_files.append(file_path)

        # Phase 2: Signature-only summaries for remaining files
        sig_tokens_used = 0
        for file_path in remaining_files:
            sig_text = self._extract_signatures(file_path, builder, index)
            if not sig_text:
                continue

            sig_tokens = self.estimate_tokens(sig_text)
            if sig_tokens_used + sig_tokens > sig_budget:
                break

            results.append((file_path, f"(signatures only)\n{sig_text}"))
            sig_tokens_used += sig_tokens

        return results

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count with code/comment-aware heuristic.

        Code lines average ~3.5 chars/token due to variable names and syntax.
        Comment/docstring lines average ~4.5 chars/token (more natural language).

        Args:
            text: Input text.

        Returns:
            Estimated number of tokens.
        """
        if not text:
            return 0

        total = 0
        for line in text.splitlines():
            stripped = line.lstrip()
            length = len(line)
            if not stripped:
                total += 1  # blank line ≈ 1 token
            elif stripped.startswith(_COMMENT_PREFIXES):
                total += max(1, int(length / 4.5))
            else:
                total += max(1, int(length / 3.5))
        return total

    @staticmethod
    def _relevance_threshold(top_score: float) -> float:
        """Compute the minimum relevance score to include a file.

        Files below this threshold are considered noise.

        Args:
            top_score: The highest file relevance score.

        Returns:
            Threshold value (at least 1.0).
        """
        return max(top_score * 0.10, 1.0)

    @staticmethod
    def _extract_signatures(file_path: str, builder: IndexBuilder, index: CodeIndex) -> str:
        """Extract function/class signatures for a file as a compact summary.

        Args:
            file_path: Relative file path.
            builder: IndexBuilder instance.
            index: Pre-loaded code index.

        Returns:
            Formatted string of signatures, or empty string if none found.
        """
        symbols = builder.get_file_symbols(file_path, index)
        if not symbols:
            return ""

        parts: list[str] = []
        for sym in symbols:
            if sym.kind == "import":
                continue
            parts.append(sym.signature)
            if sym.docstring:
                parts.append(f"    {sym.docstring}")
        return "\n".join(parts)
