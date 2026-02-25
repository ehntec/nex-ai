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

    def select_scoped_code(
        self,
        file_paths: list[str],
        task: str,
        index: CodeIndex | None,
        budget_tokens: int = 20_000,
    ) -> list[tuple[str, str]]:
        """Select code scoped to specific files, fitting within a token budget.

        Reads only the specified files. If they exceed the budget, truncates
        content or falls back to signature-only summaries. When file_paths is
        empty, delegates to select_relevant_code().

        Args:
            file_paths: Relative file paths to include.
            task: The subtask description (used for fallback ranking).
            index: Pre-loaded code index.
            budget_tokens: Maximum tokens for the returned code.

        Returns:
            List of (file_path, content) pairs.
        """
        if not file_paths:
            return self.select_relevant_code(task, index, budget_tokens)

        results: list[tuple[str, str]] = []
        tokens_used = 0

        for file_path in file_paths:
            abs_path = self._project_dir / file_path
            if not abs_path.is_file():
                continue

            try:
                content = abs_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            file_tokens = self.estimate_tokens(content)
            if tokens_used + file_tokens <= budget_tokens:
                results.append((file_path, content))
                tokens_used += file_tokens
                continue

            # Try truncation (first 200 lines)
            lines = content.splitlines()[:200]
            truncated = "\n".join(lines) + "\n... (truncated)"
            trunc_tokens = self.estimate_tokens(truncated)
            if tokens_used + trunc_tokens <= budget_tokens:
                results.append((file_path, truncated))
                tokens_used += trunc_tokens
                continue

            # Fall back to signatures if index available
            if index is not None:
                builder = IndexBuilder(self._project_dir)
                sig_text = self._extract_signatures(file_path, builder, index)
                if sig_text:
                    sig_tokens = self.estimate_tokens(sig_text)
                    if tokens_used + sig_tokens <= budget_tokens:
                        results.append((file_path, f"(signatures only)\n{sig_text}"))
                        tokens_used += sig_tokens

        return results

    def build_subtask_prompt(
        self,
        subtask_description: str,
        project_memory: str,
        error_patterns: list[ErrorPattern],
        relevant_code: list[tuple[str, str]],
        prior_context: str = "",
    ) -> str:
        """Build a compact system prompt scoped to a single subtask.

        Truncates memory to 50 lines and limits error patterns to top 2
        to keep the prompt small for rate-limited usage.

        Args:
            subtask_description: What this subtask should accomplish.
            project_memory: Contents of .nex/memory.md.
            error_patterns: Recent similar error patterns (top 2 used).
            relevant_code: (file_path, content) pairs scoped to this subtask.
            prior_context: Summary of prior subtask results.

        Returns:
            The assembled system prompt string.
        """
        # Truncate memory to 50 lines
        if project_memory:
            mem_lines = project_memory.splitlines()[:50]
            truncated_memory = "\n".join(mem_lines)
            if len(mem_lines) < len(project_memory.splitlines()):
                truncated_memory += "\n... (truncated)"
        else:
            truncated_memory = "No project memory found."

        # Limit to top 2 error patterns
        top_errors = error_patterns[:2]
        if top_errors:
            error_text = "\n".join(
                f"- [{ep.error_type}] {ep.what_failed} -> Fix: {ep.what_fixed}"
                + (f" (in {ep.file_path})" if ep.file_path else "")
                for ep in top_errors
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

        # Build prior context section
        prior_section = ""
        if prior_context:
            prior_section = f"\n## Prior Subtask Results\n{prior_context}\n"

        return (
            f"You are Nex, an AI coding agent working on a subtask.\n\n"
            f"## Current Subtask\n{subtask_description}\n\n"
            f"## Project Context\n{truncated_memory}\n\n"
            f"## Past Errors to Avoid\n{error_text}\n\n"
            f"## Relevant Code\n{code_text}\n"
            f"{prior_section}\n"
            f"## Rules\n"
            f"- Always read existing code before modifying it. Match the project's style.\n"
            f"- Run tests after making changes. If tests fail, fix them before reporting success.\n"
            f"- Never execute destructive commands without user approval.\n"
            f"- Keep changes minimal — don't refactor code that isn't related to the task.\n"
        )

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
