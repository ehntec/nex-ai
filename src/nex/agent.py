"""Core agent loop — the heart of Nex AI.

Follows the agentic REPL pattern:
1. Assemble context (system prompt + memory + code + errors)
2. Call Claude API with tool definitions
3. Parse response: if tool_use blocks, execute the tools
4. Feed tool results back to Claude
5. Repeat until text-only response or max iterations
6. Log errors encountered + fixes applied
7. If code modified: run tests, show diff, ask to commit
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax

from nex.api_client import AnthropicClient, RateLimiter
from nex.config import NexConfig
from nex.context import ContextAssembler
from nex.exceptions import NexError, SafetyError, ToolError
from nex.memory.errors import ErrorPatternDB
from nex.memory.project import ProjectMemory
from nex.planner import Planner
from nex.safety import SafetyLayer
from nex.test_runner import TestRunner
from nex.tools import TOOL_DEFINITIONS, ToolResult
from nex.tools.file_ops import read_file, write_file
from nex.tools.git_ops import GitOperations
from nex.tools.search import search_files
from nex.tools.shell import run_command

console = Console()


@dataclass
class AgentConfig:
    """Configuration for a single agent run.

    Attributes:
        project_dir: Project root directory.
        task: The user's task description.
        dry_run: If True, show what would happen without executing.
        max_iterations: Maximum number of tool call iterations.
    """

    project_dir: Path
    task: str
    dry_run: bool = False
    max_iterations: int = 25


@dataclass
class SubtaskResult:
    """Result of a single subtask execution.

    Attributes:
        text: The final text response from the subtask.
        files_touched: Paths of files written during the subtask.
        iterations: Number of agent iterations used.
    """

    text: str
    files_touched: list[str] = field(default_factory=list)
    iterations: int = 0


_MID_SUBTASK_MEMORY_THRESHOLD: int = 8

_MEMORY_SUMMARY_PROMPT: str = """\
You are summarizing what a coding agent just did. Given the subtask and result, \
write a 1-2 sentence summary. Include what was accomplished and key files modified. \
Be extremely concise. No markdown. Just 1-2 plain sentences.\
"""


async def execute_tool(
    name: str,
    tool_input: dict[str, Any],
    project_dir: Path,
    safety: SafetyLayer,
    dry_run: bool = False,
) -> tuple[ToolResult, bool]:
    """Route a tool call to the correct handler.

    Shared between Agent (single-task) and ChatSession (interactive REPL).

    Args:
        name: Name of the tool to execute.
        tool_input: Tool input parameters.
        project_dir: Project root directory.
        safety: Safety layer for command approval.
        dry_run: If True, skip destructive operations.

    Returns:
        Tuple of (ToolResult, files_modified_flag).
    """
    files_modified = False
    try:
        if name == "read_file":
            return await read_file(tool_input["path"], project_dir), False

        if name == "write_file":
            path = tool_input["path"]
            check = safety.check_file_write(path, project_dir)
            if not check.is_safe:
                if check.requires_approval:
                    approved = await safety.request_approval(f"Write to {path}", check.reason or "")
                    if not approved:
                        return ToolResult(success=False, output="", error="Write denied"), False
                else:
                    reason = check.reason or "Write blocked"
                    return ToolResult(success=False, output="", error=reason), False

            result = await write_file(path, tool_input["content"], project_dir)
            if result.success:
                files_modified = True
            return result, files_modified

        if name == "run_command":
            try:
                await safety.guard_command(tool_input["command"])
            except SafetyError as exc:
                return ToolResult(success=False, output="", error=str(exc)), False

            if dry_run:
                return (
                    ToolResult(
                        success=True,
                        output=f"[dry-run] Would execute: {tool_input['command']}",
                    ),
                    False,
                )

            return await run_command(tool_input["command"], project_dir), False

        if name == "search_files":
            result = await search_files(
                tool_input["pattern"],
                tool_input.get("path", "."),
                project_dir,
            )
            return result, False

        if name == "list_directory":
            result = await list_directory(
                tool_input.get("path", "."),
                tool_input.get("depth", 3),
                project_dir,
            )
            return result, False

        return ToolResult(success=False, output="", error=f"Unknown tool: {name}"), False

    except Exception as exc:
        return ToolResult(success=False, output="", error=f"Tool error: {exc}"), False


async def list_directory(path: str, depth: int, project_dir: Path) -> ToolResult:
    """List directory contents recursively.

    Args:
        path: Directory path relative to project root.
        depth: Maximum recursion depth.
        project_dir: Project root directory.

    Returns:
        ToolResult with the directory tree listing.
    """
    from nex.tools.file_ops import _resolve_safe_path

    try:
        resolved = _resolve_safe_path(path, project_dir)
    except ToolError as exc:
        return ToolResult(success=False, output="", error=str(exc))

    if not resolved.is_dir():
        return ToolResult(success=False, output="", error=f"Not a directory: {path}")

    lines: list[str] = []
    _walk_tree(resolved, "", depth, lines)
    return ToolResult(success=True, output="\n".join(lines))


def _walk_tree(directory: Path, prefix: str, depth: int, lines: list[str]) -> None:
    """Recursively build a directory tree listing.

    Args:
        directory: Current directory to list.
        prefix: Line prefix for indentation.
        depth: Remaining recursion depth.
        lines: Accumulator for output lines.
    """
    if depth <= 0:
        return

    try:
        entries = sorted(directory.iterdir(), key=lambda e: (not e.is_dir(), e.name))
    except PermissionError:
        return

    for i, entry in enumerate(entries):
        if entry.name.startswith(".") or entry.name in ("node_modules", "__pycache__", ".nex"):
            continue

        is_last = i == len(entries) - 1
        connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
        lines.append(f"{prefix}{connector}{entry.name}")

        if entry.is_dir():
            extension = "    " if is_last else "\u2502   "
            _walk_tree(entry, prefix + extension, depth - 1, lines)


class Agent:
    """Core agent loop — orchestrates tools, API calls, and memory."""

    def __init__(
        self,
        config: AgentConfig,
        api_client: AnthropicClient,
        safety: SafetyLayer,
        nex_config: NexConfig | None = None,
    ) -> None:
        """Initialize the agent.

        Args:
            config: Agent run configuration.
            api_client: Anthropic API client.
            safety: Safety layer for command approval.
            nex_config: Full Nex configuration (for test runner settings).
        """
        self._config = config
        self._nex_config = nex_config
        self._client = api_client
        self._safety = safety
        self._project_dir = config.project_dir
        self._messages: list[dict[str, Any]] = []
        self._iteration = 0
        self._files_modified = False

    async def run(self) -> str:
        """Execute the agent loop and return the final response.

        Always decomposes the task into subtasks. Falls back to the
        standard single-loop execution only if planning fails.

        Returns:
            The agent's final text response.
        """
        rate_limit = 0
        if self._nex_config:
            rate_limit = self._nex_config.token_rate_limit

        try:
            return await self._run_with_subtasks(rate_limit)
        except Exception as exc:
            console.print(f"[yellow]Subtask decomposition failed ({exc}), falling back...[/yellow]")
            return await self._run_single()

    async def _run_single(self) -> str:
        """Execute the standard agent loop (no rate limiting).

        Returns:
            The agent's final text response.
        """
        # Load context
        memory = ProjectMemory(self._project_dir)
        error_db = ErrorPatternDB(self._project_dir)
        assembler = ContextAssembler(self._project_dir)

        project_memory = memory.load()
        error_patterns = error_db.find_similar(task_summary=self._config.task)

        # Try to load index for relevant code
        from nex.indexer.index import IndexBuilder

        builder = IndexBuilder(self._project_dir)
        index = builder.load()
        relevant_code = assembler.select_relevant_code(self._config.task, index)

        system_prompt = assembler.build_system_prompt(
            project_memory=project_memory,
            error_patterns=error_patterns,
            relevant_code=relevant_code,
        )

        # Initial user message
        self._messages = [{"role": "user", "content": self._config.task}]

        console.print(f"\n[bold]Running task:[/bold] {self._config.task}")
        console.print(f"[dim]Max iterations: {self._config.max_iterations}[/dim]\n")

        final_response = ""

        try:
            while self._iteration < self._config.max_iterations:
                self._iteration += 1
                console.print(f"[dim]--- Iteration {self._iteration} ---[/dim]")

                response = await self._client.send_message(
                    messages=self._messages,
                    system=system_prompt,
                    tools=TOOL_DEFINITIONS,
                )

                # Show token usage
                console.print(
                    f"[dim]Tokens: {response.input_tokens} in / "
                    f"{response.output_tokens} out | "
                    f"Cost: ${self._client.usage.estimated_cost:.4f}[/dim]"
                )

                # Check for cost warning
                if self._client.usage.estimated_cost > 1.0:
                    console.print(
                        "[bold yellow]Warning:[/bold yellow] Task has exceeded $1.00 in API costs"
                    )

                # Process response
                has_tool_use = any(block.get("type") == "tool_use" for block in response.content)

                if not has_tool_use:
                    # Task complete — extract text
                    for block in response.content:
                        if block.get("type") == "text":
                            final_response = block.get("text", "")
                            console.print()
                            console.print(
                                Panel(
                                    Markdown(final_response),
                                    title="[bold green]Task Complete[/bold green]",
                                    border_style="green",
                                )
                            )
                    break

                # Execute tool calls
                assistant_content = response.content
                self._messages.append({"role": "assistant", "content": assistant_content})

                tool_results: list[dict[str, Any]] = []
                for block in assistant_content:
                    if block.get("type") == "tool_use":
                        tool_name = block["name"]
                        tool_input = block["input"]
                        tool_id = block["id"]

                        summary = _summarize_input(tool_input)
                        console.print(f"  [cyan]Tool:[/cyan] {tool_name}({summary})")

                        result, modified = await execute_tool(
                            tool_name,
                            tool_input,
                            self._project_dir,
                            self._safety,
                            self._config.dry_run,
                        )
                        if modified:
                            self._files_modified = True

                        if result.success:
                            console.print(f"  [green]OK[/green] ({len(result.output)} chars)")
                        else:
                            console.print(f"  [red]Error:[/red] {result.error}")

                        content = result.output if result.success else f"Error: {result.error}"
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": content,
                                "is_error": not result.success,
                            }
                        )

                self._messages.append({"role": "user", "content": tool_results})

            else:
                max_iter = self._config.max_iterations
                console.print(f"\n[bold yellow]Reached max iterations ({max_iter})[/bold yellow]")
                final_response = "Task did not complete within the iteration limit."

        except KeyboardInterrupt:
            console.print("\n[yellow]Agent interrupted by user.[/yellow]")
            final_response = "Task interrupted."
        finally:
            error_db.close()

        # Post-task: run tests, show diff, and offer to commit
        if self._files_modified:
            tests_passed = await self._run_tests()
            if not tests_passed:
                console.print(
                    "[yellow]Tests failed. Review the changes before committing.[/yellow]"
                )
            await self._post_task_git()

        return final_response

    async def _run_with_subtasks(self, token_rate_limit: int) -> str:
        """Execute the task via planner decomposition with memory updates.

        Always decomposes the task into subtasks, builds scoped context for
        each, updates project memory after every subtask, and optionally
        paces API calls when a token rate limit is set.

        Args:
            token_rate_limit: Max input tokens per minute (0 = no limit).

        Returns:
            The combined final response.
        """
        memory = ProjectMemory(self._project_dir)
        error_db = ErrorPatternDB(self._project_dir)
        assembler = ContextAssembler(self._project_dir)

        project_memory = memory.load()
        error_patterns = error_db.find_similar(task_summary=self._config.task)

        # Load index
        from nex.indexer.index import IndexBuilder

        builder = IndexBuilder(self._project_dir)
        index = builder.load()

        # Decompose task via planner
        haiku_model = "claude-haiku-4-5-20251001"
        if self._nex_config:
            haiku_model = self._nex_config.haiku_model

        planner = Planner(self._client, haiku_model=haiku_model)
        rate_limiter = RateLimiter(tokens_per_minute=token_rate_limit)

        console.print("\n[bold]Decomposing task into subtasks...[/bold]")
        subtasks = await planner.plan(self._config.task, project_memory, rate_limiter)

        console.print(f"\n[bold]Running task:[/bold] {self._config.task}")
        rate_info = f"Rate limit: {token_rate_limit} tokens/min | " if token_rate_limit else ""
        console.print(
            f"[dim]Subtasks: {len(subtasks)} | "
            f"{rate_info}"
            f"Max iterations: {self._config.max_iterations}[/dim]\n"
        )
        budget = 20_000
        if self._nex_config:
            budget = self._nex_config.subtask_token_budget

        # Split iteration budget across subtasks (min 5 each)
        iters_per_subtask = max(5, self._config.max_iterations // max(len(subtasks), 1))

        prior_context = ""
        subtask_results: list[str] = []

        try:
            for i, subtask in enumerate(subtasks, 1):
                console.print(
                    Panel(
                        f"[bold]{subtask.description}[/bold]\n"
                        f"[dim]Files: {', '.join(subtask.file_paths) or 'auto'}[/dim]",
                        title=f"[bold cyan]Subtask {i}/{len(subtasks)}[/bold cyan]",
                        border_style="cyan",
                    )
                )

                # Build scoped context
                scoped_code = assembler.select_scoped_code(
                    subtask.file_paths, subtask.description, index, budget
                )

                system_prompt = assembler.build_subtask_prompt(
                    subtask_description=subtask.description,
                    project_memory=project_memory,
                    error_patterns=error_patterns,
                    relevant_code=scoped_code,
                    prior_context=prior_context,
                )

                sub_result = await self._run_subtask_loop(
                    system_prompt=system_prompt,
                    task=subtask.description,
                    max_iterations=iters_per_subtask,
                    rate_limiter=rate_limiter,
                    memory=memory,
                )

                subtask_results.append(sub_result.text)

                # Update memory after each subtask
                await self._generate_memory_update(
                    memory, subtask.description, sub_result, haiku_model, rate_limiter
                )
                memory.prune_section("Session Log")
                project_memory = memory.load()

                # Build prior context for the next subtask (keep it compact)
                prior_context += (
                    f"- Subtask {i}: {subtask.description} -> {sub_result.text[:200]}\n"
                )

        except KeyboardInterrupt:
            console.print("\n[yellow]Agent interrupted by user.[/yellow]")
        finally:
            error_db.close()

        final_response = (
            "\n\n".join(subtask_results) if subtask_results else "No subtasks completed."
        )

        console.print()
        console.print(
            Panel(
                Markdown(final_response),
                title="[bold green]All Subtasks Complete[/bold green]",
                border_style="green",
            )
        )

        # Post-task: run tests, show diff, and offer to commit
        if self._files_modified:
            tests_passed = await self._run_tests()
            if not tests_passed:
                console.print(
                    "[yellow]Tests failed. Review the changes before committing.[/yellow]"
                )
            await self._post_task_git()

        return final_response

    async def _run_subtask_loop(
        self,
        system_prompt: str,
        task: str,
        max_iterations: int,
        rate_limiter: RateLimiter,
        memory: ProjectMemory | None = None,
    ) -> SubtaskResult:
        """Run a focused mini agent loop for a single subtask.

        Uses fresh message history to prevent token accumulation. Paces
        API calls via the rate limiter. Tracks files touched and writes
        a mid-subtask memory checkpoint for long-running subtasks.

        Args:
            system_prompt: Scoped system prompt for this subtask.
            task: The subtask description.
            max_iterations: Max tool call iterations for this subtask.
            rate_limiter: Rate limiter to pace API calls.
            memory: Project memory for mid-subtask checkpoints.

        Returns:
            A SubtaskResult with the response text, files touched, and
            iteration count.
        """
        messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
        iteration = 0
        files_touched: list[str] = []

        while iteration < max_iterations:
            iteration += 1
            console.print(f"[dim]--- Subtask iteration {iteration} ---[/dim]")

            # Mid-subtask memory checkpoint for long-running subtasks
            if iteration == _MID_SUBTASK_MEMORY_THRESHOLD and memory is not None:
                self._generate_mid_subtask_memory_update(memory, task, iteration, files_touched)

            # Estimate tokens and wait if needed
            estimated = ContextAssembler.estimate_tokens(system_prompt + task)
            await rate_limiter.wait_if_needed(estimated)

            response = await self._client.send_message(
                messages=messages,
                system=system_prompt,
                tools=TOOL_DEFINITIONS,
            )

            # Record actual tokens
            rate_limiter.record(response.input_tokens)

            # Show token usage
            console.print(
                f"[dim]Tokens: {response.input_tokens} in / "
                f"{response.output_tokens} out | "
                f"Cost: ${self._client.usage.estimated_cost:.4f}[/dim]"
            )

            has_tool_use = any(block.get("type") == "tool_use" for block in response.content)

            if not has_tool_use:
                text = ""
                for block in response.content:
                    if block.get("type") == "text":
                        text = str(block.get("text", ""))
                        break
                return SubtaskResult(text=text, files_touched=files_touched, iterations=iteration)

            # Execute tool calls
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results: list[dict[str, Any]] = []
            for block in assistant_content:
                if block.get("type") == "tool_use":
                    tool_name = block["name"]
                    tool_input = block["input"]
                    tool_id = block["id"]

                    summary = _summarize_input(tool_input)
                    console.print(f"  [cyan]Tool:[/cyan] {tool_name}({summary})")

                    result, modified = await execute_tool(
                        tool_name,
                        tool_input,
                        self._project_dir,
                        self._safety,
                        self._config.dry_run,
                    )
                    if modified:
                        self._files_modified = True
                        # Track written files
                        if tool_name == "write_file" and "path" in tool_input:
                            files_touched.append(tool_input["path"])

                    if result.success:
                        console.print(f"  [green]OK[/green] ({len(result.output)} chars)")
                    else:
                        console.print(f"  [red]Error:[/red] {result.error}")

                    content = result.output if result.success else f"Error: {result.error}"
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": content,
                            "is_error": not result.success,
                        }
                    )

            messages.append({"role": "user", "content": tool_results})

        return SubtaskResult(
            text="Subtask did not complete within iteration limit.",
            files_touched=files_touched,
            iterations=iteration,
        )

    async def _generate_memory_update(
        self,
        memory: ProjectMemory,
        subtask_description: str,
        result: SubtaskResult,
        haiku_model: str,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        """Summarize a completed subtask and append to Session Log.

        Uses Haiku to produce a concise 1-2 sentence summary of what
        the subtask accomplished and which files were modified.

        Args:
            memory: Project memory instance.
            subtask_description: What the subtask was supposed to do.
            result: The subtask result with text, files, and iterations.
            haiku_model: Model ID for the summary call.
            rate_limiter: Optional rate limiter to pace the API call.
        """
        truncated = result.text[:500]
        files_str = ", ".join(result.files_touched[:10]) or "none"
        user_msg = (
            f"Subtask: {subtask_description}\nResult: {truncated}\nFiles modified: {files_str}"
        )

        try:
            if rate_limiter is not None:
                estimated = ContextAssembler.estimate_tokens(user_msg)
                await rate_limiter.wait_if_needed(estimated)

            response = await self._client.send_message(
                messages=[{"role": "user", "content": user_msg}],
                system=_MEMORY_SUMMARY_PROMPT,
                model=haiku_model,
                max_tokens=256,
            )

            if rate_limiter is not None:
                rate_limiter.record(response.input_tokens)

            summary = ""
            for block in response.content:
                if block.get("type") == "text":
                    summary = block.get("text", "").strip()
                    break

            if summary:
                today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
                memory.append("Session Log", f"- [{today}] {summary}")
        except Exception as exc:
            console.print(f"[yellow]Memory update skipped:[/yellow] {exc}")

    def _generate_mid_subtask_memory_update(
        self,
        memory: ProjectMemory,
        task: str,
        iteration: int,
        files_touched: list[str],
    ) -> None:
        """Write a lightweight progress checkpoint to Session Log.

        Called when a subtask exceeds ``_MID_SUBTASK_MEMORY_THRESHOLD``
        iterations, without making an additional API call.

        Args:
            memory: Project memory instance.
            task: The subtask description.
            iteration: Current iteration number.
            files_touched: Files written so far.
        """
        files_str = ", ".join(files_touched[:10]) or "none"
        note = f"- [In progress] {task} ({iteration} iterations, files: {files_str})"
        try:
            memory.append("Session Log", note)
        except Exception:
            pass

    async def _run_tests(self) -> bool:
        """Detect and run the project's test suite.

        Returns:
            True if tests passed or no test runner was detected.
        """
        runner = TestRunner(self._project_dir)

        # Use config override if available
        command = None
        timeout = 120
        if self._nex_config:
            if self._nex_config.test_command:
                command = self._nex_config.test_command
            timeout = self._nex_config.test_timeout

        if command is None:
            command = runner.detect()

        if command is None:
            return True

        console.print(f"\n[bold]Running tests:[/bold] {command}")
        result = await runner.run(command, timeout)

        if result.success:
            console.print(
                Panel(
                    result.output[:2000] if len(result.output) > 2000 else result.output,
                    title="[bold green]Tests Passed[/bold green]",
                    border_style="green",
                )
            )
        else:
            console.print(
                Panel(
                    result.output[:2000] if len(result.output) > 2000 else result.output,
                    title="[bold red]Tests Failed[/bold red]",
                    border_style="red",
                )
            )

        return result.success

    async def _post_task_git(self) -> None:
        """Show diff and offer to commit after file modifications."""
        try:
            git = GitOperations(self._project_dir)
            if not git.is_repo():
                return

            diff = git.diff()
            if diff:
                console.print("\n[bold]Changes made:[/bold]")
                console.print(Syntax(diff, "diff", theme="monokai"))

            answer = Prompt.ask(
                "\n[bold]Commit these changes?[/bold]",
                choices=["y", "n"],
                default="n",
            )
            if answer.lower() == "y":
                msg = Prompt.ask("[bold]Commit message[/bold]")
                if msg:
                    git.commit(msg)
        except (ToolError, NexError) as exc:
            console.print(f"[yellow]Git operation failed:[/yellow] {exc}")


class ChatSession:
    """Interactive chat session with subtask decomposition.

    Each user turn is decomposed into subtasks via the planner.
    Each subtask runs with scoped context in its own mini loop,
    keeping the main chat history clean. Falls back to direct
    execution if the planner fails.
    """

    def __init__(
        self,
        api_client: AnthropicClient,
        system_prompt: str,
        project_dir: Path,
        safety: SafetyLayer,
        dry_run: bool = False,
        max_iterations: int = 25,
        rate_limiter: RateLimiter | None = None,
        memory: ProjectMemory | None = None,
        haiku_model: str = "claude-haiku-4-5-20251001",
        assembler: ContextAssembler | None = None,
        error_patterns: list[Any] | None = None,
        index: Any | None = None,
        subtask_token_budget: int = 20_000,
    ) -> None:
        """Initialize a chat session.

        Args:
            api_client: Anthropic API client.
            system_prompt: Pre-assembled system prompt.
            project_dir: Project root directory.
            safety: Safety layer for command approval.
            dry_run: If True, skip destructive operations.
            max_iterations: Max tool calls per user turn.
            rate_limiter: Optional rate limiter for API calls.
            memory: Optional project memory for session log updates.
            haiku_model: Model for memory summary and planning calls.
            assembler: Context assembler for building scoped subtask prompts.
            error_patterns: Past error patterns for subtask context.
            index: Code index for relevant code selection.
            subtask_token_budget: Token budget per subtask context.
        """
        self._client = api_client
        self._system_prompt = system_prompt
        self._project_dir = project_dir
        self._safety = safety
        self._dry_run = dry_run
        self._max_iterations = max_iterations
        self._rate_limiter = rate_limiter
        self._memory = memory
        self._haiku_model = haiku_model
        self._assembler = assembler
        self._error_patterns = error_patterns or []
        self._index = index
        self._subtask_token_budget = subtask_token_budget
        self._messages: list[dict[str, Any]] = []
        self._turn_count = 0
        self._files_modified = False

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Return the conversation message history."""
        return self._messages

    @property
    def turn_count(self) -> int:
        """Return the number of user turns processed."""
        return self._turn_count

    @property
    def files_modified(self) -> bool:
        """Return whether any files have been modified."""
        return self._files_modified

    async def send(self, user_message: str) -> str:
        """Process a user turn via subtask decomposition.

        Decomposes the message into subtasks, executes each with
        scoped context, and appends a summary to chat history. Falls
        back to direct execution if the planner fails.

        Args:
            user_message: The user's message text.

        Returns:
            The assistant's final text response for this turn.
        """
        self._turn_count += 1

        # Try subtask decomposition if assembler is available
        if self._assembler is not None:
            try:
                return await self._send_with_subtasks(user_message)
            except Exception as exc:
                console.print(
                    f"[yellow]Subtask decomposition failed ({exc}), using direct mode...[/yellow]"
                )

        return await self._send_direct(user_message)

    async def _send_with_subtasks(self, user_message: str) -> str:
        """Decompose user message into subtasks and execute each.

        Each subtask runs in its own mini loop with scoped context.
        Results are collected and a combined response is appended to
        the main chat history.

        Args:
            user_message: The user's message text.

        Returns:
            Combined response from all subtasks.
        """
        assert self._assembler is not None

        project_memory = self._memory.load() if self._memory else ""

        # Decompose via planner
        planner = Planner(self._client, haiku_model=self._haiku_model)
        console.print("\n[bold]Decomposing into subtasks...[/bold]")
        subtasks = await planner.plan(user_message, project_memory, self._rate_limiter)

        console.print(
            f"[dim]Subtasks: {len(subtasks)} | Max iterations: {self._max_iterations}[/dim]\n"
        )

        iters_per_subtask = max(5, self._max_iterations // max(len(subtasks), 1))

        prior_context = ""
        subtask_results: list[str] = []

        for i, subtask in enumerate(subtasks, 1):
            console.print(
                Panel(
                    f"[bold]{subtask.description}[/bold]\n"
                    f"[dim]Files: {', '.join(subtask.file_paths) or 'auto'}[/dim]",
                    title=f"[bold cyan]Subtask {i}/{len(subtasks)}[/bold cyan]",
                    border_style="cyan",
                )
            )

            # Build scoped context for this subtask
            scoped_code = self._assembler.select_scoped_code(
                subtask.file_paths,
                subtask.description,
                self._index,
                self._subtask_token_budget,
            )

            system_prompt = self._assembler.build_subtask_prompt(
                subtask_description=subtask.description,
                project_memory=project_memory,
                error_patterns=self._error_patterns,
                relevant_code=scoped_code,
                prior_context=prior_context,
            )

            # Run subtask in a mini loop with fresh messages
            sub_result = await self._run_subtask_loop(
                system_prompt=system_prompt,
                task=subtask.description,
                max_iterations=iters_per_subtask,
            )

            subtask_results.append(sub_result.text)

            # Update memory after each subtask
            if self._memory is not None:
                await self._update_memory_for_subtask(subtask.description, sub_result)
                self._memory.prune_section("Session Log")
                project_memory = self._memory.load()

            prior_context += f"- Subtask {i}: {subtask.description} -> {sub_result.text[:200]}\n"

        combined = "\n\n".join(subtask_results) if subtask_results else "No subtasks completed."

        # Append to main chat history as a clean user/assistant pair
        self._messages.append({"role": "user", "content": user_message})
        self._messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": combined}]}
        )

        return combined

    async def _run_subtask_loop(
        self,
        system_prompt: str,
        task: str,
        max_iterations: int,
    ) -> SubtaskResult:
        """Run a focused mini loop for a single subtask.

        Uses fresh message history to keep each subtask isolated.

        Args:
            system_prompt: Scoped system prompt for this subtask.
            task: The subtask description.
            max_iterations: Max iterations for this subtask.

        Returns:
            SubtaskResult with text, files, and iteration count.
        """
        messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
        iteration = 0
        files_touched: list[str] = []

        while iteration < max_iterations:
            iteration += 1
            console.print(f"[dim]--- Subtask iteration {iteration} ---[/dim]")

            if self._rate_limiter is not None:
                estimated = ContextAssembler.estimate_tokens(system_prompt + task)
                await self._rate_limiter.wait_if_needed(estimated)

            response = await self._client.send_message(
                messages=messages,
                system=system_prompt,
                tools=TOOL_DEFINITIONS,
            )

            if self._rate_limiter is not None:
                self._rate_limiter.record(response.input_tokens)

            console.print(
                f"[dim]Tokens: {response.input_tokens} in / "
                f"{response.output_tokens} out | "
                f"Cost: ${self._client.usage.estimated_cost:.4f}[/dim]"
            )

            cost = self._client.usage.estimated_cost
            if cost > 5.0:
                console.print(
                    "[bold red]Warning:[/bold red] Session has exceeded $5.00 in API costs"
                )
            elif cost > 1.0:
                console.print(
                    "[bold yellow]Warning:[/bold yellow] Session has exceeded $1.00 in API costs"
                )

            has_tool_use = any(block.get("type") == "tool_use" for block in response.content)

            if not has_tool_use:
                text = ""
                for block in response.content:
                    if block.get("type") == "text":
                        text = str(block.get("text", ""))
                        break
                return SubtaskResult(text=text, files_touched=files_touched, iterations=iteration)

            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results: list[dict[str, Any]] = []
            for block in assistant_content:
                if block.get("type") == "tool_use":
                    tool_name = block["name"]
                    tool_input = block["input"]
                    tool_id = block["id"]

                    summary = _summarize_input(tool_input)
                    console.print(f"  [cyan]Tool:[/cyan] {tool_name}({summary})")

                    result, modified = await execute_tool(
                        tool_name,
                        tool_input,
                        self._project_dir,
                        self._safety,
                        self._dry_run,
                    )
                    if modified:
                        self._files_modified = True
                        if tool_name == "write_file" and "path" in tool_input:
                            files_touched.append(tool_input["path"])

                    if result.success:
                        console.print(f"  [green]OK[/green] ({len(result.output)} chars)")
                    else:
                        console.print(f"  [red]Error:[/red] {result.error}")

                    content = result.output if result.success else f"Error: {result.error}"
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": content,
                            "is_error": not result.success,
                        }
                    )

            messages.append({"role": "user", "content": tool_results})

        return SubtaskResult(
            text="Subtask did not complete within iteration limit.",
            files_touched=files_touched,
            iterations=iteration,
        )

    async def _send_direct(self, user_message: str) -> str:
        """Process a user turn directly without subtask decomposition.

        Fallback mode when planner is unavailable or fails.

        Args:
            user_message: The user's message text.

        Returns:
            The assistant's final text response.
        """
        self._messages.append({"role": "user", "content": user_message})

        iterations = 0
        final_text = ""
        turn_files: list[str] = []

        while iterations < self._max_iterations:
            iterations += 1

            if self._rate_limiter is not None:
                estimated = ContextAssembler.estimate_tokens(self._system_prompt + user_message)
                await self._rate_limiter.wait_if_needed(estimated)

            response = await self._client.send_message(
                messages=self._messages,
                system=self._system_prompt,
                tools=TOOL_DEFINITIONS,
            )

            if self._rate_limiter is not None:
                self._rate_limiter.record(response.input_tokens)

            console.print(
                f"[dim]Tokens: {response.input_tokens} in / "
                f"{response.output_tokens} out | "
                f"Cost: ${self._client.usage.estimated_cost:.4f}[/dim]"
            )

            cost = self._client.usage.estimated_cost
            if cost > 5.0:
                console.print(
                    "[bold red]Warning:[/bold red] Session has exceeded $5.00 in API costs"
                )
            elif cost > 1.0:
                console.print(
                    "[bold yellow]Warning:[/bold yellow] Session has exceeded $1.00 in API costs"
                )

            has_tool_use = any(block.get("type") == "tool_use" for block in response.content)

            if not has_tool_use:
                for block in response.content:
                    if block.get("type") == "text":
                        final_text = block.get("text", "")
                self._messages.append({"role": "assistant", "content": response.content})
                break

            assistant_content = response.content
            self._messages.append({"role": "assistant", "content": assistant_content})

            tool_results: list[dict[str, Any]] = []
            for block in assistant_content:
                if block.get("type") == "tool_use":
                    tool_name = block["name"]
                    tool_input = block["input"]
                    tool_id = block["id"]

                    summary = _summarize_input(tool_input)
                    console.print(f"  [cyan]Tool:[/cyan] {tool_name}({summary})")

                    result, modified = await execute_tool(
                        tool_name,
                        tool_input,
                        self._project_dir,
                        self._safety,
                        self._dry_run,
                    )
                    if modified:
                        self._files_modified = True
                        if tool_name == "write_file" and "path" in tool_input:
                            turn_files.append(tool_input["path"])

                    if result.success:
                        console.print(f"  [green]OK[/green] ({len(result.output)} chars)")
                    else:
                        console.print(f"  [red]Error:[/red] {result.error}")

                    content = result.output if result.success else f"Error: {result.error}"
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": content,
                            "is_error": not result.success,
                        }
                    )

            self._messages.append({"role": "user", "content": tool_results})

        # Update memory after turns that modified files
        if turn_files and self._memory is not None:
            await self._update_memory_for_direct(user_message, final_text, turn_files)

        return final_text

    async def _update_memory_for_subtask(
        self, subtask_description: str, result: SubtaskResult
    ) -> None:
        """Summarize a completed subtask and append to Session Log.

        Args:
            subtask_description: What the subtask was supposed to do.
            result: The subtask result.
        """
        if self._memory is None:
            return

        truncated = result.text[:500]
        files_str = ", ".join(result.files_touched[:10]) or "none"
        user_msg = (
            f"Subtask: {subtask_description}\nResult: {truncated}\nFiles modified: {files_str}"
        )

        try:
            if self._rate_limiter is not None:
                estimated = ContextAssembler.estimate_tokens(user_msg)
                await self._rate_limiter.wait_if_needed(estimated)

            resp = await self._client.send_message(
                messages=[{"role": "user", "content": user_msg}],
                system=_MEMORY_SUMMARY_PROMPT,
                model=self._haiku_model,
                max_tokens=256,
            )

            if self._rate_limiter is not None:
                self._rate_limiter.record(resp.input_tokens)

            summary = ""
            for block in resp.content:
                if block.get("type") == "text":
                    summary = block.get("text", "").strip()
                    break

            if summary:
                today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
                self._memory.append("Session Log", f"- [{today}] {summary}")
        except Exception as exc:
            console.print(f"[yellow]Memory update skipped:[/yellow] {exc}")

    async def _update_memory_for_direct(
        self, user_message: str, response_text: str, files: list[str]
    ) -> None:
        """Summarize a direct-mode turn and append to Session Log.

        Args:
            user_message: The user's input for this turn.
            response_text: The assistant's final response.
            files: Files written during this turn.
        """
        if self._memory is None:
            return

        truncated_response = response_text[:300]
        files_str = ", ".join(files[:10]) or "none"
        user_msg = (
            f"User asked: {user_message[:200]}\n"
            f"Result: {truncated_response}\n"
            f"Files modified: {files_str}"
        )

        try:
            if self._rate_limiter is not None:
                estimated = ContextAssembler.estimate_tokens(user_msg)
                await self._rate_limiter.wait_if_needed(estimated)

            resp = await self._client.send_message(
                messages=[{"role": "user", "content": user_msg}],
                system=_MEMORY_SUMMARY_PROMPT,
                model=self._haiku_model,
                max_tokens=256,
            )

            if self._rate_limiter is not None:
                self._rate_limiter.record(resp.input_tokens)

            summary = ""
            for block in resp.content:
                if block.get("type") == "text":
                    summary = block.get("text", "").strip()
                    break

            if summary:
                today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
                self._memory.append("Session Log", f"- [{today}] {summary}")
                self._memory.prune_section("Session Log")
        except Exception as exc:
            console.print(f"[yellow]Memory update skipped:[/yellow] {exc}")


async def run_task(task: str, config: NexConfig) -> None:
    """Entry point called by the CLI to run a task.

    Args:
        task: The user's task description.
        config: Nex configuration.
    """
    api_key = config.api_key
    client = AnthropicClient(api_key=api_key, default_model=config.model)
    safety = SafetyLayer(dry_run=config.dry_run)

    agent_config = AgentConfig(
        project_dir=config.project_dir,
        task=task,
        dry_run=config.dry_run,
        max_iterations=config.max_iterations,
    )

    agent = Agent(config=agent_config, api_client=client, safety=safety, nex_config=config)

    try:
        await agent.run()
    finally:
        await client.close()

    # Print final cost summary
    console.print(
        f"\n[dim]Total cost: ${client.usage.estimated_cost:.4f} "
        f"({client.usage.total_input} input + {client.usage.total_output} output tokens)[/dim]"
    )


def _summarize_input(tool_input: dict[str, Any]) -> str:
    """Create a short summary of tool input for display."""
    parts: list[str] = []
    for key, value in tool_input.items():
        if isinstance(value, str) and len(value) > 50:
            parts.append(f'{key}="{value[:47]}..."')
        elif isinstance(value, str):
            parts.append(f'{key}="{value}"')
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)
