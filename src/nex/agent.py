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

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax

from nex.api_client import AnthropicClient, APIResponse
from nex.config import NexConfig
from nex.context import ContextAssembler
from nex.exceptions import NexError, SafetyError, ToolError
from nex.memory.decisions import DecisionLog
from nex.memory.errors import ErrorPattern, ErrorPatternDB
from nex.memory.project import ProjectMemory
from nex.safety import SafetyLayer
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


class Agent:
    """Core agent loop — orchestrates tools, API calls, and memory."""

    def __init__(
        self,
        config: AgentConfig,
        api_client: AnthropicClient,
        safety: SafetyLayer,
    ) -> None:
        """Initialize the agent.

        Args:
            config: Agent run configuration.
            api_client: Anthropic API client.
            safety: Safety layer for command approval.
        """
        self._config = config
        self._client = api_client
        self._safety = safety
        self._project_dir = config.project_dir
        self._messages: list[dict[str, Any]] = []
        self._iteration = 0
        self._files_modified = False

    async def run(self) -> str:
        """Execute the agent loop and return the final response.

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
                has_tool_use = any(
                    block.get("type") == "tool_use" for block in response.content
                )

                if not has_tool_use:
                    # Task complete — extract text
                    for block in response.content:
                        if block.get("type") == "text":
                            final_response = block.get("text", "")
                            console.print()
                            console.print(Panel(
                                Markdown(final_response),
                                title="[bold green]Task Complete[/bold green]",
                                border_style="green",
                            ))
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

                        console.print(f"  [cyan]Tool:[/cyan] {tool_name}({_summarize_input(tool_input)})")

                        result = await self._execute_tool(tool_name, tool_input)

                        if result.success:
                            console.print(f"  [green]OK[/green] ({len(result.output)} chars)")
                        else:
                            console.print(f"  [red]Error:[/red] {result.error}")

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": result.output if result.success else f"Error: {result.error}",
                            "is_error": not result.success,
                        })

                self._messages.append({"role": "user", "content": tool_results})

            else:
                console.print(
                    f"\n[bold yellow]Reached max iterations ({self._config.max_iterations})[/bold yellow]"
                )
                final_response = "Task did not complete within the iteration limit."

        except KeyboardInterrupt:
            console.print("\n[yellow]Agent interrupted by user.[/yellow]")
            final_response = "Task interrupted."
        finally:
            error_db.close()

        # Post-task: show diff and offer to commit
        if self._files_modified:
            await self._post_task_git()

        return final_response

    async def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> ToolResult:
        """Route a tool call to the correct handler.

        Args:
            tool_name: Name of the tool to execute.
            tool_input: Tool input parameters.

        Returns:
            ToolResult from the tool execution.
        """
        try:
            if tool_name == "read_file":
                return await read_file(tool_input["path"], self._project_dir)

            elif tool_name == "write_file":
                # Safety check for file writes
                check = self._safety.check_file_write(tool_input["path"], self._project_dir)
                if not check.is_safe:
                    if check.requires_approval:
                        approved = await self._safety.request_approval(
                            f"Write to {tool_input['path']}", check.reason or ""
                        )
                        if not approved:
                            return ToolResult(success=False, output="", error="Write denied by user")
                    else:
                        return ToolResult(success=False, output="", error=check.reason or "Write blocked")

                result = await write_file(tool_input["path"], tool_input["content"], self._project_dir)
                if result.success:
                    self._files_modified = True
                return result

            elif tool_name == "run_command":
                # Safety check for commands
                try:
                    await self._safety.guard_command(tool_input["command"])
                except SafetyError as exc:
                    return ToolResult(success=False, output="", error=str(exc))

                if self._config.dry_run:
                    return ToolResult(
                        success=True,
                        output=f"[dry-run] Would execute: {tool_input['command']}",
                    )

                return await run_command(tool_input["command"], self._project_dir)

            elif tool_name == "search_files":
                return await search_files(
                    tool_input["pattern"],
                    tool_input.get("path", "."),
                    self._project_dir,
                )

            elif tool_name == "list_directory":
                return await self._list_directory(
                    tool_input.get("path", "."),
                    tool_input.get("depth", 3),
                )

            else:
                return ToolResult(success=False, output="", error=f"Unknown tool: {tool_name}")

        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, output="", error=f"Tool error: {exc}")

    async def _list_directory(self, path: str, depth: int) -> ToolResult:
        """List directory contents recursively."""
        from nex.tools.file_ops import _resolve_safe_path

        try:
            resolved = _resolve_safe_path(path, self._project_dir)
        except ToolError as exc:
            return ToolResult(success=False, output="", error=str(exc))

        if not resolved.is_dir():
            return ToolResult(success=False, output="", error=f"Not a directory: {path}")

        lines: list[str] = []
        self._walk_tree(resolved, "", depth, lines)
        return ToolResult(success=True, output="\n".join(lines))

    def _walk_tree(self, directory: Path, prefix: str, depth: int, lines: list[str]) -> None:
        """Recursively build a directory tree listing."""
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
                self._walk_tree(entry, prefix + extension, depth - 1, lines)

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

    agent = Agent(config=agent_config, api_client=client, safety=safety)

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
