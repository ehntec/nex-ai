"""Typer CLI entry point for Nex AI.

Bridges the synchronous Typer world to the async agent internals via asyncio.run().
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from nex import __version__
from nex.config import NexConfig, load_config, save_global_config
from nex.exceptions import NexError

app = typer.Typer(
    name="nex",
    help="Nex AI — The coding agent that remembers.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()

_MEMORY_TEMPLATE = """\
# Project Overview

<!-- Describe what this project is and what it does. -->

## Tech Stack

<!-- List the main technologies, frameworks, and tools used. -->

## Architecture

<!-- High-level architecture notes. -->

## Conventions

<!-- Coding conventions, naming patterns, commit style. -->

## Notes

<!-- Anything else the agent should remember between sessions. -->
"""

_CONFIG_TEMPLATE = """\
# Nex project configuration
# model = "claude-sonnet-4-20250514"
# max_iterations = 25
# dry_run = false
token_rate_limit = 20000
"""


def _error_exit(message: str, hint: str | None = None) -> None:
    """Print a styled error and exit."""
    console.print(f"[bold red]Error:[/bold red] {message}")
    if hint:
        console.print(f"[dim]Hint: {hint}[/dim]")
    raise typer.Exit(code=1)


@app.command()
def main(
    task: Annotated[str, typer.Argument(help="The coding task to execute")],
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show plan without executing")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose output")] = False,
) -> None:
    """Run a coding task with Nex AI."""
    try:
        config = load_config(Path.cwd())

        if dry_run:
            config.dry_run = True
        if verbose:
            config.log_level = "DEBUG"

        if not config.api_key:
            _error_exit(
                "Anthropic API key not found.",
                hint="Run 'nex auth' or set ANTHROPIC_API_KEY environment variable.",
            )

        nex_dir = Path.cwd() / ".nex"
        if not nex_dir.is_dir():
            _error_exit("Project not initialized.", hint="Run 'nex init' first.")

        console.print(
            Panel(
                f"[bold]Task:[/bold] {task}",
                title=f"[bold cyan]Nex AI[/bold cyan] v{__version__}",
                border_style="cyan",
            )
        )

        if config.dry_run:
            console.print("[yellow]Running in dry-run mode.[/yellow]\n")

        from nex.agent import run_task

        asyncio.run(run_task(task, config))

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(code=130)
    except NexError as exc:
        _error_exit(str(exc))
    except Exception as exc:
        console.print_exception(show_locals=False)
        _error_exit(f"Unexpected error: {exc}")


@app.command()
def init() -> None:
    """Initialize .nex/ directory for this project."""
    project_dir = Path.cwd().resolve()
    nex_dir = project_dir / ".nex"

    if nex_dir.is_dir():
        console.print(f"[yellow]Already initialized:[/yellow] .nex/ exists at {nex_dir}")
        raise typer.Exit(code=0)

    nex_dir.mkdir(parents=True, exist_ok=True)

    (nex_dir / "memory.md").write_text(_MEMORY_TEMPLATE, encoding="utf-8")
    (nex_dir / "decisions.md").write_text("# Decision Log\n\n", encoding="utf-8")
    (nex_dir / "config.toml").write_text(_CONFIG_TEMPLATE, encoding="utf-8")
    (nex_dir / ".gitignore").write_text("errors.db\nindex.json\n", encoding="utf-8")

    console.print(
        Panel(
            Text.assemble(
                ("Initialized Nex AI in ", "green"),
                (str(nex_dir), "bold green"),
                ("\n\nCreated:\n", "white"),
                ("  memory.md     ", "cyan"),
                ("- project memory (edit this!)\n", "dim"),
                ("  decisions.md  ", "cyan"),
                ("- decision log\n", "dim"),
                ("  config.toml   ", "cyan"),
                ("- project settings\n", "dim"),
                ("  .gitignore    ", "cyan"),
                ("- ignores errors.db, index.json", "dim"),
            ),
            title="[bold green]Project Initialized[/bold green]",
            border_style="green",
        )
    )

    console.print(
        "\n[dim]Next steps:[/dim]\n"
        "  1. Edit [cyan].nex/memory.md[/cyan] with your project details\n"
        "  2. Run [cyan]nex auth[/cyan] to configure your API key\n"
        '  3. Run [cyan]nex "your first task"[/cyan]\n'
    )


@app.command()
def index() -> None:
    """Build the codebase index (.nex/index.json)."""
    nex_dir = Path.cwd() / ".nex"
    if not nex_dir.is_dir():
        _error_exit("Project not initialized.", hint="Run 'nex init' first.")
        return

    from nex.indexer.index import IndexBuilder

    builder = IndexBuilder(Path.cwd())

    start = time.perf_counter()
    idx = builder.build()
    elapsed = time.perf_counter() - start

    if not idx.files:
        console.print(
            "[yellow]No source files found.[/yellow]\n"
            "[dim]Hint: Make sure your project has .py, .js, .ts, .go, .rs, or other "
            "supported source files.[/dim]"
        )
        raise typer.Exit(code=0)

    console.print(
        Panel(
            Text.assemble(
                ("Files indexed:   ", "cyan"),
                (str(len(idx.files)), "bold"),
                ("\n", ""),
                ("Symbols found:   ", "cyan"),
                (str(len(idx.symbols)), "bold"),
                ("\n", ""),
                ("Time:            ", "cyan"),
                (f"{elapsed:.2f}s", "bold"),
                ("\n", ""),
                ("Saved to:        ", "cyan"),
                (str(builder.index_path), "dim"),
            ),
            title="[bold green]Index Built[/bold green]",
            border_style="green",
        )
    )


@app.command()
def status() -> None:
    """Show project memory, error count, and index stats."""
    nex_dir = Path.cwd() / ".nex"
    if not nex_dir.is_dir():
        _error_exit("Project not initialized.", hint="Run 'nex init' first.")
        return

    config = load_config(Path.cwd())

    table = Table(title="Nex AI Status", border_style="cyan", header_style="bold cyan")
    table.add_column("Property", style="bold")
    table.add_column("Value")

    table.add_row("Project", str(Path.cwd()))

    # Memory
    memory_path = nex_dir / "memory.md"
    if memory_path.is_file():
        lines = len(memory_path.read_text(encoding="utf-8").splitlines())
        table.add_row("Memory", f"{lines} lines")
    else:
        table.add_row("Memory", "[red]Not found[/red]")

    # Error DB
    errors_path = nex_dir / "errors.db"
    if errors_path.is_file():
        import sqlite3

        try:
            conn = sqlite3.connect(str(errors_path))
            count = conn.execute("SELECT COUNT(*) FROM error_patterns").fetchone()[0]
            conn.close()
            table.add_row("Error patterns", str(count))
        except sqlite3.Error:
            table.add_row("Error patterns", "[yellow]Unreadable[/yellow]")
    else:
        table.add_row("Error patterns", "[dim]None yet[/dim]")

    # Index
    index_path = nex_dir / "index.json"
    if index_path.is_file():
        import json

        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
            table.add_row(
                "Index",
                f"{len(data.get('files', []))} files, {len(data.get('symbols', []))} symbols",
            )
        except (json.JSONDecodeError, KeyError):
            table.add_row("Index", "[yellow]Malformed[/yellow]")
    else:
        table.add_row("Index", "[dim]Not built[/dim]")

    # Config
    table.add_row("Model", config.model)
    table.add_row("Max iterations", str(config.max_iterations))
    table.add_row("API key", "[green]Set[/green]" if config.api_key else "[red]Missing[/red]")

    console.print()
    console.print(table)
    console.print()


@app.command()
def auth() -> None:
    """Set up Anthropic API key."""
    console.print(
        Panel(
            "Configure your Anthropic API key.\n"
            "Get one at [link=https://console.anthropic.com/]console.anthropic.com[/link]",
            title="[bold cyan]API Key Setup[/bold cyan]",
            border_style="cyan",
        )
    )

    api_key = Prompt.ask("\n[bold]Anthropic API key[/bold]")
    if not api_key.strip():
        _error_exit("No key provided.")
        return

    api_key = api_key.strip()
    if not api_key.startswith("sk-ant-"):
        console.print("[yellow]Warning:[/yellow] Key doesn't start with 'sk-ant-'.")
        proceed = Prompt.ask("Save anyway?", choices=["y", "n"], default="n")
        if proceed.lower() != "y":
            raise typer.Exit(code=0)

    save_global_config("api_key", api_key)
    console.print("\n[green]API key saved.[/green]")


@app.command()
def rollback() -> None:
    """Undo the last agent change (git revert on nex/* branch)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        branch = result.stdout.strip()

        if not branch.startswith("nex/"):
            _error_exit(
                f"Current branch is '{branch}', not a nex/* branch.",
                hint="Rollback only works on nex/* branches.",
            )
            return

        log = subprocess.run(
            ["git", "log", "-1", "--oneline"],
            capture_output=True,
            text=True,
            check=False,
        )
        console.print(f"[bold]Last commit:[/bold] {log.stdout.strip()}")

        confirm = Prompt.ask("[bold]Revert?[/bold]", choices=["y", "n"], default="n")
        if confirm.lower() != "y":
            raise typer.Exit(code=0)

        revert = subprocess.run(
            ["git", "revert", "HEAD", "--no-edit"],
            capture_output=True,
            text=True,
            check=False,
        )
        if revert.returncode != 0:
            _error_exit(f"Revert failed: {revert.stderr.strip()}")
            return

        console.print("[green]Reverted last commit.[/green]")
    except FileNotFoundError:
        _error_exit("git not found.", hint="Install git.")


@app.command(name="memory")
def memory_cmd(
    action: Annotated[str, typer.Argument(help="Action: show, edit")] = "show",
) -> None:
    """View or edit project memory."""
    nex_dir = Path.cwd() / ".nex"
    if not nex_dir.is_dir():
        _error_exit("Project not initialized.", hint="Run 'nex init' first.")
        return

    memory_path = nex_dir / "memory.md"

    if action == "show":
        if not memory_path.is_file():
            _error_exit("No memory.md found.")
            return
        content = memory_path.read_text(encoding="utf-8")
        console.print(
            Panel(
                Markdown(content),
                title="[bold cyan]Project Memory[/bold cyan]",
                border_style="cyan",
            )
        )

    elif action == "edit":
        if not memory_path.is_file():
            _error_exit("No memory.md found.")
            return
        editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
        if editor:
            subprocess.run([editor, str(memory_path)], check=False)
        else:
            console.print(
                "[yellow]No $EDITOR set.[/yellow] Edit .nex/memory.md directly in your IDE."
            )
    else:
        _error_exit(f"Unknown action '{action}'.", hint="Valid: show, edit")


@app.command()
def chat(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose output")] = False,
) -> None:
    """Start an interactive chat session with Nex AI."""
    nex_dir = Path.cwd() / ".nex"
    if not nex_dir.is_dir():
        _error_exit("Project not initialized.", hint="Run 'nex init' first.")
        return

    try:
        config = load_config(Path.cwd())
    except NexError as exc:
        _error_exit(str(exc))
        return

    if verbose:
        config.log_level = "DEBUG"

    if not config.api_key:
        _error_exit(
            "Anthropic API key not found.",
            hint="Run 'nex auth' or set ANTHROPIC_API_KEY environment variable.",
        )
        return

    console.print(
        Panel(
            "Interactive chat mode. The agent remembers your conversation.\n"
            'Type [bold]"exit"[/bold] or [bold]"quit"[/bold] to end the session.',
            title=f"[bold cyan]Nex AI Chat[/bold cyan] v{__version__}",
            border_style="cyan",
        )
    )

    try:
        asyncio.run(_run_chat(config))
    except KeyboardInterrupt:
        console.print("\n[yellow]Chat ended.[/yellow]")


async def _run_chat(config: NexConfig) -> None:
    """Run the interactive chat REPL.

    Args:
        config: Nex configuration.
    """
    from nex.agent import ChatSession
    from nex.api_client import AnthropicClient
    from nex.context import ContextAssembler
    from nex.indexer.index import IndexBuilder
    from nex.memory.errors import ErrorPatternDB
    from nex.memory.project import ProjectMemory
    from nex.safety import SafetyLayer

    # Load context once at startup
    memory = ProjectMemory(config.project_dir)
    error_db = ErrorPatternDB(config.project_dir)
    assembler = ContextAssembler(config.project_dir)
    builder = IndexBuilder(config.project_dir)

    project_memory = memory.load()
    error_patterns = error_db.find_similar(task_summary="interactive chat session")
    idx = builder.load()
    relevant_code = assembler.select_relevant_code("interactive chat session", idx)

    system_prompt = assembler.build_system_prompt(
        project_memory=project_memory,
        error_patterns=error_patterns,
        relevant_code=relevant_code,
    )

    client = AnthropicClient(api_key=config.api_key, default_model=config.model)
    safety = SafetyLayer(dry_run=config.dry_run)

    session = ChatSession(
        api_client=client,
        system_prompt=system_prompt,
        project_dir=config.project_dir,
        safety=safety,
        dry_run=config.dry_run,
        max_iterations=config.max_iterations,
    )

    try:
        while True:
            try:
                user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
            except EOFError:
                break

            if not user_input.strip():
                continue
            if user_input.strip().lower() in ("exit", "quit"):
                break

            response = await session.send(user_input)
            if response:
                console.print(
                    Panel(
                        Markdown(response),
                        title="[bold green]Nex[/bold green]",
                        border_style="green",
                    )
                )
    finally:
        error_db.close()
        await client.close()

        console.print(
            f"\n[dim]Session: {session.turn_count} turns | "
            f"Cost: ${client.usage.estimated_cost:.4f} "
            f"({client.usage.total_input} in + {client.usage.total_output} out tokens)[/dim]"
        )
        console.print("[dim]Chat ended.[/dim]")
