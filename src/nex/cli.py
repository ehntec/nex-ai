"""Typer CLI entry point for Nex AI.

Bridges the synchronous Typer world to the async agent internals via asyncio.run().
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
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
from nex.exceptions import ConfigError, NexError

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

        console.print(Panel(
            f"[bold]Task:[/bold] {task}",
            title=f"[bold cyan]Nex AI[/bold cyan] v{__version__}",
            border_style="cyan",
        ))

        if config.dry_run:
            console.print("[yellow]Running in dry-run mode.[/yellow]\n")

        from nex.agent import run_task
        asyncio.run(run_task(task, config))

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(code=130)
    except NexError as exc:
        _error_exit(str(exc))
    except Exception as exc:  # noqa: BLE001
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

    console.print(Panel(
        Text.assemble(
            ("Initialized Nex AI in ", "green"),
            (str(nex_dir), "bold green"),
            ("\n\nCreated:\n", "white"),
            ("  memory.md     ", "cyan"), ("- project memory (edit this!)\n", "dim"),
            ("  decisions.md  ", "cyan"), ("- decision log\n", "dim"),
            ("  config.toml   ", "cyan"), ("- project settings\n", "dim"),
            ("  .gitignore    ", "cyan"), ("- ignores errors.db, index.json", "dim"),
        ),
        title="[bold green]Project Initialized[/bold green]",
        border_style="green",
    ))

    console.print(
        "\n[dim]Next steps:[/dim]\n"
        "  1. Edit [cyan].nex/memory.md[/cyan] with your project details\n"
        "  2. Run [cyan]nex auth[/cyan] to configure your API key\n"
        "  3. Run [cyan]nex \"your first task\"[/cyan]\n"
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
                f"{len(data.get('files', []))} files, "
                f"{len(data.get('symbols', []))} symbols",
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
    console.print(Panel(
        "Configure your Anthropic API key.\n"
        "Get one at [link=https://console.anthropic.com/]console.anthropic.com[/link]",
        title="[bold cyan]API Key Setup[/bold cyan]",
        border_style="cyan",
    ))

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
            capture_output=True, text=True, check=False,
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
            capture_output=True, text=True, check=False,
        )
        console.print(f"[bold]Last commit:[/bold] {log.stdout.strip()}")

        confirm = Prompt.ask("[bold]Revert?[/bold]", choices=["y", "n"], default="n")
        if confirm.lower() != "y":
            raise typer.Exit(code=0)

        revert = subprocess.run(
            ["git", "revert", "HEAD", "--no-edit"],
            capture_output=True, text=True, check=False,
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
        console.print(Panel(Markdown(content), title="[bold cyan]Project Memory[/bold cyan]", border_style="cyan"))

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
