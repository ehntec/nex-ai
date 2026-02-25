"""Configuration management for Nex AI.

Settings are loaded from three sources in order of priority:
1. Environment variables (highest priority)
2. Project-level config: .nex/config.toml
3. Global config: ~/.config/nex/config.toml (lowest priority)
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console

from nex.exceptions import ConfigError

console = Console(stderr=True)

_GLOBAL_CONFIG_DIR = Path.home() / ".config" / "nex"
_GLOBAL_CONFIG_PATH = _GLOBAL_CONFIG_DIR / "config.toml"


@dataclass
class NexConfig:
    """Nex AI configuration.

    Attributes:
        api_key: Anthropic API key.
        model: Default model for code generation and reasoning.
        haiku_model: Model for planning and lightweight tasks.
        max_iterations: Maximum tool calls per task.
        dry_run: If True, show planned actions without executing.
        log_level: Logging verbosity (DEBUG, INFO, WARNING, ERROR).
        nex_dir: Path to the .nex directory (relative to project root).
        test_command: Override for auto-detected test command (empty = auto-detect).
        test_timeout: Maximum seconds to wait for test suite to complete.
        token_rate_limit: Max input tokens per minute. 0 = no rate limiting.
        subtask_token_budget: Max tokens per subtask context window.
    """

    project_dir: Path = field(default_factory=Path.cwd)
    api_key: str = ""
    model: str = "claude-sonnet-4-20250514"
    haiku_model: str = "claude-haiku-4-5-20251001"
    max_iterations: int = 25
    dry_run: bool = False
    log_level: str = "INFO"
    nex_dir: Path = field(default_factory=lambda: Path(".nex"))
    test_command: str = ""
    test_timeout: int = 120
    token_rate_limit: int = 20_000
    subtask_token_budget: int = 20_000


def load_config(project_dir: Path) -> NexConfig:
    """Load configuration from env vars, project config, and global config.

    Priority: env vars > .nex/config.toml > ~/.config/nex/config.toml

    Args:
        project_dir: Root directory of the project.

    Returns:
        A fully resolved NexConfig instance.

    Raises:
        ConfigError: If the API key is not set anywhere.
    """
    config = NexConfig(project_dir=project_dir)

    # Layer 1: Global config (lowest priority)
    global_settings = _load_toml(_GLOBAL_CONFIG_PATH)
    _apply_toml(config, global_settings)

    # Layer 2: Project config
    project_config_path = project_dir / ".nex" / "config.toml"
    project_settings = _load_toml(project_config_path)
    _apply_toml(config, project_settings)

    # Layer 3: Environment variables (highest priority)
    _apply_env(config)

    return config


def ensure_api_key(config: NexConfig) -> None:
    """Validate that an API key is configured.

    Args:
        config: The configuration to validate.

    Raises:
        ConfigError: If api_key is empty.
    """
    if not config.api_key:
        raise ConfigError(
            "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable "
            "or run 'nex auth' to configure it."
        )


def save_global_config(key: str, value: str) -> None:
    """Save a key-value pair to the global config file.

    Args:
        key: Configuration key (e.g. "api_key").
        value: Configuration value.
    """
    _GLOBAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    settings: dict[str, Any] = {}
    if _GLOBAL_CONFIG_PATH.is_file():
        settings = _load_toml(_GLOBAL_CONFIG_PATH)

    settings[key] = value

    lines = [f'{k} = "{v}"' for k, v in settings.items()]
    _GLOBAL_CONFIG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    console.print(f"[green]Saved[/green] {key} to {_GLOBAL_CONFIG_PATH}")


def _load_toml(path: Path) -> dict[str, Any]:
    """Load a TOML file, returning an empty dict if missing or invalid."""
    if not path.is_file():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError) as exc:
        console.print(f"[yellow]Warning:[/yellow] Could not parse {path}: {exc}")
        return {}


def _apply_toml(config: NexConfig, settings: dict[str, Any]) -> None:
    """Merge TOML settings into a NexConfig (only set non-empty values)."""
    if "api_key" in settings:
        config.api_key = str(settings["api_key"])
    if "model" in settings:
        config.model = str(settings["model"])
    if "haiku_model" in settings:
        config.haiku_model = str(settings["haiku_model"])
    if "max_iterations" in settings:
        config.max_iterations = int(settings["max_iterations"])
    if "dry_run" in settings:
        config.dry_run = bool(settings["dry_run"])
    if "log_level" in settings:
        config.log_level = str(settings["log_level"]).upper()
    if "test_command" in settings:
        config.test_command = str(settings["test_command"])
    if "test_timeout" in settings:
        config.test_timeout = int(settings["test_timeout"])
    if "token_rate_limit" in settings:
        config.token_rate_limit = int(settings["token_rate_limit"])
    if "subtask_token_budget" in settings:
        config.subtask_token_budget = int(settings["subtask_token_budget"])


def _apply_env(config: NexConfig) -> None:
    """Override config with environment variables where set."""
    if api_key := os.environ.get("ANTHROPIC_API_KEY"):
        config.api_key = api_key
    if model := os.environ.get("NEX_MODEL"):
        config.model = model
    if max_iter := os.environ.get("NEX_MAX_ITERATIONS"):
        config.max_iterations = int(max_iter)
    if dry_run := os.environ.get("NEX_DRY_RUN"):
        config.dry_run = dry_run.lower() in ("true", "1", "yes")
    if log_level := os.environ.get("NEX_LOG_LEVEL"):
        config.log_level = log_level.upper()
    if test_cmd := os.environ.get("NEX_TEST_COMMAND"):
        config.test_command = test_cmd
    if test_timeout := os.environ.get("NEX_TEST_TIMEOUT"):
        config.test_timeout = int(test_timeout)
    if token_rate := os.environ.get("NEX_TOKEN_RATE_LIMIT"):
        config.token_rate_limit = int(token_rate)
    if subtask_budget := os.environ.get("NEX_SUBTASK_TOKEN_BUDGET"):
        config.subtask_token_budget = int(subtask_budget)
