"""Safety layer — destructive operation detection and user approval.

Detects rm -rf, DROP TABLE, force push, etc. and requires explicit user
confirmation before execution. In dry-run mode, dangerous actions are
always blocked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from nex.exceptions import SafetyError

console = Console()

_SENSITIVE_FILENAMES: frozenset[str] = frozenset(
    {
        ".env",
        ".env.local",
        ".env.production",
        ".env.development",
        "credentials.json",
        "service-account.json",
        "secrets.yaml",
        "secrets.yml",
        "id_rsa",
        "id_ed25519",
        ".npmrc",
        ".pypirc",
    }
)


@dataclass(frozen=True)
class SafetyCheck:
    """Result of a safety evaluation.

    Attributes:
        is_safe: True when the operation is allowed without prompting.
        reason: Explanation when the check fails.
        requires_approval: True when user must explicitly confirm.
    """

    is_safe: bool
    reason: str | None = None
    requires_approval: bool = False


class SafetyLayer:
    """Detects destructive operations and requires user approval."""

    DESTRUCTIVE_PATTERNS: ClassVar[list[tuple[str, str]]] = [
        (r"rm\s+-rf?\s+", "Recursive file deletion"),
        (r"rm\s+-r\s+/", "Deleting from root"),
        (r"DROP\s+TABLE", "SQL table drop"),
        (r"DROP\s+DATABASE", "SQL database drop"),
        (r"DELETE\s+FROM\s+\w+\s*;?\s*$", "SQL delete without WHERE"),
        (r"TRUNCATE\s+TABLE", "SQL table truncation"),
        (r"git\s+push\s+--force", "Force push"),
        (r"git\s+push\s+-f\s+", "Force push"),
        (r"git\s+clean\s+-[fd]", "Git clean"),
        (r"git\s+reset\s+--hard", "Hard reset"),
        (r"chmod\s+-R\s+777", "Insecure permissions"),
        (r">\s*/dev/sd[a-z]", "Writing to disk device"),
        (r"mkfs\.", "Formatting filesystem"),
        (r"dd\s+if=", "Direct disk operation"),
        (r"curl.*\|\s*bash", "Piping curl to bash"),
        (r"wget.*\|\s*bash", "Piping wget to bash"),
    ]

    _compiled: ClassVar[list[tuple[re.Pattern[str], str]] | None] = None

    def __init__(self, dry_run: bool = False) -> None:
        """Initialize the safety layer.

        Args:
            dry_run: If True, always block dangerous actions without prompting.
        """
        self._dry_run = dry_run

        if SafetyLayer._compiled is None:
            SafetyLayer._compiled = [
                (re.compile(pattern, re.IGNORECASE), label)
                for pattern, label in self.DESTRUCTIVE_PATTERNS
            ]

    def check_command(self, command: str) -> SafetyCheck:
        """Evaluate a shell command against destructive patterns.

        Args:
            command: The shell command to inspect.

        Returns:
            SafetyCheck indicating whether the command is safe.
        """
        assert self._compiled is not None
        for pattern, label in self._compiled:
            if pattern.search(command):
                return SafetyCheck(
                    is_safe=False,
                    reason=f"Destructive operation detected: {label}",
                    requires_approval=True,
                )
        return SafetyCheck(is_safe=True)

    def check_file_write(self, path: str, project_dir: Path) -> SafetyCheck:
        """Evaluate whether writing to path is allowed.

        Args:
            path: The file path to write to.
            project_dir: The project root directory.

        Returns:
            SafetyCheck with the evaluation result.
        """
        traversal = self.check_path_traversal(path, project_dir)
        if not traversal.is_safe:
            return traversal

        filename = Path(path).name.lower()
        if filename in _SENSITIVE_FILENAMES:
            return SafetyCheck(
                is_safe=False,
                reason=f"Writing to sensitive file '{filename}' which may contain secrets.",
                requires_approval=True,
            )

        suffix = Path(path).suffix.lower()
        if suffix in {".pem", ".key", ".p12", ".pfx"}:
            return SafetyCheck(
                is_safe=False,
                reason=f"Writing to cryptographic key file ('{suffix}').",
                requires_approval=True,
            )

        return SafetyCheck(is_safe=True)

    def check_path_traversal(self, path: str, project_dir: Path) -> SafetyCheck:
        """Detect path-traversal attempts escaping the project root.

        Args:
            path: Target file path.
            project_dir: Project root directory.

        Returns:
            SafetyCheck — unsafe if resolved path is outside project.
        """
        try:
            resolved = (project_dir / path).resolve()
            project_resolved = project_dir.resolve()
            if not str(resolved).startswith(str(project_resolved)):
                return SafetyCheck(
                    is_safe=False,
                    reason=f"Path traversal: '{path}' resolves outside project directory.",
                    requires_approval=False,
                )
        except (OSError, ValueError) as exc:
            return SafetyCheck(is_safe=False, reason=f"Invalid path '{path}': {exc}")

        return SafetyCheck(is_safe=True)

    async def request_approval(self, action: str, reason: str) -> bool:
        """Prompt user for explicit approval of a dangerous action.

        In dry-run mode, always returns False.

        Args:
            action: Description of the action.
            reason: Why it was flagged.

        Returns:
            True if user approves, False otherwise.
        """
        if self._dry_run:
            console.print(
                Panel(
                    Text.assemble(
                        ("BLOCKED (dry-run): ", "bold red"),
                        (action, "white"),
                        ("\nReason: ", "bold yellow"),
                        (reason, "yellow"),
                    ),
                    title="[bold red]Safety Layer[/bold red]",
                    border_style="red",
                )
            )
            return False

        console.print(
            Panel(
                Text.assemble(
                    ("Action: ", "bold white"),
                    (action, "white"),
                    ("\nReason: ", "bold yellow"),
                    (reason, "yellow"),
                ),
                title="[bold red]Dangerous Operation[/bold red]",
                border_style="red",
            )
        )

        answer = Prompt.ask("[bold]Allow?[/bold]", choices=["y", "n"], default="n")
        return answer.lower() == "y"

    async def guard_command(self, command: str) -> bool:
        """Check command and prompt for approval if needed.

        Args:
            command: Shell command to evaluate.

        Returns:
            True if command may proceed.

        Raises:
            SafetyError: If command is blocked.
        """
        check = self.check_command(command)
        if check.is_safe:
            return True

        if not check.requires_approval:
            raise SafetyError(check.reason or "Operation blocked")

        approved = await self.request_approval(command, check.reason or "Destructive operation")
        if not approved:
            raise SafetyError(f"User denied: {check.reason}")
        return True
