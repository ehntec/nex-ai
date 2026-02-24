"""Append-only decision log stored in .nex/decisions.md.

Every architectural or non-trivial implementation choice the agent makes
is recorded here so future sessions understand why the codebase looks
the way it does.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from rich.console import Console

from nex.exceptions import NexMemoryError

console: Final[Console] = Console(stderr=True)

_HEADER: Final[str] = "# Decision Log\n\n"

_ENTRY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^### \[(?P<timestamp>[^\]]+)\] — Task: (?P<task>.+)\n"
    r"\*\*Decision:\*\* (?P<decision>.+)\n"
    r"\*\*Reasoning:\*\* (?P<reasoning>.+)\n"
    r"---$",
    re.MULTILINE,
)


@dataclass
class Decision:
    """A single decision record.

    Attributes:
        timestamp: ISO-8601 datetime string.
        task: The task context in which the decision was made.
        decision: What was decided.
        reasoning: Why this choice was made over alternatives.
    """

    timestamp: str
    task: str
    decision: str
    reasoning: str


class DecisionLog:
    """Append-only decision log stored in .nex/decisions.md."""

    def __init__(self, project_dir: Path) -> None:
        """Initialize the decision log.

        Args:
            project_dir: Project root containing the .nex folder.
        """
        self._project_dir = project_dir

    @property
    def log_path(self) -> Path:
        """Return the absolute path to the decisions file."""
        return self._project_dir / ".nex" / "decisions.md"

    def log(self, task: str, decision: str, reasoning: str) -> None:
        """Append a new decision entry to the log.

        Args:
            task: The task context (e.g. "add user registration").
            decision: What was decided.
            reasoning: Why this option was chosen.

        Raises:
            NexMemoryError: If the file cannot be written.
        """
        timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        entry = (
            f"### [{timestamp}] \u2014 Task: {task}\n"
            f"**Decision:** {decision}\n"
            f"**Reasoning:** {reasoning}\n"
            f"---\n\n"
        )

        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.log_path.is_file():
                self.log_path.write_text(_HEADER + entry, encoding="utf-8")
            else:
                with self.log_path.open("a", encoding="utf-8") as fh:
                    fh.write(entry)
        except OSError as exc:
            raise NexMemoryError(f"Failed to write decision log at {self.log_path}: {exc}") from exc

    def load(self) -> list[Decision]:
        """Parse all decision entries from the log file.

        Returns:
            A list of Decision instances in chronological order.

        Raises:
            NexMemoryError: If the file exists but cannot be read.
        """
        if not self.log_path.is_file():
            return []
        try:
            text = self.log_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise NexMemoryError(f"Failed to read decision log at {self.log_path}: {exc}") from exc

        return [
            Decision(
                timestamp=m.group("timestamp"),
                task=m.group("task"),
                decision=m.group("decision"),
                reasoning=m.group("reasoning"),
            )
            for m in _ENTRY_PATTERN.finditer(text)
        ]

    def recent(self, n: int = 5) -> list[Decision]:
        """Return the last n decisions.

        Args:
            n: Number of recent decisions to retrieve.

        Returns:
            Up to n Decision instances, most recent last.
        """
        all_decisions = self.load()
        return all_decisions[-n:]
