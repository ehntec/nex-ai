"""Independent code reviewer — separate API call, no access to the plan."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from rich.console import Console

from nex.api_client import AnthropicClient

console = Console(stderr=True)

_REVIEWER_SYSTEM_PROMPT = """\
You are an independent code reviewer. You will be shown a git diff of changes made by a \
coding agent. Review the diff for:

1. Correctness: Does the code do what the task asks?
2. Bugs: Any obvious logic errors, off-by-one errors, null/undefined issues?
3. Security: Any injection vulnerabilities, hardcoded secrets, unsafe operations?
4. Style: Does it match the project's existing conventions?
5. Tests: Were tests added or updated for the changes?

Respond with a JSON object:
{
    "approved": true/false,
    "issues": ["list of problems found"],
    "suggestions": ["list of improvement suggestions"]
}

Be concise. Only flag real issues, not style nitpicks. If the code looks good, approve it.
Respond ONLY with the JSON object, no other text.
"""


@dataclass
class ReviewResult:
    """Result of an independent code review.

    Attributes:
        approved: Whether the changes pass review.
        issues: List of problems found.
        suggestions: List of improvement suggestions.
    """

    approved: bool
    issues: list[str]
    suggestions: list[str]


class Reviewer:
    """Independent code reviewer using a separate API call."""

    def __init__(self, api_client: AnthropicClient) -> None:
        """Initialize the reviewer.

        Args:
            api_client: The Anthropic API client.
        """
        self._client = api_client

    async def review(self, diff: str, task: str) -> ReviewResult:
        """Review a git diff against the original task.

        Args:
            diff: The git diff to review.
            task: The original task description.

        Returns:
            Structured ReviewResult.
        """
        if not diff.strip():
            return ReviewResult(approved=True, issues=[], suggestions=[])

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": f"Original task: {task}\n\nGit diff:\n```\n{diff}\n```",
            }
        ]

        console.print("[dim]Reviewing changes...[/dim]")

        response = await self._client.send_message(
            messages=messages,
            system=_REVIEWER_SYSTEM_PROMPT,
            max_tokens=2048,
        )

        text = ""
        for block in response.content:
            if block.get("type") == "text":
                text = block.get("text", "")
                break

        try:
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]

            raw: dict[str, Any] = json.loads(text.strip())
            return ReviewResult(
                approved=bool(raw.get("approved", False)),
                issues=raw.get("issues", []),
                suggestions=raw.get("suggestions", []),
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            console.print("[yellow]Warning[/yellow]: Could not parse review, assuming approval")
            return ReviewResult(approved=True, issues=[], suggestions=[])
