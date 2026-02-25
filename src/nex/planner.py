"""Task decomposition using Claude Haiku."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from rich.console import Console

from nex.api_client import AnthropicClient, RateLimiter

console = Console(stderr=True)

_PLANNER_SYSTEM_PROMPT = """\
You are a task planner for a coding agent. Given a task description and project context, \
decompose the task into a list of concrete subtasks.

Respond with a JSON array of objects, each with:
- "description": what to do (be specific)
- "file_paths": list of files likely to be touched
- "priority": 1 = do first, 2 = do second, etc.

Keep the list concise (3-7 subtasks). Order by dependency — things that must happen first \
get lower priority numbers.

Respond ONLY with the JSON array, no other text.
"""


@dataclass
class Subtask:
    """A single subtask from the planner.

    Attributes:
        description: What to do.
        file_paths: Expected files to touch.
        priority: 1 = highest priority.
    """

    description: str
    file_paths: list[str]
    priority: int


class Planner:
    """Decomposes a task into subtasks using Claude Haiku."""

    def __init__(
        self,
        api_client: AnthropicClient,
        haiku_model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        """Initialize the planner.

        Args:
            api_client: The Anthropic API client.
            haiku_model: Model to use for planning.
        """
        self._client = api_client
        self._model = haiku_model

    async def plan(
        self,
        task: str,
        project_context: str,
        rate_limiter: RateLimiter | None = None,
    ) -> list[Subtask]:
        """Decompose a task into subtasks.

        Args:
            task: The user's task description.
            project_context: Project memory and relevant context.
            rate_limiter: Optional rate limiter to pace the API call.

        Returns:
            List of Subtask instances sorted by priority.
        """
        user_content = f"Project context:\n{project_context}\n\nTask: {task}"
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": user_content,
            }
        ]

        console.print("[dim]Planning task decomposition...[/dim]")

        if rate_limiter is not None:
            estimated = len(user_content) // 4
            await rate_limiter.wait_if_needed(estimated)

        response = await self._client.send_message(
            messages=messages,
            system=_PLANNER_SYSTEM_PROMPT,
            model=self._model,
            max_tokens=2048,
        )

        if rate_limiter is not None:
            rate_limiter.record(response.input_tokens)

        # Extract text response
        text = ""
        for block in response.content:
            if block.get("type") == "text":
                text = block.get("text", "")
                break

        # Parse JSON
        try:
            # Handle markdown code fences
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]

            raw: list[dict[str, Any]] = json.loads(text.strip())
            subtasks = [
                Subtask(
                    description=item.get("description", ""),
                    file_paths=item.get("file_paths", []),
                    priority=item.get("priority", 99),
                )
                for item in raw
                if isinstance(item, dict)
            ]
            subtasks.sort(key=lambda s: s.priority)
            console.print(f"[green]Planned[/green] {len(subtasks)} subtasks")
            return subtasks

        except (json.JSONDecodeError, KeyError, TypeError):
            console.print(
                "[yellow]Warning[/yellow]: Could not parse plan, proceeding without decomposition"
            )
            return [Subtask(description=task, file_paths=[], priority=1)]
