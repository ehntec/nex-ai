"""Async wrapper around the Anthropic API with retry logic and token tracking."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console

from nex.exceptions import APIError

console = Console(stderr=True)

# Pricing per million tokens (approximate)
_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0),
}


@dataclass
class APIResponse:
    """Structured response from the Anthropic API.

    Attributes:
        content: Content blocks from the API response.
        model: Model used for the response.
        input_tokens: Number of input tokens consumed.
        output_tokens: Number of output tokens generated.
        stop_reason: Why the response stopped (end_turn, tool_use, etc.).
    """

    content: list[dict[str, Any]]
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str | None


@dataclass
class TokenUsage:
    """Cumulative token usage across API calls.

    Attributes:
        total_input: Total input tokens consumed.
        total_output: Total output tokens generated.
    """

    total_input: int = 0
    total_output: int = 0

    @property
    def estimated_cost(self) -> float:
        """Estimate cost in USD based on Sonnet pricing."""
        input_cost = (self.total_input / 1_000_000) * 3.0
        output_cost = (self.total_output / 1_000_000) * 15.0
        return input_cost + output_cost


class AnthropicClient:
    """Async wrapper around the Anthropic API.

    Handles retries with exponential backoff for rate limits (429) and
    server errors (500+). Tracks cumulative token usage.

    Usage::

        client = AnthropicClient(api_key="sk-ant-...")
        response = await client.send_message(messages=[...], system="...")
        print(client.usage.estimated_cost)
        await client.close()
    """

    def __init__(
        self,
        api_key: str,
        default_model: str = "claude-sonnet-4-20250514",
        max_retries: int = 3,
    ) -> None:
        """Initialize the Anthropic client.

        Args:
            api_key: Anthropic API key.
            default_model: Default model for requests.
            max_retries: Maximum number of retries for transient errors.
        """
        from anthropic import AsyncAnthropic  # type: ignore[import-untyped]

        self._client = AsyncAnthropic(api_key=api_key)
        self._default_model = default_model
        self._max_retries = max_retries
        self._usage = TokenUsage()

    @property
    def usage(self) -> TokenUsage:
        """Return cumulative token usage."""
        return self._usage

    async def send_message(
        self,
        messages: list[dict[str, Any]],
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 8192,
    ) -> APIResponse:
        """Send a message to the Anthropic API.

        Args:
            messages: Conversation messages.
            system: System prompt.
            tools: Tool definitions for tool_use.
            model: Override default model.
            max_tokens: Maximum response tokens.

        Returns:
            Structured APIResponse.

        Raises:
            APIError: If the request fails after all retries.
        """
        use_model = model or self._default_model
        kwargs: dict[str, Any] = {
            "model": use_model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.messages.create(**kwargs)

                # Track usage
                self._usage.total_input += response.usage.input_tokens
                self._usage.total_output += response.usage.output_tokens

                # Convert content blocks to dicts
                content_dicts: list[dict[str, Any]] = []
                for block in response.content:
                    if block.type == "text":
                        content_dicts.append({"type": "text", "text": block.text})
                    elif block.type == "tool_use":
                        content_dicts.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        })

                return APIResponse(
                    content=content_dicts,
                    model=response.model,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    stop_reason=response.stop_reason,
                )

            except Exception as exc:  # noqa: BLE001
                last_error = exc
                status_code = getattr(exc, "status_code", None)

                # Retry on rate limit or server errors
                if status_code in (429, 500, 502, 503, 529) and attempt < self._max_retries:
                    wait = 2 ** attempt
                    retry_after = getattr(exc, "retry_after", None)
                    if retry_after:
                        wait = max(wait, float(retry_after))
                    console.print(
                        f"[yellow]API error {status_code}, retrying in {wait}s "
                        f"(attempt {attempt + 1}/{self._max_retries})...[/yellow]"
                    )
                    await asyncio.sleep(wait)
                    continue

                raise APIError(
                    f"Anthropic API error: {exc}",
                    status_code=status_code,
                ) from exc

        raise APIError(
            f"Failed after {self._max_retries} retries: {last_error}",
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()
