"""Async wrapper around the Anthropic API with retry logic and token tracking."""

from __future__ import annotations

import asyncio
import time
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


@dataclass
class RateLimiter:
    """Token-bucket rate limiter using a 60-second sliding window.

    Tracks timestamps and token counts of recent API calls. Before each call,
    checks whether the next request would exceed the configured token limit
    and sleeps if necessary.

    Attributes:
        tokens_per_minute: Maximum input tokens allowed per 60-second window.
            Set to 0 to disable rate limiting.
    """

    tokens_per_minute: int = 0
    _entries: list[tuple[float, int]] = field(default_factory=list, repr=False)

    @property
    def is_enabled(self) -> bool:
        """Return True if rate limiting is active."""
        return self.tokens_per_minute > 0

    def record(self, input_tokens: int) -> None:
        """Record a completed API call's actual input token count.

        Args:
            input_tokens: Actual input tokens consumed by the call.
        """
        self._entries.append((time.monotonic(), input_tokens))

    def tokens_in_window(self) -> int:
        """Return total input tokens consumed in the last 60 seconds."""
        cutoff = time.monotonic() - 60.0
        self._entries = [(t, n) for t, n in self._entries if t > cutoff]
        return sum(n for _, n in self._entries)

    async def wait_if_needed(self, estimated_tokens: int) -> None:
        """Sleep if the next call would exceed the rate limit.

        Args:
            estimated_tokens: Estimated input tokens for the upcoming call.
        """
        if not self.is_enabled:
            return

        while True:
            used = self.tokens_in_window()
            if used + estimated_tokens <= self.tokens_per_minute:
                return

            # Find the oldest entry and wait until it expires
            if self._entries:
                oldest_time = self._entries[0][0]
                wait = oldest_time + 60.0 - time.monotonic() + 0.1
                if wait > 0:
                    console.print(
                        f"[yellow]Rate limit: {used} tokens in window, "
                        f"waiting {wait:.1f}s...[/yellow]"
                    )
                    await asyncio.sleep(wait)
            else:
                return


def _extract_retry_after(exc: Exception, default: float = 60.0) -> float:
    """Extract retry-after seconds from an Anthropic API error.

    Inspects the exception's response headers for a ``retry-after`` value.
    Falls back to *default* if the header is missing or unparseable.

    Args:
        exc: The exception raised by the Anthropic SDK.
        default: Fallback wait time in seconds.

    Returns:
        Number of seconds to wait before retrying.
    """
    resp = getattr(exc, "response", None)
    if resp is not None:
        header = getattr(resp, "headers", {}).get("retry-after")
        if header:
            try:
                return max(float(header), 1.0)
            except (ValueError, TypeError):
                pass
    return default


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
        max_retries: int = 5,
    ) -> None:
        """Initialize the Anthropic client.

        Args:
            api_key: Anthropic API key.
            default_model: Default model for requests.
            max_retries: Maximum number of retries for transient errors.
        """
        from anthropic import AsyncAnthropic

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
        # 429 rate limits need more retries with longer waits than server errors
        max_attempts = self._max_retries + 1
        for attempt in range(max_attempts):
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
                        content_dicts.append(
                            {
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": block.input,
                            }
                        )

                return APIResponse(
                    content=content_dicts,
                    model=response.model,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    stop_reason=response.stop_reason,
                )

            except Exception as exc:
                last_error = exc
                status_code = getattr(exc, "status_code", None)
                is_rate_limit = status_code == 429
                is_retryable = status_code in (500, 502, 503, 529)

                if (is_rate_limit or is_retryable) and attempt < max_attempts - 1:
                    if is_rate_limit:
                        # Rate limits: extract retry-after from response headers,
                        # or default to 60s (the full rate-limit window).
                        wait = _extract_retry_after(exc, default=60.0)
                    else:
                        # Server errors: short exponential backoff
                        wait = float(2**attempt)

                    console.print(
                        f"[yellow]API error {status_code}, retrying in {wait:.0f}s "
                        f"(attempt {attempt + 1}/{max_attempts - 1})...[/yellow]"
                    )
                    await asyncio.sleep(wait)
                    continue

                raise APIError(
                    f"Anthropic API error: {exc}",
                    status_code=status_code,
                ) from exc

        raise APIError(
            f"Failed after {max_attempts - 1} retries: {last_error}",
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()
