"""Tests for the RateLimiter class."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from nex.api_client import RateLimiter


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_no_wait_under_limit(self) -> None:
        """Should not wait when tokens in window are under the limit."""
        limiter = RateLimiter(tokens_per_minute=30_000)
        limiter.record(10_000)

        # 10K recorded + 15K requested = 25K < 30K — no wait needed
        start = time.monotonic()
        await limiter.wait_if_needed(15_000)
        elapsed = time.monotonic() - start

        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_wait_when_over_limit(self) -> None:
        """Should wait when adding estimated tokens would exceed the limit."""
        limiter = RateLimiter(tokens_per_minute=30_000)
        limiter.record(25_000)

        # 25K recorded + 10K requested = 35K > 30K — must wait
        # Mock asyncio.sleep so we don't actually wait
        with patch("nex.api_client.asyncio.sleep") as mock_sleep:
            # After sleep, tokens_in_window should return 0 (entries expired)
            original_tokens = limiter.tokens_in_window

            call_count = 0

            def side_effect(duration: float) -> None:
                nonlocal call_count
                call_count += 1
                # Simulate time passing: clear entries to break the loop
                limiter._entries.clear()

            mock_sleep.side_effect = side_effect
            await limiter.wait_if_needed(10_000)

            assert mock_sleep.called

    @pytest.mark.asyncio
    async def test_disabled_when_zero(self) -> None:
        """Should never wait when limit is 0 (disabled)."""
        limiter = RateLimiter(tokens_per_minute=0)

        assert not limiter.is_enabled
        limiter.record(999_999)

        start = time.monotonic()
        await limiter.wait_if_needed(999_999)
        elapsed = time.monotonic() - start

        assert elapsed < 0.1

    def test_window_expiry(self) -> None:
        """Old entries should expire after 60 seconds."""
        limiter = RateLimiter(tokens_per_minute=30_000)

        # Record an entry "65 seconds ago" by manipulating internal state
        old_time = time.monotonic() - 65.0
        limiter._entries.append((old_time, 20_000))

        # Also add a recent entry
        limiter.record(5_000)

        # Only the recent 5K should be in the window
        assert limiter.tokens_in_window() == 5_000

    def test_is_enabled(self) -> None:
        """is_enabled should reflect the tokens_per_minute setting."""
        assert RateLimiter(tokens_per_minute=30_000).is_enabled
        assert not RateLimiter(tokens_per_minute=0).is_enabled

    def test_record_and_tokens_in_window(self) -> None:
        """record() should add entries and tokens_in_window() should sum them."""
        limiter = RateLimiter(tokens_per_minute=100_000)
        limiter.record(1_000)
        limiter.record(2_000)
        limiter.record(3_000)

        assert limiter.tokens_in_window() == 6_000
