"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from typing import Any

import pytest

from nex.api_client import APIResponse, AnthropicClient
from nex.config import NexConfig


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Create a temporary project directory with .nex/ initialized."""
    nex_dir = tmp_path / ".nex"
    nex_dir.mkdir()
    (nex_dir / "memory.md").write_text("# Test Project\n", encoding="utf-8")
    (nex_dir / "decisions.md").write_text("# Decision Log\n\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def mock_api_response() -> APIResponse:
    """Create a mock API response with text content."""
    return APIResponse(
        content=[{"type": "text", "text": "Task completed successfully."}],
        model="claude-sonnet-4-20250514",
        input_tokens=100,
        output_tokens=50,
        stop_reason="end_turn",
    )


@pytest.fixture
def mock_tool_response() -> APIResponse:
    """Create a mock API response with a tool_use block."""
    return APIResponse(
        content=[{
            "type": "tool_use",
            "id": "toolu_test123",
            "name": "read_file",
            "input": {"path": "test.py"},
        }],
        model="claude-sonnet-4-20250514",
        input_tokens=100,
        output_tokens=50,
        stop_reason="tool_use",
    )


@pytest.fixture
def mock_api_client(mock_api_response: APIResponse) -> MagicMock:
    """Create a mock AnthropicClient that returns a text response."""
    client = MagicMock(spec=AnthropicClient)
    client.send_message = AsyncMock(return_value=mock_api_response)
    client.close = AsyncMock()
    client.usage = MagicMock()
    client.usage.estimated_cost = 0.001
    client.usage.total_input = 100
    client.usage.total_output = 50
    return client


@pytest.fixture
def test_config(tmp_project: Path) -> NexConfig:
    """Create a test NexConfig pointing at tmp_project."""
    return NexConfig(
        api_key="sk-ant-test-key",
        model="claude-sonnet-4-20250514",
        max_iterations=5,
        dry_run=False,
        log_level="DEBUG",
    )
