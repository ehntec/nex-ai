"""Tests for the core agent loop."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nex.agent import Agent, AgentConfig
from nex.api_client import APIResponse
from nex.safety import SafetyLayer


@pytest.fixture
def agent_config(tmp_path: Path) -> AgentConfig:
    nex_dir = tmp_path / ".nex"
    nex_dir.mkdir()
    (nex_dir / "memory.md").write_text("# Test\n", encoding="utf-8")
    return AgentConfig(
        project_dir=tmp_path,
        task="add a hello function",
        max_iterations=3,
    )


@pytest.fixture
def text_response() -> APIResponse:
    return APIResponse(
        content=[{"type": "text", "text": "Done! I added the hello function."}],
        model="claude-sonnet-4-20250514",
        input_tokens=100,
        output_tokens=50,
        stop_reason="end_turn",
    )


@pytest.fixture
def tool_then_text_responses(tmp_path: Path) -> list[APIResponse]:
    return [
        APIResponse(
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_001",
                    "name": "read_file",
                    "input": {"path": "main.py"},
                }
            ],
            model="claude-sonnet-4-20250514",
            input_tokens=100,
            output_tokens=50,
            stop_reason="tool_use",
        ),
        APIResponse(
            content=[{"type": "text", "text": "Task complete."}],
            model="claude-sonnet-4-20250514",
            input_tokens=150,
            output_tokens=60,
            stop_reason="end_turn",
        ),
    ]


class TestAgent:
    @pytest.mark.asyncio
    async def test_simple_text_response(
        self, agent_config: AgentConfig, text_response: APIResponse
    ) -> None:
        client = MagicMock()
        client.send_message = AsyncMock(return_value=text_response)
        client.close = AsyncMock()
        client.usage = MagicMock()
        client.usage.estimated_cost = 0.001

        safety = SafetyLayer(dry_run=False)
        agent = Agent(config=agent_config, api_client=client, safety=safety)
        result = await agent.run()

        assert "hello function" in result.lower() or "done" in result.lower()
        assert client.send_message.call_count == 1

    @pytest.mark.asyncio
    async def test_tool_use_then_complete(
        self,
        agent_config: AgentConfig,
        tool_then_text_responses: list[APIResponse],
    ) -> None:
        client = MagicMock()
        client.send_message = AsyncMock(side_effect=tool_then_text_responses)
        client.close = AsyncMock()
        client.usage = MagicMock()
        client.usage.estimated_cost = 0.002

        # Create the file the tool will read
        (agent_config.project_dir / "main.py").write_text("print('hi')", encoding="utf-8")

        safety = SafetyLayer(dry_run=False)
        agent = Agent(config=agent_config, api_client=client, safety=safety)
        result = await agent.run()

        assert client.send_message.call_count == 2

    @pytest.mark.asyncio
    async def test_max_iterations(self, agent_config: AgentConfig) -> None:
        """Agent should stop after max_iterations."""
        tool_response = APIResponse(
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_loop",
                    "name": "list_directory",
                    "input": {"path": "."},
                }
            ],
            model="claude-sonnet-4-20250514",
            input_tokens=50,
            output_tokens=30,
            stop_reason="tool_use",
        )

        client = MagicMock()
        client.send_message = AsyncMock(return_value=tool_response)
        client.close = AsyncMock()
        client.usage = MagicMock()
        client.usage.estimated_cost = 0.01

        safety = SafetyLayer(dry_run=False)
        agent = Agent(config=agent_config, api_client=client, safety=safety)
        result = await agent.run()

        # Should have stopped after max_iterations (3)
        assert client.send_message.call_count == agent_config.max_iterations
