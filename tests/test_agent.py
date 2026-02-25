"""Tests for the core agent loop."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nex.agent import Agent, AgentConfig
from nex.api_client import APIResponse
from nex.config import NexConfig
from nex.safety import SafetyLayer


@pytest.fixture
def agent_config(tmp_path: Path) -> AgentConfig:
    nex_dir = tmp_path / ".nex"
    nex_dir.mkdir()
    (nex_dir / "memory.md").write_text("# Test\n", encoding="utf-8")
    return AgentConfig(
        project_dir=tmp_path,
        task="add a hello function",
        max_iterations=10,
    )


@pytest.fixture
def planner_response() -> APIResponse:
    """Planner returns a single subtask."""
    return APIResponse(
        content=[
            {
                "type": "text",
                "text": (
                    '[{"description": "Add a hello function", "file_paths": [], "priority": 1}]'
                ),
            }
        ],
        model="claude-haiku-4-5-20251001",
        input_tokens=200,
        output_tokens=100,
        stop_reason="end_turn",
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
def memory_update_response() -> APIResponse:
    """Haiku summary for memory update."""
    return APIResponse(
        content=[{"type": "text", "text": "Added hello function to main.py."}],
        model="claude-haiku-4-5-20251001",
        input_tokens=50,
        output_tokens=20,
        stop_reason="end_turn",
    )


@pytest.fixture
def tool_then_text_responses() -> list[APIResponse]:
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
        self,
        agent_config: AgentConfig,
        planner_response: APIResponse,
        text_response: APIResponse,
        memory_update_response: APIResponse,
    ) -> None:
        client = MagicMock()
        client.send_message = AsyncMock(
            side_effect=[planner_response, text_response, memory_update_response]
        )
        client.close = AsyncMock()
        client.usage = MagicMock()
        client.usage.estimated_cost = 0.001

        safety = SafetyLayer(dry_run=False)
        agent = Agent(config=agent_config, api_client=client, safety=safety)
        result = await agent.run()

        assert "hello function" in result.lower() or "done" in result.lower()
        # planner + subtask + memory update
        assert client.send_message.call_count == 3

    @pytest.mark.asyncio
    async def test_tool_use_then_complete(
        self,
        agent_config: AgentConfig,
        planner_response: APIResponse,
        tool_then_text_responses: list[APIResponse],
        memory_update_response: APIResponse,
    ) -> None:
        client = MagicMock()
        client.send_message = AsyncMock(
            side_effect=[
                planner_response,
                *tool_then_text_responses,
                memory_update_response,
            ]
        )
        client.close = AsyncMock()
        client.usage = MagicMock()
        client.usage.estimated_cost = 0.002

        # Create the file the tool will read
        (agent_config.project_dir / "main.py").write_text("print('hi')", encoding="utf-8")

        safety = SafetyLayer(dry_run=False)
        agent = Agent(config=agent_config, api_client=client, safety=safety)
        await agent.run()

        # planner + 2 subtask iterations + memory update
        assert client.send_message.call_count == 4

    @pytest.mark.asyncio
    async def test_max_iterations(
        self,
        agent_config: AgentConfig,
        planner_response: APIResponse,
        memory_update_response: APIResponse,
    ) -> None:
        """Subtask should stop after its iteration budget."""
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

        # max_iterations=10, 1 subtask -> iters_per_subtask = max(5, 10) = 10
        client = MagicMock()
        client.send_message = AsyncMock(
            side_effect=[
                planner_response,
                *([tool_response] * 10),
                memory_update_response,
            ]
        )
        client.close = AsyncMock()
        client.usage = MagicMock()
        client.usage.estimated_cost = 0.01

        safety = SafetyLayer(dry_run=False)
        agent = Agent(config=agent_config, api_client=client, safety=safety)
        await agent.run()

        # planner + 10 subtask iterations + memory update
        assert client.send_message.call_count == 12

    @pytest.mark.asyncio
    async def test_run_with_subtasks_calls_planner(self, tmp_path: Path) -> None:
        """Agent should always use the planner path."""
        nex_dir = tmp_path / ".nex"
        nex_dir.mkdir()
        (nex_dir / "memory.md").write_text("# Test\n", encoding="utf-8")

        agent_cfg = AgentConfig(
            project_dir=tmp_path,
            task="add a hello function",
            max_iterations=10,
        )
        nex_cfg = NexConfig(
            project_dir=tmp_path,
            api_key="test-key",
            token_rate_limit=30_000,
            subtask_token_budget=20_000,
        )

        planner_resp = APIResponse(
            content=[
                {
                    "type": "text",
                    "text": (
                        '[{"description": "Create hello function",'
                        ' "file_paths": [], "priority": 1}]'
                    ),
                }
            ],
            model="claude-haiku-4-5-20251001",
            input_tokens=200,
            output_tokens=100,
            stop_reason="end_turn",
        )

        subtask_resp = APIResponse(
            content=[{"type": "text", "text": "Done! Created hello function."}],
            model="claude-sonnet-4-20250514",
            input_tokens=500,
            output_tokens=50,
            stop_reason="end_turn",
        )

        mem_resp = APIResponse(
            content=[{"type": "text", "text": "Created hello function."}],
            model="claude-haiku-4-5-20251001",
            input_tokens=50,
            output_tokens=20,
            stop_reason="end_turn",
        )

        client = MagicMock()
        client.send_message = AsyncMock(side_effect=[planner_resp, subtask_resp, mem_resp])
        client.close = AsyncMock()
        client.usage = MagicMock()
        client.usage.estimated_cost = 0.001

        safety = SafetyLayer(dry_run=False)
        agent = Agent(
            config=agent_cfg,
            api_client=client,
            safety=safety,
            nex_config=nex_cfg,
        )
        result = await agent.run()

        # planner + subtask + memory update
        assert client.send_message.call_count == 3
        assert "hello function" in result.lower() or "done" in result.lower()

    @pytest.mark.asyncio
    async def test_always_uses_subtasks_even_without_rate_limit(
        self,
        agent_config: AgentConfig,
        planner_response: APIResponse,
        text_response: APIResponse,
        memory_update_response: APIResponse,
    ) -> None:
        """Even with token_rate_limit=0, the planner path is used."""
        nex_cfg = NexConfig(
            project_dir=agent_config.project_dir,
            api_key="test-key",
            token_rate_limit=0,
        )

        client = MagicMock()
        client.send_message = AsyncMock(
            side_effect=[planner_response, text_response, memory_update_response]
        )
        client.close = AsyncMock()
        client.usage = MagicMock()
        client.usage.estimated_cost = 0.001

        safety = SafetyLayer(dry_run=False)
        agent = Agent(
            config=agent_config,
            api_client=client,
            safety=safety,
            nex_config=nex_cfg,
        )
        result = await agent.run()

        # planner + subtask + memory update = 3 (proves planner is always used)
        assert client.send_message.call_count == 3
        assert "hello function" in result.lower() or "done" in result.lower()

    @pytest.mark.asyncio
    async def test_planner_failure_falls_back_to_single(
        self, agent_config: AgentConfig, text_response: APIResponse
    ) -> None:
        """If planner fails, agent falls back to _run_single."""
        client = MagicMock()
        client.send_message = AsyncMock(side_effect=[Exception("Planner API error"), text_response])
        client.close = AsyncMock()
        client.usage = MagicMock()
        client.usage.estimated_cost = 0.001

        safety = SafetyLayer(dry_run=False)
        agent = Agent(config=agent_config, api_client=client, safety=safety)
        result = await agent.run()

        # First call fails (planner), second succeeds (single loop)
        assert client.send_message.call_count == 2
        assert "hello function" in result.lower() or "done" in result.lower()

    @pytest.mark.asyncio
    async def test_memory_update_after_subtask(
        self,
        agent_config: AgentConfig,
        planner_response: APIResponse,
        text_response: APIResponse,
    ) -> None:
        """Session Log should be written to memory.md after each subtask."""
        mem_resp = APIResponse(
            content=[{"type": "text", "text": "Added hello function to the project."}],
            model="claude-haiku-4-5-20251001",
            input_tokens=50,
            output_tokens=20,
            stop_reason="end_turn",
        )

        client = MagicMock()
        client.send_message = AsyncMock(side_effect=[planner_response, text_response, mem_resp])
        client.close = AsyncMock()
        client.usage = MagicMock()
        client.usage.estimated_cost = 0.001

        safety = SafetyLayer(dry_run=False)
        agent = Agent(config=agent_config, api_client=client, safety=safety)
        await agent.run()

        # Check memory.md was updated with Session Log
        memory_path = agent_config.project_dir / ".nex" / "memory.md"
        content = memory_path.read_text(encoding="utf-8")
        assert "Session Log" in content
        assert "Added hello function" in content
