"""Tests for the ChatSession class."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nex.agent import ChatSession
from nex.api_client import AnthropicClient, APIResponse
from nex.safety import SafetyLayer


@pytest.fixture
def text_response() -> APIResponse:
    return APIResponse(
        content=[{"type": "text", "text": "Hello! How can I help?"}],
        model="claude-sonnet-4-20250514",
        input_tokens=100,
        output_tokens=30,
        stop_reason="end_turn",
    )


@pytest.fixture
def tool_then_text_responses(tmp_path: Path) -> list[APIResponse]:
    return [
        APIResponse(
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_chat_001",
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
            content=[{"type": "text", "text": "I read the file. It contains a main function."}],
            model="claude-sonnet-4-20250514",
            input_tokens=200,
            output_tokens=40,
            stop_reason="end_turn",
        ),
    ]


def _make_mock_client(responses: list[APIResponse] | APIResponse) -> MagicMock:
    """Create a mock AnthropicClient."""
    client = MagicMock(spec=AnthropicClient)
    if isinstance(responses, list):
        client.send_message = AsyncMock(side_effect=responses)
    else:
        client.send_message = AsyncMock(return_value=responses)
    client.close = AsyncMock()
    client.usage = MagicMock()
    client.usage.estimated_cost = 0.001
    client.usage.total_input = 100
    client.usage.total_output = 50
    return client


class TestChatSession:
    @pytest.mark.asyncio
    async def test_single_turn_text_response(
        self, tmp_path: Path, text_response: APIResponse
    ) -> None:
        (tmp_path / ".nex").mkdir()
        client = _make_mock_client(text_response)
        safety = SafetyLayer(dry_run=False)

        session = ChatSession(
            api_client=client,
            system_prompt="You are Nex.",
            project_dir=tmp_path,
            safety=safety,
        )

        result = await session.send("Hello!")
        assert "Hello" in result or "help" in result.lower()
        assert session.turn_count == 1
        assert len(session.messages) == 2  # user + assistant

    @pytest.mark.asyncio
    async def test_multi_turn_preserves_history(
        self, tmp_path: Path, text_response: APIResponse
    ) -> None:
        (tmp_path / ".nex").mkdir()
        client = _make_mock_client(text_response)
        safety = SafetyLayer(dry_run=False)

        session = ChatSession(
            api_client=client,
            system_prompt="You are Nex.",
            project_dir=tmp_path,
            safety=safety,
        )

        await session.send("First message")
        await session.send("Second message")

        assert session.turn_count == 2
        # 2 user messages + 2 assistant messages = 4
        assert len(session.messages) == 4

    @pytest.mark.asyncio
    async def test_tool_use_in_chat(
        self, tmp_path: Path, tool_then_text_responses: list[APIResponse]
    ) -> None:
        (tmp_path / ".nex").mkdir()
        (tmp_path / "main.py").write_text("def main(): pass\n", encoding="utf-8")

        client = _make_mock_client(tool_then_text_responses)
        safety = SafetyLayer(dry_run=False)

        session = ChatSession(
            api_client=client,
            system_prompt="You are Nex.",
            project_dir=tmp_path,
            safety=safety,
        )

        result = await session.send("Read main.py")
        assert client.send_message.call_count == 2
        assert "main function" in result.lower() or "read" in result.lower()
