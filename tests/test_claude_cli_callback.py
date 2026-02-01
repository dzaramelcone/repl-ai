"""Tests for ChatClaudeCLI callback integration."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from mahtab.llm.claude_cli import ChatClaudeCLI


@pytest.mark.asyncio
async def test_agenerate_emits_tokens_to_run_manager():
    """_agenerate should call run_manager.on_llm_new_token for each token."""
    llm = ChatClaudeCLI()

    run_manager = AsyncMock()
    tokens_received = []

    async def capture_token(token, **kwargs):
        tokens_received.append(token)

    run_manager.on_llm_new_token = capture_token

    # Simulate streaming JSON output from CLI
    stream_lines = [
        json.dumps(
            {
                "type": "stream_event",
                "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}},
            }
        ),
        json.dumps(
            {
                "type": "stream_event",
                "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": " world"}},
            }
        ),
        json.dumps({"type": "result", "result": "Hello world", "usage": {"input_tokens": 10, "output_tokens": 2}}),
    ]

    async def mock_readline():
        for line in stream_lines:
            yield (line + "\n").encode()

    mock_proc = AsyncMock()
    mock_proc.stdout = mock_readline()
    mock_proc.stderr = AsyncMock()
    mock_proc.stderr.read = AsyncMock(return_value=b"")
    mock_proc.wait = AsyncMock()
    mock_proc.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        from langchain_core.messages import HumanMessage

        messages = [HumanMessage(content="test")]

        await llm._agenerate(messages, run_manager=run_manager)

    assert tokens_received == ["Hello", " world"]
