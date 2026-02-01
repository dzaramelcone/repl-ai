"""Tests for StreamingHandler as LangChain callback."""

from unittest.mock import patch

import pytest
from langchain_core.callbacks import BaseCallbackHandler

from mahtab.ui.console import console as default_console
from mahtab.ui.streaming import StreamingHandler


def _make_handler() -> StreamingHandler:
    """Create a StreamingHandler with explicit args for tests."""
    return StreamingHandler(console=default_console, chars_per_second=200.0)


def test_streaming_handler_is_callback():
    """StreamingHandler should be a LangChain BaseCallbackHandler."""
    handler = _make_handler()
    assert isinstance(handler, BaseCallbackHandler)


def test_on_llm_new_token_calls_process_token():
    """on_llm_new_token should delegate to process_token."""
    from unittest.mock import MagicMock, patch

    from mahtab.ui.streaming import StreamState

    handler = _make_handler()
    handler.reset()
    handler._write = lambda _: None

    with patch("mahtab.ui.markdown_panel.Live") as mock_live:
        mock_live.return_value = MagicMock()
        # Call the callback method with XML tag
        handler.on_llm_new_token("<assistant-chat>hello")

    # Should be in chat state (tag consumed, content in panel)
    assert handler._state == StreamState.IN_CHAT
    # Content should be in the panel buffer
    assert "hello" in handler._chat_panel.buffer


def test_on_llm_start_calls_start_spinner():
    """on_llm_start should call start_spinner."""
    handler = _make_handler()

    with patch.object(handler, "start_spinner") as mock_start_spinner:
        handler.on_llm_start(None, None)
        mock_start_spinner.assert_called_once()


def test_on_llm_end_calls_flush_and_stop_spinner():
    """on_llm_end should call flush and stop_spinner."""
    from unittest.mock import MagicMock

    handler = _make_handler()

    # Provide a proper mock response (nested list structure)
    mock_generation = MagicMock()
    mock_generation.generation_info = {"usage": {"input_tokens": 10, "output_tokens": 5, "total_cost_usd": 0.01}}
    mock_response = MagicMock()
    mock_response.generations = [[mock_generation]]

    with patch.object(handler, "flush") as mock_flush, patch.object(handler, "stop_spinner") as mock_stop_spinner:
        handler.on_llm_end(mock_response)
        mock_flush.assert_called_once()
        mock_stop_spinner.assert_called_once()


def test_on_llm_end_captures_usage_from_response():
    """on_llm_end should capture usage from response's generation_info."""
    from unittest.mock import MagicMock

    handler = _make_handler()
    handler.reset()

    # Mock response with usage in generation_info (matching ChatClaudeCLI structure)
    mock_generation = MagicMock()
    mock_generation.generation_info = {
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_cost_usd": 0.05,
        }
    }

    mock_response = MagicMock()
    mock_response.generations = [[mock_generation]]  # Nested list structure

    with patch.object(handler, "flush"), patch.object(handler, "stop_spinner"):
        handler.on_llm_end(mock_response)

    assert handler.last_usage is not None
    assert handler.last_usage["input_tokens"] == 100
    assert handler.last_usage["output_tokens"] == 50
    assert handler.last_usage["total_cost_usd"] == 0.05


def test_on_llm_end_crashes_on_response_without_usage():
    """on_llm_end should crash if response lacks usage - fail fast."""
    from unittest.mock import MagicMock

    handler = _make_handler()
    handler.reset()

    # Response with no generation_info
    mock_generation = MagicMock()
    mock_generation.generation_info = None

    mock_response = MagicMock()
    mock_response.generations = [[mock_generation]]  # Nested list structure

    with patch.object(handler, "flush"), patch.object(handler, "stop_spinner"):
        with pytest.raises(TypeError):
            handler.on_llm_end(mock_response)


def test_reset_clears_last_usage():
    """reset() should clear the last_usage field."""
    handler = _make_handler()
    handler.last_usage = {"input_tokens": 100}

    handler.reset()

    assert handler.last_usage == {}
