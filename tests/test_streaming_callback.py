"""Tests for StreamingHandler as LangChain callback."""

from unittest.mock import patch

from langchain_core.callbacks import BaseCallbackHandler

from mahtab.ui.streaming import StreamingHandler


def test_streaming_handler_is_callback():
    """StreamingHandler should be a LangChain BaseCallbackHandler."""
    handler = StreamingHandler()
    assert isinstance(handler, BaseCallbackHandler)


def test_on_llm_new_token_calls_process_token():
    """on_llm_new_token should delegate to process_token."""
    handler = StreamingHandler()
    handler.reset()

    # Call the callback method
    handler.on_llm_new_token("hello")

    # Text should be in the buffer (process_token buffers short text)
    assert "hello" in handler._text_buffer


def test_on_llm_start_calls_start_spinner():
    """on_llm_start should call start_spinner."""
    handler = StreamingHandler()

    with patch.object(handler, "start_spinner") as mock_start_spinner:
        handler.on_llm_start(None, None)
        mock_start_spinner.assert_called_once()


def test_on_llm_end_calls_flush_and_stop_spinner():
    """on_llm_end should call flush and stop_spinner."""
    handler = StreamingHandler()

    with patch.object(handler, "flush") as mock_flush, patch.object(handler, "stop_spinner") as mock_stop_spinner:
        handler.on_llm_end(None)
        mock_flush.assert_called_once()
        mock_stop_spinner.assert_called_once()


def test_on_llm_end_captures_usage_from_response():
    """on_llm_end should capture usage from response's generation_info."""
    from unittest.mock import MagicMock

    handler = StreamingHandler()
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
    mock_response.generations = [mock_generation]

    with patch.object(handler, "flush"), patch.object(handler, "stop_spinner"):
        handler.on_llm_end(mock_response)

    assert handler.last_usage is not None
    assert handler.last_usage["input_tokens"] == 100
    assert handler.last_usage["output_tokens"] == 50
    assert handler.last_usage["total_cost_usd"] == 0.05


def test_on_llm_end_handles_response_without_usage():
    """on_llm_end should handle response without usage gracefully."""
    from unittest.mock import MagicMock

    handler = StreamingHandler()
    handler.reset()

    # Response with no generation_info
    mock_generation = MagicMock()
    mock_generation.generation_info = None

    mock_response = MagicMock()
    mock_response.generations = [mock_generation]

    with patch.object(handler, "flush"), patch.object(handler, "stop_spinner"):
        handler.on_llm_end(mock_response)

    assert handler.last_usage is None


def test_reset_clears_last_usage():
    """reset() should clear the last_usage field."""
    handler = StreamingHandler()
    handler.last_usage = {"input_tokens": 100}

    handler.reset()

    assert handler.last_usage is None
