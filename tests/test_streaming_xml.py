"""Tests for StreamingHandler XML tag state machine."""

from mahtab.ui.console import console as default_console
from mahtab.ui.streaming import StreamingHandler, StreamState


def _make_handler() -> StreamingHandler:
    """Create a StreamingHandler with explicit args for tests."""
    return StreamingHandler(console=default_console, chars_per_second=200.0)


def test_streaming_transitions_to_chat_state():
    """Opening chat tag transitions state to IN_CHAT."""
    handler = _make_handler()
    handler.reset()
    handler._write_smooth = lambda _: None  # Suppress output

    handler.process_token("<assistant-chat>")

    assert handler._state == StreamState.IN_CHAT


def test_streaming_transitions_to_repl_state():
    """Opening repl tag transitions state to IN_REPL."""
    handler = _make_handler()
    handler.reset()
    handler._write = lambda _: None  # Suppress output
    handler._code_live = None  # Skip Live panel

    # Mock Live to avoid actual UI
    from unittest.mock import patch

    with patch("mahtab.ui.streaming.Live"):
        handler.process_token("<assistant-repl-in>")

    assert handler._state == StreamState.IN_REPL


def test_streaming_chat_back_to_outside():
    """Closing chat tag transitions back to OUTSIDE."""
    handler = _make_handler()
    handler.reset()
    handler._write_smooth = lambda _: None

    handler.process_token("<assistant-chat>hello</assistant-chat>")

    assert handler._state == StreamState.OUTSIDE


def test_streaming_repl_back_to_outside():
    """Closing repl tag transitions back to OUTSIDE."""
    from unittest.mock import MagicMock, patch

    handler = _make_handler()
    handler.reset()
    handler._write = lambda _: None

    with patch("mahtab.ui.streaming.Live") as mock_live:
        mock_instance = MagicMock()
        mock_live.return_value = mock_instance
        handler.process_token("<assistant-repl-in>x = 1</assistant-repl-in>")

    assert handler._state == StreamState.OUTSIDE


def test_streaming_chat_content_written():
    """Chat content is written via _write_smooth."""
    handler = _make_handler()
    handler.reset()
    written = []
    handler._write_smooth = lambda text: written.append(text)

    handler.process_token("<assistant-chat>hello world</assistant-chat>")

    assert "hello world" in "".join(written)


def test_streaming_repl_content_accumulated():
    """Repl content is accumulated in code buffer."""
    from unittest.mock import MagicMock, patch

    handler = _make_handler()
    handler.reset()
    handler._write = lambda _: None

    with patch("mahtab.ui.streaming.Live") as mock_live:
        mock_instance = MagicMock()
        mock_live.return_value = mock_instance
        handler.process_token("<assistant-repl-in>")
        handler.process_token("x = 42")

    assert "x = 42" in handler._code_panel.buffer


def test_streaming_partial_tag_waits():
    """Partial opening tag in OUTSIDE state waits for more tokens."""
    handler = _make_handler()
    handler.reset()
    handler._write_smooth = lambda _: None

    # Send partial tag
    handler.process_token("<assistant-ch")

    assert handler._state == StreamState.OUTSIDE
    assert handler._buffer == "<assistant-ch"


def test_streaming_partial_tag_completes():
    """Partial tag completes when rest arrives."""
    handler = _make_handler()
    handler.reset()
    written = []
    handler._write_smooth = lambda text: written.append(text)

    handler.process_token("<assistant-ch")
    handler.process_token("at>hello")

    assert handler._state == StreamState.IN_CHAT
    assert "hello" in "".join(written)


def test_streaming_partial_close_tag_waits():
    """Partial closing tag waits for more tokens."""
    handler = _make_handler()
    handler.reset()
    handler._write_smooth = lambda _: None

    handler.process_token("<assistant-chat>hello</")

    assert handler._state == StreamState.IN_CHAT
    assert handler._buffer == "</"


def test_streaming_multiple_blocks():
    """Multiple chat and repl blocks are handled correctly."""
    from unittest.mock import MagicMock, patch

    handler = _make_handler()
    handler.reset()
    written = []
    handler._write_smooth = lambda text: written.append(text)
    handler._write = lambda _: None

    with patch("mahtab.ui.streaming.Live") as mock_live:
        mock_instance = MagicMock()
        mock_live.return_value = mock_instance
        handler.process_token("<assistant-chat>First</assistant-chat>")
        handler.process_token("<assistant-repl-in>x = 1</assistant-repl-in>")
        handler.process_token("<assistant-chat>Second</assistant-chat>")

    assert handler._state == StreamState.OUTSIDE
    assert "First" in "".join(written)
    assert "Second" in "".join(written)


def test_streaming_outputs_outside_content():
    """Content outside known tags is output directly."""
    handler = _make_handler()
    handler.reset()
    written = []
    handler._write_smooth = lambda text: written.append(text)

    handler.process_token("garbage before")

    assert handler._state == StreamState.OUTSIDE
    assert handler._buffer == ""
    assert "garbage before" in "".join(written)


def test_streaming_outputs_unknown_tags_and_finds_known():
    """Unknown tags like <thinking> are output, then known tags are handled."""
    handler = _make_handler()
    handler.reset()
    written = []
    handler._write_smooth = lambda text: written.append(text)

    # Simulate <thinking>...</thinking><assistant-chat>Hello</assistant-chat>
    handler.process_token("<thinking>some thoughts</thinking><assistant-chat>Hello</assistant-chat>")

    assert handler._state == StreamState.OUTSIDE
    all_written = "".join(written)
    assert "<thinking>some thoughts</thinking>" in all_written
    assert "Hello" in all_written


def test_flush_in_chat_state():
    """Flush writes remaining buffer when in chat state."""
    handler = _make_handler()
    handler.reset()
    written = []
    handler._write_smooth = lambda text: written.append(text)
    handler._write = lambda _: None

    handler.process_token("<assistant-chat>partial content")
    handler.flush()

    assert "partial content" in "".join(written)
