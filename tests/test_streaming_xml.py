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
    """Chat content is accumulated in chat panel buffer."""
    from unittest.mock import MagicMock, patch

    handler = _make_handler()
    handler.reset()
    handler._write = lambda _: None

    with patch("mahtab.ui.markdown_panel.Live") as mock_live:
        mock_instance = MagicMock()
        mock_live.return_value = mock_instance
        handler.process_token("<assistant-chat>hello world</assistant-chat>")

    # Content was in the panel (buffer is cleared after finish)
    mock_live.assert_called()


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
    from unittest.mock import MagicMock, patch

    handler = _make_handler()
    handler.reset()
    handler._write = lambda _: None

    with patch("mahtab.ui.markdown_panel.Live") as mock_live:
        mock_instance = MagicMock()
        mock_live.return_value = mock_instance
        handler.process_token("<assistant-ch")
        handler.process_token("at>hello")

    assert handler._state == StreamState.IN_CHAT
    assert "hello" in handler._chat_panel.buffer


def test_streaming_partial_close_tag_waits():
    """Partial closing tag waits for more tokens."""
    from unittest.mock import MagicMock, patch

    handler = _make_handler()
    handler.reset()
    handler._write = lambda _: None

    with patch("mahtab.ui.markdown_panel.Live") as mock_live:
        mock_instance = MagicMock()
        mock_live.return_value = mock_instance
        handler.process_token("<assistant-chat>hello</")

    assert handler._state == StreamState.IN_CHAT
    assert handler._buffer == "</"


def test_streaming_multiple_blocks():
    """Multiple chat and repl blocks are handled correctly."""
    from unittest.mock import MagicMock, patch

    handler = _make_handler()
    handler.reset()
    handler._write = lambda _: None

    with (
        patch("mahtab.ui.streaming.Live") as mock_code_live,
        patch("mahtab.ui.markdown_panel.Live") as mock_chat_live,
    ):
        mock_code_live.return_value = MagicMock()
        mock_chat_live.return_value = MagicMock()
        handler.process_token("<assistant-chat>First</assistant-chat>")
        handler.process_token("<assistant-repl-in>x = 1</assistant-repl-in>")
        handler.process_token("<assistant-chat>Second</assistant-chat>")

    assert handler._state == StreamState.OUTSIDE
    # Chat panels were created
    assert mock_chat_live.call_count >= 2


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


def test_streaming_generic_xml_in_panel():
    """Unknown tags like <thinking> are displayed in panels."""
    from unittest.mock import MagicMock, patch

    handler = _make_handler()
    handler.reset()
    handler._write = lambda _: None

    with (
        patch("mahtab.ui.xml_panel.Live") as mock_xml_live,
        patch("mahtab.ui.markdown_panel.Live") as mock_chat_live,
    ):
        mock_xml_live.return_value = MagicMock()
        mock_chat_live.return_value = MagicMock()
        handler.process_token("<thinking>some thoughts</thinking><assistant-chat>Hello</assistant-chat>")

    assert handler._state == StreamState.OUTSIDE
    # The <thinking> content was shown in a panel
    mock_xml_live.assert_called()
    # The chat content was shown in a panel
    mock_chat_live.assert_called()


def test_flush_in_chat_state():
    """Flush finishes chat panel when in chat state."""
    from unittest.mock import MagicMock, patch

    handler = _make_handler()
    handler.reset()
    handler._write = lambda _: None

    with patch("mahtab.ui.markdown_panel.Live") as mock_live:
        mock_instance = MagicMock()
        mock_live.return_value = mock_instance
        handler.process_token("<assistant-chat>partial content")
        handler.flush()

    # Panel was created and stopped
    mock_live.assert_called()
    mock_instance.stop.assert_called()
