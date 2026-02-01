"""Tests for logging handlers."""

import logging
from io import StringIO
from unittest.mock import MagicMock

from rich.console import Console

from mahtab.io.handlers import DisplayHandler, PromptHandler, StoreHandler


def test_prompt_handler_accumulates_xml():
    handler = PromptHandler()
    log = logging.getLogger("test_prompt")
    log.addHandler(handler)
    log.setLevel(logging.INFO)

    record1 = log.makeRecord("test", logging.INFO, "", 0, "hello", (), None)
    record1.tag = "user-chat"
    handler.emit(record1)

    record2 = log.makeRecord("test", logging.INFO, "", 0, "hi there", (), None)
    record2.tag = "assistant-chat"
    handler.emit(record2)

    context = handler.get_context()
    assert "<user-chat>hello</user-chat>" in context
    assert "<assistant-chat>hi there</assistant-chat>" in context


def test_prompt_handler_clear():
    handler = PromptHandler()

    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0, msg="hello", args=(), exc_info=None
    )
    record.tag = "user-chat"
    handler.emit(record)

    assert handler.get_context() != ""
    handler.clear()
    assert handler.get_context() == ""


class MockStore:
    """Mock store for testing."""

    def __init__(self) -> None:
        self.data = bytearray()

    def append(self, data: bytes) -> None:
        self.data.extend(data)


def test_store_handler_appends_bytes():
    store = MockStore()
    handler = StoreHandler(store)

    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0, msg="hello", args=(), exc_info=None
    )
    record.tag = "user-chat"
    handler.emit(record)

    assert store.data == b"<user-chat>hello</user-chat>"


def test_display_handler_prints_formatted():
    output = StringIO()
    console = Console(file=output, force_terminal=True)
    handler = DisplayHandler(console)

    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0, msg="hello", args=(), exc_info=None
    )
    record.tag = "user-chat"
    handler.emit(record)

    result = output.getvalue()
    assert "You:" in result
    assert "hello" in result


def test_display_handler_streams_tokens():
    console = MagicMock(spec=Console)
    handler = DisplayHandler(console)
    handler.streamer = MagicMock()

    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0, msg="tok", args=(), exc_info=None
    )
    record.tag = "assistant-chat-stream"
    handler.emit(record)

    handler.streamer.process_token.assert_called_once_with("tok")


def test_display_handler_skips_repl_in():
    """assistant-repl-in is skipped since it's shown in streaming code panel."""
    output = StringIO()
    console = Console(file=output, force_terminal=True)
    handler = DisplayHandler(console)

    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0, msg="x = 42", args=(), exc_info=None
    )
    record.tag = "assistant-repl-in"
    handler.emit(record)

    # Output should be empty - skipped
    assert output.getvalue() == ""


def test_display_handler_skips_assistant_chat():
    """assistant-chat is skipped since it's shown during streaming."""
    output = StringIO()
    console = Console(file=output, force_terminal=True)
    handler = DisplayHandler(console)

    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0, msg="Hello world", args=(), exc_info=None
    )
    record.tag = "assistant-chat"
    handler.emit(record)

    # Output should be empty - skipped (already shown during streaming)
    assert output.getvalue() == ""


def test_display_handler_skips_repl_out():
    """assistant-repl-out is skipped since on_execution callback shows it."""
    output = StringIO()
    console = Console(file=output, force_terminal=True)
    handler = DisplayHandler(console)

    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0, msg="42\n", args=(), exc_info=None
    )
    record.tag = "assistant-repl-out"
    handler.emit(record)

    # Output should be empty - skipped (on_execution shows Output panel)
    assert output.getvalue() == ""
