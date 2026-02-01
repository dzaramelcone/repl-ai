"""Tests for logging handlers."""

import logging

from mahtab.io.handlers import PromptHandler, StoreHandler


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
