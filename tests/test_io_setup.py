"""Tests for logger setup."""

import logging

from mahtab.io.handlers import PromptHandler
from mahtab.io.setup import setup_logging


class MockStore:
    def __init__(self) -> None:
        self.data = bytearray()

    def append(self, data: bytes) -> None:
        self.data.extend(data)


def test_setup_logging_returns_logger_and_prompt():
    store = MockStore()
    log, prompt = setup_logging(store)

    assert isinstance(log, logging.Logger)
    assert log.name == "mahtab"
    assert isinstance(prompt, PromptHandler)


def test_setup_logging_routes_to_all_handlers():
    store = MockStore()
    log, prompt = setup_logging(store)

    # Clear any prior state
    prompt.clear()
    store.data.clear()

    log.info("hello", extra={"tag": "user-chat"})

    # Check prompt handler received it
    assert "<user-chat>hello</user-chat>" in prompt.get_context()

    # Check store received it
    assert b"<user-chat>hello</user-chat>" in store.data
