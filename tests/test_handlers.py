"""Tests for logging handlers."""

import logging

from mahtab.store import Store
from mahtab.ui.handlers import StoreHandler


def test_store_handler_appends_to_store():
    store = Store()
    handler = StoreHandler(store)
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger = logging.getLogger("test.store_handler")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info("hello world")

    assert b"hello world" in store.data


def test_store_handler_includes_newline():
    store = Store()
    handler = StoreHandler(store)
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger = logging.getLogger("test.store_handler_newline")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info("line1")
    logger.info("line2")

    assert store.data == b"line1\nline2\n"
