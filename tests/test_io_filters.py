"""Tests for logging filters."""

import logging

from mahtab.io.filters import TagFilter


def test_tag_filter_allows_matching_tag():
    f = TagFilter({"user-chat", "assistant-chat"})
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0, msg="hello", args=(), exc_info=None
    )
    record.tag = "user-chat"
    assert f.filter(record) is True


def test_tag_filter_blocks_non_matching_tag():
    f = TagFilter({"user-chat"})
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0, msg="hello", args=(), exc_info=None
    )
    record.tag = "assistant-chat"
    assert f.filter(record) is False
