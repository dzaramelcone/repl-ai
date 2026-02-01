"""Tests for logging formatters."""

import logging

from mahtab.io.formatters import XMLFormatter


def test_xml_formatter_wraps_in_tag():
    f = XMLFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0, msg="x = 5", args=(), exc_info=None
    )
    record.tag = "user-repl-in"
    result = f.format(record)
    assert result == "<user-repl-in>x = 5</user-repl-in>"


def test_xml_formatter_handles_multiline():
    f = XMLFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0, msg="line1\nline2", args=(), exc_info=None
    )
    record.tag = "assistant-chat"
    result = f.format(record)
    assert result == "<assistant-chat>line1\nline2</assistant-chat>"
