"""Tests for logging formatters."""

import logging

from mahtab.io.formatters import RichFormatter, XMLFormatter


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


def test_rich_formatter_user_repl_in():
    f = RichFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="x = 5",
        args=(),
        exc_info=None,
    )
    record.tag = "user-repl-in"
    result = f.format(record)
    assert result == "[bold cyan]>>> [/]x = 5"


def test_rich_formatter_user_chat():
    f = RichFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.tag = "user-chat"
    result = f.format(record)
    assert result == "[bold green]You:[/] hello"


def test_rich_formatter_assistant_chat():
    f = RichFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="hi there",
        args=(),
        exc_info=None,
    )
    record.tag = "assistant-chat"
    result = f.format(record)
    assert result == "[bold blue]Claude:[/] hi there"
