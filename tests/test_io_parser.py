"""Tests for response parsing."""

import logging

from mahtab.io.handlers import PromptHandler
from mahtab.io.parser import parse_response, route_response


def test_parse_response_extracts_chat():
    response = "<assistant-chat>Hello there!</assistant-chat>"
    result = parse_response(response)
    assert result == [("assistant-chat", "Hello there!")]


def test_parse_response_extracts_repl():
    response = "<assistant-repl-in>x = 5</assistant-repl-in>"
    result = parse_response(response)
    assert result == [("assistant-repl-in", "x = 5")]


def test_parse_response_extracts_multiple():
    response = """<assistant-chat>Let me calculate that.</assistant-chat>
<assistant-repl-in>result = 2 + 2</assistant-repl-in>"""
    result = parse_response(response)
    assert len(result) == 2
    assert result[0] == ("assistant-chat", "Let me calculate that.")
    assert result[1] == ("assistant-repl-in", "result = 2 + 2")


def test_parse_response_multiline_content():
    response = """<assistant-repl-in>def foo():
    return 42</assistant-repl-in>"""
    result = parse_response(response)
    assert result == [("assistant-repl-in", "def foo():\n    return 42")]


def test_route_response_sends_to_logger():
    # Set up logger with prompt handler
    log = logging.getLogger("mahtab")
    log.setLevel(logging.INFO)
    handler = PromptHandler()
    log.addHandler(handler)

    try:
        response = "<assistant-chat>Hello!</assistant-chat>"
        route_response(response)

        context = handler.get_context()
        assert "<assistant-chat>Hello!</assistant-chat>" in context
    finally:
        log.removeHandler(handler)
