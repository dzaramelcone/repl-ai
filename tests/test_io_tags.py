"""Tests for IO tag constants."""

from mahtab.io.tags import COMPLETE_TAGS, STREAM_TAGS, TAGS


def test_tags_contains_all_six():
    expected = {
        "user-repl-in",
        "user-repl-out",
        "assistant-repl-in",
        "assistant-repl-out",
        "user-chat",
        "assistant-chat",
    }
    assert TAGS == expected


def test_stream_tags():
    assert STREAM_TAGS == {"assistant-chat-stream"}


def test_complete_tags_excludes_stream():
    assert "assistant-chat-stream" not in COMPLETE_TAGS
    assert "assistant-chat" in COMPLETE_TAGS
