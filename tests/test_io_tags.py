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


def test_public_api_exports():
    from mahtab.io import (
        COMPLETE_TAGS,
        STREAM_TAGS,
        TAGS,
        BytesFormatter,
        DisplayHandler,
        PromptHandler,
        RichFormatter,
        Store,
        StoreHandler,
        TagFilter,
        XMLFormatter,
        parse_response,
        route_response,
        setup_logging,
    )

    # Verify all imports are accessible
    exports = [
        TAGS,
        STREAM_TAGS,
        COMPLETE_TAGS,
        TagFilter,
        XMLFormatter,
        RichFormatter,
        BytesFormatter,
        PromptHandler,
        DisplayHandler,
        StoreHandler,
        Store,
        parse_response,
        route_response,
        setup_logging,
    ]
    assert all(e is not None for e in exports)
