"""Tag constants for message routing."""

from typing import Literal

Tag = Literal[
    "user-repl-in",
    "user-repl-out",
    "assistant-repl-in",
    "assistant-repl-out",
    "user-chat",
    "assistant-chat",
]

TAGS: set[str] = {
    "user-repl-in",
    "user-repl-out",
    "assistant-repl-in",
    "assistant-repl-out",
    "user-chat",
    "assistant-chat",
}

STREAM_TAGS: set[str] = {"assistant-chat-stream"}

COMPLETE_TAGS: set[str] = TAGS  # Excludes stream tags
