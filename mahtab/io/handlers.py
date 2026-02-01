"""Logging handlers for message routing."""

from __future__ import annotations

import logging

from mahtab.io.formatters import XMLFormatter


class PromptHandler(logging.Handler):
    """Accumulates XML-formatted messages for Claude's context."""

    def __init__(self) -> None:
        super().__init__()
        self.buffer: list[str] = []
        self.setFormatter(XMLFormatter())

    def emit(self, record: logging.LogRecord) -> None:
        self.buffer.append(self.format(record))

    def get_context(self) -> str:
        return "\n".join(self.buffer)

    def clear(self) -> None:
        self.buffer.clear()
