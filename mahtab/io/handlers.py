"""Logging handlers for message routing."""

from __future__ import annotations

import logging
from typing import Protocol

from mahtab.io.formatters import BytesFormatter, XMLFormatter


class Store(Protocol):
    """Protocol for message stores."""

    def append(self, data: bytes) -> None: ...


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


class StoreHandler(logging.Handler):
    """Appends bytes to a store."""

    def __init__(self, store: Store) -> None:
        super().__init__()
        self.store = store
        self.setFormatter(BytesFormatter())

    def emit(self, record: logging.LogRecord) -> None:
        self.store.append(self.format(record))
