"""Logging handlers for routing output."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mahtab.store import Store


class StoreHandler(logging.Handler):
    """Appends log records to the Store."""

    def __init__(self, store: Store):
        super().__init__()
        self.store = store

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.store.append(msg + "\n")
