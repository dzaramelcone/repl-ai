"""Logging handlers for routing output."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.callbacks import BaseCallbackHandler

if TYPE_CHECKING:
    from mahtab.session import Session
    from mahtab.store import Store


class StoreHandler(logging.Handler):
    """Appends log records to the Store."""

    def __init__(self, store: Store):
        super().__init__()
        self.store = store

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.store.append(msg + "\n")


class SessionStreamingHandler(BaseCallbackHandler):
    """LangChain callback that streams tokens to session's chat logger."""

    def __init__(self, session: Session):
        super().__init__()
        self.session = session
        self._buffer = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Stream tokens to the chat log."""
        self._buffer += token
        # Flush on newlines for responsive output
        if "\n" in self._buffer:
            lines = self._buffer.split("\n")
            for line in lines[:-1]:
                if line:
                    self.session.log_llm_chat.info(line)
            self._buffer = lines[-1]

    def on_llm_end(self, response, **kwargs) -> None:
        """Flush remaining buffer."""
        if self._buffer:
            self.session.log_llm_chat.info(self._buffer)
            self._buffer = ""

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Log errors."""
        self.session.log_llm_chat.error(f"[red]Error: {error}[/red]")
