"""Session: Async REPL with its own namespace and conversation history."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from mahtab.store import Store


class Session:
    """A REPL session. Async, can spawn children, shares the Store."""

    def __init__(
        self,
        store: Store,
        parent: Session | None = None,
        context: dict | None = None,
    ):
        self.id = uuid4().hex[:8]
        self.store = store
        self.parent = parent
        self.children: list[Session] = []

        # REPL state
        self.namespace: dict[str, Any] = {}
        self.messages: list = []

        # Inherit context from parent
        if context:
            self.namespace.update(context)

        # Track parent relationship
        if parent:
            parent.children.append(self)

        # Set up loggers
        self.log_user_repl = logging.getLogger(f"session.{self.id}.user.repl")
        self.log_user_chat = logging.getLogger(f"session.{self.id}.user.chat")
        self.log_llm_repl = logging.getLogger(f"session.{self.id}.llm.repl")
        self.log_llm_chat = logging.getLogger(f"session.{self.id}.llm.chat")

    def spawn(self, context: dict | None = None) -> Session:
        """Create a child session with shared store."""
        return Session(store=self.store, parent=self, context=context)

    def exec(self, code: str) -> Any:
        """Execute Python code in this session's namespace."""
        exec(code, self.namespace)
