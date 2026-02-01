"""Session: Async REPL with its own namespace and conversation history."""

from __future__ import annotations

import io
import logging
from code import InteractiveInterpreter
from contextlib import redirect_stderr, redirect_stdout
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from mahtab.store import Store


class SessionInterpreter(InteractiveInterpreter):
    """Python interpreter that captures output."""

    def __init__(self, session: Session):
        super().__init__(locals=session.namespace)
        self.session = session
        self._stdout_buffer = io.StringIO()
        self._stderr_buffer = io.StringIO()
        self._error_buffer = io.StringIO()

    def write(self, data: str) -> None:
        """Called by InteractiveInterpreter for error output (tracebacks)."""
        self._error_buffer.write(data)

    def run(self, source: str) -> tuple[str, str]:
        """Execute source code, returning (output, error) strings."""
        self._stdout_buffer = io.StringIO()
        self._stderr_buffer = io.StringIO()
        self._error_buffer = io.StringIO()

        with redirect_stdout(self._stdout_buffer), redirect_stderr(self._stderr_buffer):
            self.runsource(source, "<input>", "single")

        stdout = self._stdout_buffer.getvalue().rstrip()
        stderr = self._stderr_buffer.getvalue().rstrip()
        errors = self._error_buffer.getvalue().rstrip()

        # Combine stderr and interpreter errors
        all_errors = "\n".join(filter(None, [stderr, errors]))

        return stdout, all_errors


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

        # Python interpreter for this session
        self.interpreter = SessionInterpreter(self)

    def spawn(self, context: dict | None = None) -> Session:
        """Create a child session with shared store."""
        return Session(store=self.store, parent=self, context=context)

    def summarize_namespace(self, max_vars: int = 30) -> str:
        """Summarize variables in the namespace for the system prompt."""
        if not self.namespace:
            return "(empty)"

        lines = []
        for name, val in list(self.namespace.items())[:max_vars]:
            if name.startswith("_"):
                continue
            try:
                typ = type(val).__name__
                if isinstance(val, int | float | str | bool | type(None)):
                    rep = repr(val)[:50]
                elif isinstance(val, list | dict | set | tuple):
                    rep = f"{typ} with {len(val)} items"
                else:
                    rep = typ
                lines.append(f"  {name}: {rep}")
            except Exception:
                lines.append(f"  {name}: <unknown>")

        return "\n".join(lines) or "(no user variables)"
