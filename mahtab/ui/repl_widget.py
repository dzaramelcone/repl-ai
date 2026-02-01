"""REPLWidget: Textual widget for a single Session."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import RichLog, TextArea

from mahtab.ui.handlers import StoreHandler

if TYPE_CHECKING:
    from mahtab.session import Session


class RichLogHandler(logging.Handler):
    """Sends log records to a RichLog widget."""

    def __init__(self, widget: RichLog):
        super().__init__()
        self.widget = widget

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.widget.write(msg)


class REPLWidget(Widget):
    """UI for a single Session. Chat pane + REPL pane + input."""

    DEFAULT_CSS = """
    REPLWidget {
        layout: grid;
        grid-size: 1 2;
        grid-rows: 1fr auto;
    }

    REPLWidget Horizontal {
        height: 1fr;
    }

    REPLWidget #chat {
        width: 1fr;
        border: solid $primary;
    }

    REPLWidget #repl {
        width: 1fr;
        border: solid $secondary;
    }

    REPLWidget #input {
        height: 5;
        border: solid $accent;
    }
    """

    def __init__(self, session: Session):
        super().__init__()
        self.session = session

    def compose(self):
        with Horizontal():
            yield RichLog(id="chat", wrap=True, markup=True)
            yield RichLog(id="repl", wrap=True, markup=True)
        yield TextArea(id="input", language="python")

    def on_mount(self):
        self.call_after_refresh(self._setup_handlers)

    def _setup_handlers(self):
        chat = self.query_one("#chat", RichLog)
        repl = self.query_one("#repl", RichLog)

        # Chat pane gets user and LLM chat
        chat_handler = RichLogHandler(chat)
        chat_handler.setFormatter(logging.Formatter("%(message)s"))
        self.session.log_user_chat.addHandler(chat_handler)
        self.session.log_llm_chat.addHandler(chat_handler)
        self.session.log_user_chat.setLevel(logging.INFO)
        self.session.log_llm_chat.setLevel(logging.INFO)

        # REPL pane gets user and LLM repl
        repl_handler = RichLogHandler(repl)
        repl_handler.setFormatter(logging.Formatter("%(message)s"))
        self.session.log_user_repl.addHandler(repl_handler)
        self.session.log_llm_repl.addHandler(repl_handler)
        self.session.log_user_repl.setLevel(logging.INFO)
        self.session.log_llm_repl.setLevel(logging.INFO)

        # Store gets everything
        store_handler = StoreHandler(self.session.store)
        store_handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        for logger in [
            self.session.log_user_chat,
            self.session.log_llm_chat,
            self.session.log_user_repl,
            self.session.log_llm_repl,
        ]:
            logger.addHandler(store_handler)

    async def on_key(self, event):
        if event.key == "ctrl+enter":
            await self.submit()
            event.prevent_default()

    async def submit(self):
        input_widget = self.query_one("#input", TextArea)
        code = input_widget.text.strip()
        if not code:
            return
        input_widget.clear()

        # Log the input
        self.session.log_user_repl.info(f">>> {code}")

        # Execute
        try:
            # Try eval first for expressions
            try:
                result = eval(code, self.session.namespace)
                if result is not None:
                    self.session.log_user_repl.info(repr(result))
            except SyntaxError:
                # Fall back to exec for statements
                exec(code, self.session.namespace)
        except Exception as e:
            self.session.log_user_repl.error(f"[red]{type(e).__name__}: {e}[/red]")
