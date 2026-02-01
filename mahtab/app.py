"""MahtabApp: Main Textual application."""

from __future__ import annotations

import logging

from textual.app import App
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, RichLog, TabbedContent, TabPane, TextArea

from mahtab.session import Session
from mahtab.store import Store
from mahtab.ui.handlers import StoreHandler


class RichLogHandler(logging.Handler):
    """Sends log records to a RichLog widget."""

    def __init__(self, widget: RichLog):
        super().__init__()
        self.widget = widget

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.widget.write(msg)


class MahtabApp(App):
    """Main app. Manages sessions as tabs, owns the shared Store."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #sessions {
        height: 1fr;
    }

    .session-content {
        height: 1fr;
        width: 100%;
    }

    .session-content Horizontal {
        height: 1fr;
        width: 100%;
    }

    .chat-pane, .repl-pane {
        width: 1fr;
        height: 100%;
        border: solid $primary;
    }

    .repl-pane {
        border: solid $secondary;
    }

    .input-area {
        height: 5;
        border: solid $accent;
    }
    """

    BINDINGS = [
        # Tab management bindings TBD - avoiding terminal conflicts for now
    ]

    def __init__(self):
        super().__init__()
        self.store = Store()
        self.sessions: dict[str, Session] = {}

    def compose(self):
        yield Header()
        # Create initial session during compose
        session = Session(store=self.store)
        self.sessions[session.id] = session
        with TabbedContent(id="sessions"):
            with TabPane(f"Session {session.id}", id=f"tab-{session.id}"):
                yield Vertical(
                    Horizontal(
                        RichLog(id=f"chat-{session.id}", classes="chat-pane", wrap=True, markup=True),
                        RichLog(id=f"repl-{session.id}", classes="repl-pane", wrap=True, markup=True),
                    ),
                    TextArea(id=f"input-{session.id}", classes="input-area", language="python"),
                    classes="session-content",
                )
        yield Footer()

    def on_mount(self):
        # Wire up logging handlers for the initial session
        for session in self.sessions.values():
            self._wire_session_handlers(session)
            # Focus the input
            self.query_one(f"#input-{session.id}", TextArea).focus()

    def _wire_session_handlers(self, session: Session):
        """Wire logging handlers for a session's widgets."""
        chat = self.query_one(f"#chat-{session.id}", RichLog)
        repl = self.query_one(f"#repl-{session.id}", RichLog)

        # Chat pane gets user and LLM chat
        chat_handler = RichLogHandler(chat)
        chat_handler.setFormatter(logging.Formatter("%(message)s"))
        session.log_user_chat.addHandler(chat_handler)
        session.log_llm_chat.addHandler(chat_handler)
        session.log_user_chat.setLevel(logging.INFO)
        session.log_llm_chat.setLevel(logging.INFO)

        # REPL pane gets user and LLM repl
        repl_handler = RichLogHandler(repl)
        repl_handler.setFormatter(logging.Formatter("%(message)s"))
        session.log_user_repl.addHandler(repl_handler)
        session.log_llm_repl.addHandler(repl_handler)
        session.log_user_repl.setLevel(logging.INFO)
        session.log_llm_repl.setLevel(logging.INFO)

        # Store gets everything
        store_handler = StoreHandler(self.store)
        store_handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        for logger in [session.log_user_chat, session.log_llm_chat, session.log_user_repl, session.log_llm_repl]:
            logger.addHandler(store_handler)

    def spawn_session(
        self,
        parent: Session | None = None,
        context: dict | None = None,
    ) -> Session:
        session = Session(store=self.store, parent=parent, context=context)
        self.sessions[session.id] = session

        tabs = self.query_one("#sessions", TabbedContent)
        label = f"↳ {session.id}" if parent else f"Session {session.id}"

        # Create inline widgets (custom Widget classes don't work in TabPane)
        content = Vertical(
            Horizontal(
                RichLog(id=f"chat-{session.id}", classes="chat-pane", wrap=True, markup=True),
                RichLog(id=f"repl-{session.id}", classes="repl-pane", wrap=True, markup=True),
            ),
            TextArea(id=f"input-{session.id}", classes="input-area", language="python"),
            classes="session-content",
        )
        pane = TabPane(label, content, id=f"tab-{session.id}")
        tabs.add_pane(pane)

        # Wire handlers after mount
        self.call_after_refresh(lambda: self._wire_session_handlers(session))

        return session

    def action_new_session(self):
        self.spawn_session()

    def action_close_session(self):
        tabs = self.query_one("#sessions", TabbedContent)
        if len(tabs._tab_content) > 1:
            tabs.remove_pane(tabs.active)

    def action_next_tab(self):
        tabs = self.query_one("#sessions", TabbedContent)
        tabs.action_next_tab()

    def action_prev_tab(self):
        tabs = self.query_one("#sessions", TabbedContent)
        tabs.action_previous_tab()

    def _get_active_session(self) -> Session | None:
        """Get the currently active session."""
        tabs = self.query_one("#sessions", TabbedContent)
        active_tab_id = tabs.active
        if active_tab_id and active_tab_id.startswith("tab-"):
            session_id = active_tab_id[4:]  # Remove "tab-" prefix
            return self.sessions.get(session_id)
        return None

    async def on_key(self, event):
        """Handle input submission."""
        session = self._get_active_session()
        if not session:
            return

        if event.key == "cmd+shift+enter":
            # Cmd+Shift+Enter -> execute as Python in REPL
            await self._submit_to_repl(session)
            event.prevent_default()
        elif event.key == "cmd+enter":
            # Cmd+Enter -> send to chat (ask Claude)
            await self._submit_to_chat(session)
            event.prevent_default()

    async def _submit_to_repl(self, session: Session):
        """Execute code from the session's input area."""
        input_widget = self.query_one(f"#input-{session.id}", TextArea)
        code = input_widget.text.strip()
        if not code:
            return
        input_widget.clear()

        # Log the input
        session.log_user_repl.info(f">>> {code}")

        # Execute
        try:
            try:
                result = eval(code, session.namespace)
                if result is not None:
                    session.log_user_repl.info(repr(result))
            except SyntaxError:
                exec(code, session.namespace)
        except Exception as e:
            session.log_user_repl.error(f"[red]{type(e).__name__}: {e}[/red]")

    async def _submit_to_chat(self, session: Session):
        """Send input to chat (Claude). Not yet implemented."""
        input_widget = self.query_one(f"#input-{session.id}", TextArea)
        prompt = input_widget.text.strip()
        if not prompt:
            return
        input_widget.clear()

        # Log the prompt
        session.log_user_chat.info(f"You: {prompt}")
        # TODO: Wire up Claude agent here
        session.log_llm_chat.info("[dim]Claude integration coming soon...[/dim]")
