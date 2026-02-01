"""MahtabApp: Main Textual application."""

from __future__ import annotations

import logging
import re

from rich.panel import Panel
from rich.syntax import Syntax
from textual.app import App
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import (
    Footer,
    Header,
    LoadingIndicator,
    Markdown,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)

from mahtab.agent.repl_agent import REPLAgent
from mahtab.session import Session
from mahtab.store import Store
from mahtab.ui.handlers import StoreHandler


class InputArea(TextArea):
    """TextArea that posts Submit message on Enter."""

    class Submit(Message):
        """Posted when user presses Enter."""

        def __init__(self, text: str, input_id: str) -> None:
            super().__init__()
            self.text = text
            self.input_id = input_id

    def _on_key(self, event) -> None:
        if event.key == "enter":
            self.post_message(self.Submit(self.text, self.id or ""))
            self.clear()
            event.prevent_default()
            event.stop()
        else:
            super()._on_key(event)


def strip_code_blocks(text: str) -> str:
    """Remove fenced code blocks from markdown text."""
    return re.sub(r"```[\w]*\n.*?```", "", text, flags=re.DOTALL).strip()


def make_code_panel(code: str, title: str, border_style: str = "blue") -> Panel:
    """Create a Rich Panel with syntax-highlighted Python code."""
    syntax = Syntax(code, "python", theme="monokai", line_numbers=False)
    return Panel(syntax, title=f"{title} [dim]in[/dim]", border_style=border_style, expand=False)


def make_output_panel(output: str, title: str, border_style: str = "blue", is_error: bool = False) -> Panel:
    """Create a Rich Panel for execution output."""
    content = f"[red]{output}[/red]" if is_error else output
    return Panel(content, title=f"{title} [dim]out[/dim]", border_style=border_style, expand=False)


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

    .user-message {
        border: solid green;
        padding: 1;
        margin: 1 0;
    }

    .assistant-message {
        border: solid cyan;
        padding: 1;
        margin: 1 0;
    }

    .chat-loading {
        height: 3;
    }

    .mode-label {
        height: 1;
        background: $primary;
        color: $text;
        text-style: bold;
        padding: 0 1;
    }

    .mode-label.repl-mode {
        background: $success;
    }
    """

    BINDINGS = [
        ("ctrl+tab", "toggle_mode", "Toggle Chat/REPL"),
        ("f7", "new_session", "New Tab"),
        ("f8", "close_session", "Close Tab"),
        ("f9", "prev_tab", "Prev Tab"),
        ("f10", "next_tab", "Next Tab"),
    ]

    def __init__(self):
        super().__init__()
        self.store = Store()
        self.sessions: dict[str, Session] = {}
        self.agents: dict[str, REPLAgent] = {}
        self.modes: dict[str, str] = {}  # session_id -> "chat" or "repl"

    def compose(self):
        yield Header()
        # Create initial session during compose
        session = Session(store=self.store)
        self.sessions[session.id] = session
        self.modes[session.id] = "chat"  # Default to chat mode
        with TabbedContent(id="sessions"):
            with TabPane(f"Session {session.id}", id=f"tab-{session.id}"):
                yield Vertical(
                    Horizontal(
                        VerticalScroll(id=f"chat-{session.id}", classes="chat-pane"),
                        RichLog(id=f"repl-{session.id}", classes="repl-pane", wrap=True, markup=True),
                    ),
                    Static("[CHAT] Ctrl+Tab to switch", id=f"mode-{session.id}", classes="mode-label"),
                    InputArea(id=f"input-{session.id}", classes="input-area", language="python"),
                    classes="session-content",
                )
        yield Footer()

    def on_mount(self):
        # Wire up logging handlers for the initial session
        for session in self.sessions.values():
            self._wire_session_handlers(session)
            # Focus the input
            self.query_one(f"#input-{session.id}", InputArea).focus()

    def _wire_session_handlers(self, session: Session):
        """Wire logging handlers for a session's widgets."""
        repl = self.query_one(f"#repl-{session.id}", RichLog)

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
        self.modes[session.id] = "chat"  # Default to chat mode

        tabs = self.query_one("#sessions", TabbedContent)
        label = f"↳ {session.id}" if parent else f"Session {session.id}"

        # Create inline widgets (custom Widget classes don't work in TabPane)
        content = Vertical(
            Horizontal(
                VerticalScroll(id=f"chat-{session.id}", classes="chat-pane"),
                RichLog(id=f"repl-{session.id}", classes="repl-pane", wrap=True, markup=True),
            ),
            Static("[CHAT] Ctrl+Tab to switch", id=f"mode-{session.id}", classes="mode-label"),
            InputArea(id=f"input-{session.id}", classes="input-area", language="python"),
            classes="session-content",
        )
        pane = TabPane(label, content, id=f"tab-{session.id}")
        tabs.add_pane(pane)

        # Wire handlers and focus input after mount
        def after_mount():
            self._wire_session_handlers(session)
            self.query_one(f"#input-{session.id}", InputArea).focus()

        self.call_after_refresh(after_mount)

        return session

    def action_new_session(self):
        self.spawn_session()

    def action_close_session(self):
        tabs = self.query_one("#sessions", TabbedContent)
        if len(tabs._tab_content) > 1:
            active_tab_id = tabs.active
            if active_tab_id and active_tab_id.startswith("tab-"):
                session_id = active_tab_id[4:]
                self.sessions.pop(session_id, None)
                self.agents.pop(session_id, None)
                self.modes.pop(session_id, None)
            tabs.remove_pane(active_tab_id)

    def action_next_tab(self):
        tabs = self.query_one("#sessions", TabbedContent)
        tabs.action_next_tab()

    def action_prev_tab(self):
        tabs = self.query_one("#sessions", TabbedContent)
        tabs.action_previous_tab()

    def action_toggle_mode(self):
        """Toggle between chat and REPL mode."""
        session = self._get_active_session()
        if not session:
            return
        current = self.modes.get(session.id, "chat")
        new_mode = "repl" if current == "chat" else "chat"
        self.modes[session.id] = new_mode

        # Update mode label
        mode_label = self.query_one(f"#mode-{session.id}", Static)
        if new_mode == "repl":
            mode_label.update("[REPL] Ctrl+Tab to switch")
            mode_label.add_class("repl-mode")
        else:
            mode_label.update("[CHAT] Ctrl+Tab to switch")
            mode_label.remove_class("repl-mode")

    def _get_active_session(self) -> Session | None:
        """Get the currently active session."""
        tabs = self.query_one("#sessions", TabbedContent)
        active_tab_id = tabs.active
        if active_tab_id and active_tab_id.startswith("tab-"):
            session_id = active_tab_id[4:]  # Remove "tab-" prefix
            return self.sessions.get(session_id)
        return None

    async def on_input_area_submit(self, event: InputArea.Submit) -> None:
        """Handle input submission based on current mode."""
        # Extract session id from input id (format: input-{session_id})
        if not event.input_id.startswith("input-"):
            return
        session_id = event.input_id[6:]
        session = self.sessions.get(session_id)
        if not session:
            return

        mode = self.modes.get(session_id, "chat")
        if mode == "repl":
            self._submit_to_repl(session, event.text)
        else:
            await self._submit_to_chat(session, event.text)

    def _submit_to_repl(self, session: Session, text: str) -> None:
        """Execute code in the REPL."""
        code = text.strip()
        if not code:
            return

        repl_pane = self.query_one(f"#repl-{session.id}", RichLog)

        # Show code in a panel
        repl_pane.write(make_code_panel(code, "You", "green"))

        # Execute and show output in panel
        stdout, errors = session.interpreter.run(code)
        if errors:
            repl_pane.write(make_output_panel(errors, "You", "green", is_error=True))
        elif stdout:
            repl_pane.write(make_output_panel(stdout, "You", "green"))

    async def _submit_to_chat(self, session: Session, text: str) -> None:
        """Send input to chat (Claude)."""
        prompt = text.strip()
        if not prompt:
            return

        chat_pane = self.query_one(f"#chat-{session.id}", VerticalScroll)

        # Add user message
        user_msg = Static(f"[bold]You:[/bold] {prompt}", classes="user-message", markup=True)
        chat_pane.mount(user_msg)

        # Add loading indicator
        loading = LoadingIndicator(classes="chat-loading")
        chat_pane.mount(loading)
        chat_pane.scroll_end()

        # Get or create agent for this session
        if session.id not in self.agents:
            self.agents[session.id] = REPLAgent(session=session)

        agent = self.agents[session.id]
        repl_pane = self.query_one(f"#repl-{session.id}", RichLog)

        # Callback to log code execution to REPL pane
        def on_execution(code: str, output: str, is_error: bool):
            repl_pane.write(make_code_panel(code, "Claude", "cyan"))
            if is_error:
                repl_pane.write(make_output_panel(output, "Claude", "cyan", is_error=True))
            elif output and output != "(no output)":
                repl_pane.write(make_output_panel(output, "Claude", "cyan"))

        try:
            response = await agent.ask(prompt, streaming_handler=None, on_execution=on_execution)
            # Remove loading, add response (without code blocks)
            loading.remove()
            text_only = strip_code_blocks(response)
            if text_only:
                container = Vertical(
                    Static("[bold]Claude:[/bold]", markup=True),
                    Markdown(text_only),
                    classes="assistant-message",
                )
                chat_pane.mount(container)
            else:
                # Show that we got a response but it was all code
                chat_pane.mount(Static("[dim]Claude executed code (see REPL pane)[/dim]", markup=True))
        except Exception as e:
            loading.remove()
            error_msg = Static(f"[bold]Claude:[/bold] [red]Error: {e}[/red]", classes="assistant-message", markup=True)
            chat_pane.mount(error_msg)
            self.notify(f"Error: {e}", severity="error")

        chat_pane.scroll_end()
