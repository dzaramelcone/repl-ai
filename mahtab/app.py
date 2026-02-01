"""MahtabApp: Main Textual application."""

from __future__ import annotations

from textual.app import App
from textual.widgets import Footer, Header, TabbedContent, TabPane

from mahtab.session import Session
from mahtab.store import Store
from mahtab.ui.repl_widget import REPLWidget


class MahtabApp(App):
    """Main app. Manages sessions as tabs, owns the shared Store."""

    CSS = """
    #sessions {
        height: 1fr;
        width: 100%;
    }

    TabPane {
        height: 100%;
        width: 100%;
        padding: 0;
    }
    """

    BINDINGS = [
        ("cmd+t", "new_session", "New Session"),
        ("cmd+w", "close_session", "Close Session"),
        ("cmd+shift+]", "next_tab", "Next Tab"),
        ("cmd+shift+[", "prev_tab", "Previous Tab"),
    ]

    def __init__(self):
        super().__init__()
        self.store = Store()
        self.sessions: dict[str, Session] = {}

    def compose(self):
        yield Header()
        yield TabbedContent(id="sessions")
        yield Footer()

    def on_mount(self):
        self.spawn_session()

    def spawn_session(
        self,
        parent: Session | None = None,
        context: dict | None = None,
    ) -> Session:
        session = Session(store=self.store, parent=parent, context=context)
        self.sessions[session.id] = session

        tabs = self.query_one("#sessions", TabbedContent)
        label = f"↳ {session.id}" if parent else f"Session {session.id}"
        pane = TabPane(label, REPLWidget(session), id=f"tab-{session.id}")
        tabs.add_pane(pane)

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
