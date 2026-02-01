"""REPLWidget: Textual widget for a single Session."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import RichLog, TextArea

if TYPE_CHECKING:
    from mahtab.session import Session


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
