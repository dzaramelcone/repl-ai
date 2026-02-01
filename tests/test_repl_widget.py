"""Tests for REPLWidget."""

import pytest
from textual.app import App
from textual.widgets import RichLog, TextArea

from mahtab.session import Session
from mahtab.store import Store
from mahtab.ui.repl_widget import REPLWidget


@pytest.fixture
def session():
    store = Store()
    return Session(store)


def test_repl_widget_has_session(session):
    widget = REPLWidget(session)
    assert widget.session is session


@pytest.mark.asyncio
async def test_repl_widget_has_output_panes(session):
    """Test that REPLWidget composes chat and repl RichLog panes."""

    class TestApp(App):
        def compose(self):
            yield REPLWidget(session)

    app = TestApp()
    async with app.run_test():
        chat = app.query_one("#chat", RichLog)
        repl = app.query_one("#repl", RichLog)
        assert chat is not None
        assert repl is not None


@pytest.mark.asyncio
async def test_repl_widget_has_input(session):
    """Test that REPLWidget composes an input TextArea."""

    class TestApp(App):
        def compose(self):
            yield REPLWidget(session)

    app = TestApp()
    async with app.run_test():
        input_widget = app.query_one("#input", TextArea)
        assert input_widget is not None
