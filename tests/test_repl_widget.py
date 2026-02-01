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


class WidgetTestApp(App):
    """Test app that wraps a REPLWidget for a given session."""

    def __init__(self, session: Session):
        super().__init__()
        self._session = session

    def compose(self):
        yield REPLWidget(self._session)


def test_repl_widget_has_session(session):
    widget = REPLWidget(session)
    assert widget.session is session


@pytest.mark.asyncio
async def test_repl_widget_has_output_panes(session):
    """Test that REPLWidget composes chat and repl RichLog panes."""
    app = WidgetTestApp(session)
    async with app.run_test():
        chat = app.query_one("#chat", RichLog)
        repl = app.query_one("#repl", RichLog)
        assert chat is not None
        assert repl is not None


@pytest.mark.asyncio
async def test_repl_widget_has_input(session):
    """Test that REPLWidget composes an input TextArea."""
    app = WidgetTestApp(session)
    async with app.run_test():
        input_widget = app.query_one("#input", TextArea)
        assert input_widget is not None


@pytest.mark.asyncio
async def test_repl_widget_logs_to_chat_pane(session):
    app = WidgetTestApp(session)
    async with app.run_test() as pilot:
        # Wait for call_after_refresh to complete
        await pilot.pause()
        widget = app.query_one(REPLWidget)
        session.log_user_chat.info("hello chat")
        chat = widget.query_one("#chat", RichLog)
        # RichLog stores lines internally
        assert len(chat.lines) > 0


@pytest.mark.asyncio
async def test_repl_widget_logs_to_repl_pane(session):
    app = WidgetTestApp(session)
    async with app.run_test() as pilot:
        # Wait for call_after_refresh to complete
        await pilot.pause()
        widget = app.query_one(REPLWidget)
        session.log_user_repl.info(">>> x = 5")
        repl = widget.query_one("#repl", RichLog)
        assert len(repl.lines) > 0


@pytest.mark.asyncio
async def test_repl_widget_logs_to_store(session):
    app = WidgetTestApp(session)
    async with app.run_test() as pilot:
        # Wait for call_after_refresh to complete
        await pilot.pause()
        session.log_user_chat.info("stored message")
        assert b"stored message" in session.store.data


@pytest.mark.asyncio
async def test_repl_widget_exec_logs_to_repl(session):
    app = WidgetTestApp(session)
    async with app.run_test() as pilot:
        await pilot.pause()  # Wait for handlers
        widget = app.query_one(REPLWidget)
        input_area = widget.query_one("#input", TextArea)
        input_area.text = "x = 42"
        await widget.submit()
        repl = widget.query_one("#repl", RichLog)
        assert len(repl.lines) > 0
        assert session.namespace.get("x") == 42


@pytest.mark.asyncio
async def test_repl_widget_exec_error_logs(session):
    app = WidgetTestApp(session)
    async with app.run_test() as pilot:
        await pilot.pause()
        widget = app.query_one(REPLWidget)
        input_area = widget.query_one("#input", TextArea)
        input_area.text = "1/0"
        await widget.submit()
        repl = widget.query_one("#repl", RichLog)
        # Should have logged the error
        assert len(repl.lines) > 0
