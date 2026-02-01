"""Tests for MahtabApp."""

import pytest
from textual.widgets import TabbedContent

from mahtab.app import MahtabApp


@pytest.mark.asyncio
async def test_app_has_store():
    app = MahtabApp()
    async with app.run_test():
        assert app.store is not None


@pytest.mark.asyncio
async def test_app_has_tabbed_content():
    app = MahtabApp()
    async with app.run_test():
        tabs = app.query_one("#sessions", TabbedContent)
        assert tabs is not None


@pytest.mark.asyncio
async def test_app_starts_with_one_session():
    app = MahtabApp()
    async with app.run_test():
        assert len(app.sessions) == 1


@pytest.mark.asyncio
async def test_app_spawn_session_adds_tab():
    app = MahtabApp()
    async with app.run_test():
        initial_count = len(app.sessions)
        app.spawn_session()
        assert len(app.sessions) == initial_count + 1


@pytest.mark.asyncio
async def test_app_spawn_child_session():
    app = MahtabApp()
    async with app.run_test():
        parent = list(app.sessions.values())[0]
        child = app.spawn_session(parent=parent)
        assert child.parent is parent
        assert child in parent.children
