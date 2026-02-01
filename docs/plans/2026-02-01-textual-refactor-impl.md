# Textual Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor Mahtab from Rich-based REPL to Textual TUI with tabbed sessions sharing a Store.

**Architecture:** Store (byte blob) → Session (async REPL with loggers) → REPLWidget (Textual UI) → MahtabApp (tabbed container). All output routes through Python logging; widgets subscribe as handlers.

**Tech Stack:** Textual, Python logging, existing LangGraph agent

**Design Doc:** `docs/plans/2026-02-01-textual-refactor-design.md`

---

## Task 1: Add Textual Dependency

**Files:**
- Modify: `pyproject.toml:6-12`

**Step 1: Add textual to dependencies**

In `pyproject.toml`, add `"textual>=0.50.0"` to the dependencies list:

```toml
dependencies = [
    "langchain-core>=0.3.0",
    "langchain>=0.3.0",
    "langgraph>=0.2.0",
    "pydantic>=2.0.0",
    "rich>=13.0.0",
    "textual>=0.50.0",
]
```

**Step 2: Sync dependencies**

Run: `uv sync`
Expected: Dependencies installed successfully

**Step 3: Verify import works**

Run: `uv run python -c "import textual; print(textual.__version__)"`
Expected: Version number printed (0.50.0 or higher)

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add textual dependency"
```

---

## Task 2: Implement Store

**Files:**
- Create: `mahtab/store.py`
- Test: `tests/test_store.py`

**Step 1: Write the failing test**

Create `tests/test_store.py`:

```python
"""Tests for Store."""

from mahtab.store import Store


def test_store_starts_empty():
    store = Store()
    assert store.data == b""


def test_store_append_bytes():
    store = Store()
    store.append(b"hello")
    assert store.data == b"hello"


def test_store_append_str():
    store = Store()
    store.append("hello")
    assert store.data == b"hello"


def test_store_append_accumulates():
    store = Store()
    store.append(b"hello")
    store.append(b" world")
    assert store.data == b"hello world"


def test_store_load_full():
    store = Store()
    store.append(b"hello world")
    assert store.load() == b"hello world"


def test_store_load_slice():
    store = Store()
    store.append(b"hello world")
    assert store.load(0, 5) == b"hello"
    assert store.load(6, 11) == b"world"


def test_store_load_start_only():
    store = Store()
    store.append(b"hello world")
    assert store.load(6) == b"world"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_store.py -v`
Expected: FAIL with ModuleNotFoundError (mahtab.store doesn't exist)

**Step 3: Write implementation**

Create `mahtab/store.py`:

```python
"""Store: Giant byte blob shared across sessions."""

from __future__ import annotations


class Store:
    """Giant byte blob. All sessions share one instance."""

    def __init__(self):
        self.data: bytes = b""

    def load(self, start: int = 0, end: int | None = None) -> bytes:
        """Read a slice of the blob."""
        return self.data[start:end]

    def append(self, content: bytes | str) -> None:
        """Add to the blob."""
        if isinstance(content, str):
            content = content.encode()
        self.data += content
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_store.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add mahtab/store.py tests/test_store.py
git commit -m "feat: add Store (byte blob for shared context)"
```

---

## Task 3: Implement Logging Handlers

**Files:**
- Create: `mahtab/ui/handlers.py`
- Test: `tests/test_handlers.py`

**Step 1: Write the failing test**

Create `tests/test_handlers.py`:

```python
"""Tests for logging handlers."""

import logging

from mahtab.store import Store
from mahtab.ui.handlers import StoreHandler


def test_store_handler_appends_to_store():
    store = Store()
    handler = StoreHandler(store)
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger = logging.getLogger("test.store_handler")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info("hello world")

    assert b"hello world" in store.data


def test_store_handler_includes_newline():
    store = Store()
    handler = StoreHandler(store)
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger = logging.getLogger("test.store_handler_newline")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info("line1")
    logger.info("line2")

    assert store.data == b"line1\nline2\n"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_handlers.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write implementation**

Create `mahtab/ui/handlers.py`:

```python
"""Logging handlers for routing output."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mahtab.store import Store


class StoreHandler(logging.Handler):
    """Appends log records to the Store."""

    def __init__(self, store: Store):
        super().__init__()
        self.store = store

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.store.append(msg + "\n")
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_handlers.py -v`
Expected: All 2 tests PASS

**Step 5: Commit**

```bash
git add mahtab/ui/handlers.py tests/test_handlers.py
git commit -m "feat: add StoreHandler for logging to Store"
```

---

## Task 4: Implement Session

**Files:**
- Create: `mahtab/session.py`
- Test: `tests/test_session.py`

**Step 1: Write the failing test**

Create `tests/test_session.py`:

```python
"""Tests for Session."""

import logging

from mahtab.session import Session
from mahtab.store import Store


def test_session_has_id():
    store = Store()
    session = Session(store)
    assert len(session.id) == 8  # hex[:8]


def test_session_has_empty_namespace():
    store = Store()
    session = Session(store)
    assert session.namespace == {}


def test_session_has_empty_messages():
    store = Store()
    session = Session(store)
    assert session.messages == []


def test_session_shares_store():
    store = Store()
    s1 = Session(store)
    s2 = Session(store)
    assert s1.store is s2.store


def test_session_exec_updates_namespace():
    store = Store()
    session = Session(store)
    session.exec("x = 42")
    assert session.namespace["x"] == 42


def test_session_exec_can_read_namespace():
    store = Store()
    session = Session(store)
    session.namespace["y"] = 10
    session.exec("z = y * 2")
    assert session.namespace["z"] == 20


def test_session_spawn_creates_child():
    store = Store()
    parent = Session(store)
    child = parent.spawn()
    assert child.parent is parent
    assert child in parent.children
    assert child.store is parent.store


def test_session_spawn_with_context():
    store = Store()
    parent = Session(store)
    child = parent.spawn(context={"query": "find bugs"})
    assert child.namespace["query"] == "find bugs"


def test_session_has_loggers():
    store = Store()
    session = Session(store)
    assert isinstance(session.log_user_repl, logging.Logger)
    assert isinstance(session.log_user_chat, logging.Logger)
    assert isinstance(session.log_llm_repl, logging.Logger)
    assert isinstance(session.log_llm_chat, logging.Logger)


def test_session_logger_names_include_id():
    store = Store()
    session = Session(store)
    assert session.id in session.log_user_repl.name
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_session.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write implementation**

Create `mahtab/session.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_session.py -v`
Expected: All 11 tests PASS

**Step 5: Commit**

```bash
git add mahtab/session.py tests/test_session.py
git commit -m "feat: add Session (async REPL with loggers)"
```

---

## Task 5: Implement REPLWidget

**Files:**
- Create: `mahtab/ui/repl_widget.py`
- Test: `tests/test_repl_widget.py`

**Step 1: Write the failing test**

Create `tests/test_repl_widget.py`:

```python
"""Tests for REPLWidget."""

import pytest
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


async def test_repl_widget_has_output_panes(session):
    widget = REPLWidget(session)
    async with widget.run_test():
        chat = widget.query_one("#chat", RichLog)
        repl = widget.query_one("#repl", RichLog)
        assert chat is not None
        assert repl is not None


async def test_repl_widget_has_input(session):
    widget = REPLWidget(session)
    async with widget.run_test():
        input_widget = widget.query_one("#input", TextArea)
        assert input_widget is not None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_repl_widget.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write implementation**

Create `mahtab/ui/repl_widget.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_repl_widget.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add mahtab/ui/repl_widget.py tests/test_repl_widget.py
git commit -m "feat: add REPLWidget (Textual widget for Session)"
```

---

## Task 6: Implement MahtabApp

**Files:**
- Create: `mahtab/app.py`
- Test: `tests/test_app.py`

**Step 1: Write the failing test**

Create `tests/test_app.py`:

```python
"""Tests for MahtabApp."""

import pytest
from textual.widgets import TabbedContent

from mahtab.app import MahtabApp


async def test_app_has_store():
    app = MahtabApp()
    async with app.run_test():
        assert app.store is not None


async def test_app_has_tabbed_content():
    app = MahtabApp()
    async with app.run_test():
        tabs = app.query_one("#sessions", TabbedContent)
        assert tabs is not None


async def test_app_starts_with_one_session():
    app = MahtabApp()
    async with app.run_test():
        assert len(app.sessions) == 1


async def test_app_spawn_session_adds_tab():
    app = MahtabApp()
    async with app.run_test():
        initial_count = len(app.sessions)
        app.spawn_session()
        assert len(app.sessions) == initial_count + 1


async def test_app_spawn_child_session():
    app = MahtabApp()
    async with app.run_test():
        parent = list(app.sessions.values())[0]
        child = app.spawn_session(parent=parent)
        assert child.parent is parent
        assert child in parent.children
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_app.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write implementation**

Create `mahtab/app.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_app.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add mahtab/app.py tests/test_app.py
git commit -m "feat: add MahtabApp (tabbed Textual container)"
```

---

## Task 7: Wire Up Logging Handlers in REPLWidget

**Files:**
- Modify: `mahtab/ui/repl_widget.py`
- Modify: `tests/test_repl_widget.py`

**Step 1: Write the failing test**

Add to `tests/test_repl_widget.py`:

```python
async def test_repl_widget_logs_to_chat_pane(session):
    widget = REPLWidget(session)
    async with widget.run_test():
        session.log_user_chat.info("hello chat")
        chat = widget.query_one("#chat", RichLog)
        # RichLog stores lines internally
        assert len(chat.lines) > 0


async def test_repl_widget_logs_to_repl_pane(session):
    widget = REPLWidget(session)
    async with widget.run_test():
        session.log_user_repl.info(">>> x = 5")
        repl = widget.query_one("#repl", RichLog)
        assert len(repl.lines) > 0


async def test_repl_widget_logs_to_store(session):
    widget = REPLWidget(session)
    async with widget.run_test():
        session.log_user_chat.info("stored message")
        assert b"stored message" in session.store.data
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_repl_widget.py::test_repl_widget_logs_to_chat_pane -v`
Expected: FAIL (no handlers attached yet)

**Step 3: Update implementation**

Modify `mahtab/ui/repl_widget.py`, add to imports:

```python
import logging

from mahtab.ui.handlers import StoreHandler
```

Add new handler class and update REPLWidget:

```python
class RichLogHandler(logging.Handler):
    """Sends log records to a RichLog widget."""

    def __init__(self, widget: RichLog):
        super().__init__()
        self.widget = widget

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.widget.write(msg)
```

Add `on_mount` method to REPLWidget:

```python
    def on_mount(self):
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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_repl_widget.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add mahtab/ui/repl_widget.py tests/test_repl_widget.py
git commit -m "feat: wire logging handlers to REPLWidget panes"
```

---

## Task 8: Add Input Handling to REPLWidget

**Files:**
- Modify: `mahtab/ui/repl_widget.py`
- Modify: `tests/test_repl_widget.py`

**Step 1: Write the failing test**

Add to `tests/test_repl_widget.py`:

```python
async def test_repl_widget_exec_logs_to_repl(session):
    widget = REPLWidget(session)
    async with widget.run_test():
        input_area = widget.query_one("#input", TextArea)
        input_area.text = "x = 42"
        await widget.submit()
        repl = widget.query_one("#repl", RichLog)
        assert len(repl.lines) > 0
        assert session.namespace.get("x") == 42


async def test_repl_widget_exec_error_logs(session):
    widget = REPLWidget(session)
    async with widget.run_test():
        input_area = widget.query_one("#input", TextArea)
        input_area.text = "1/0"
        await widget.submit()
        repl = widget.query_one("#repl", RichLog)
        # Should have logged the error
        assert len(repl.lines) > 0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_repl_widget.py::test_repl_widget_exec_logs_to_repl -v`
Expected: FAIL (submit method doesn't exist)

**Step 3: Update implementation**

Add to REPLWidget class in `mahtab/ui/repl_widget.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_repl_widget.py -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add mahtab/ui/repl_widget.py tests/test_repl_widget.py
git commit -m "feat: add input handling to REPLWidget"
```

---

## Task 9: Add Entry Point

**Files:**
- Modify: `mahtab/__main__.py`
- Modify: `pyproject.toml`

**Step 1: Update __main__.py**

Replace contents of `mahtab/__main__.py`:

```python
"""Entry point for mahtab."""

from mahtab.app import MahtabApp


def main():
    app = MahtabApp()
    app.run()


if __name__ == "__main__":
    main()
```

**Step 2: Update pyproject.toml entry point**

Change the scripts section in `pyproject.toml`:

```toml
[project.scripts]
mahtab = "mahtab.__main__:main"
```

**Step 3: Test the entry point**

Run: `uv run mahtab`
Expected: Textual app launches with one session tab

Press: `Cmd+Q` or `Ctrl+C` to exit

**Step 4: Commit**

```bash
git add mahtab/__main__.py pyproject.toml
git commit -m "feat: update entry point for Textual app"
```

---

## Task 10: Manual Integration Test

**No code changes - verification only**

**Step 1: Launch the app**

Run: `uv run mahtab`

**Step 2: Test REPL functionality**

Type in input area:
```python
x = 5
```
Press Ctrl+Enter

Expected: REPL pane shows `>>> x = 5`

Type:
```python
x * 2
```
Press Ctrl+Enter

Expected: REPL pane shows `>>> x * 2` and `10`

**Step 3: Test tab creation**

Press: `Cmd+T`
Expected: New tab appears

**Step 4: Test tab switching**

Press: `Cmd+Shift+[` and `Cmd+Shift+]`
Expected: Switches between tabs

**Step 5: Test error handling**

Type:
```python
1/0
```
Press Ctrl+Enter

Expected: REPL pane shows error in red

**Step 6: Exit**

Press: `Cmd+Q`

---

## Future Tasks (Not in This Plan)

These are out of scope but noted for future work:

- Wire up Claude/LangGraph agent to Session.ask()
- Add ask() mode detection or toggle
- Implement RLM tools (peek, grep, partition) as LangChain tools
- Add streaming handler integration
- Persist store to disk
- Add session save/load
