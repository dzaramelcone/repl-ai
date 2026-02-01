# XML I/O Protocol Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement structured XML message routing using Python's logging module.

**Architecture:** Single logger with message tags in record metadata. Three handler types (Prompt, Display, Store) with formatters and filters. Parser extracts XML tags from Claude responses and routes to appropriate channels.

**Tech Stack:** Python logging, Rich console, existing StreamingHandler

---

## Task 1: Tag Constants

**Files:**
- Create: `mahtab/io/__init__.py`
- Create: `mahtab/io/tags.py`
- Create: `tests/test_io_tags.py`

**Step 1: Write the failing test**

```python
# tests/test_io_tags.py
"""Tests for IO tag constants."""

from mahtab.io.tags import TAGS, STREAM_TAGS, COMPLETE_TAGS


def test_tags_contains_all_six():
    expected = {
        "user-repl-in",
        "user-repl-out",
        "assistant-repl-in",
        "assistant-repl-out",
        "user-chat",
        "assistant-chat",
    }
    assert TAGS == expected


def test_stream_tags():
    assert STREAM_TAGS == {"assistant-chat-stream"}


def test_complete_tags_excludes_stream():
    assert "assistant-chat-stream" not in COMPLETE_TAGS
    assert "assistant-chat" in COMPLETE_TAGS
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_io_tags.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'mahtab.io'"

**Step 3: Write minimal implementation**

```python
# mahtab/io/__init__.py
"""Structured I/O with XML tags and logging-based routing."""
```

```python
# mahtab/io/tags.py
"""Tag constants for message routing."""

TAGS: set[str] = {
    "user-repl-in",
    "user-repl-out",
    "assistant-repl-in",
    "assistant-repl-out",
    "user-chat",
    "assistant-chat",
}

STREAM_TAGS: set[str] = {"assistant-chat-stream"}

COMPLETE_TAGS: set[str] = TAGS  # Excludes stream tags
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_io_tags.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mahtab/io/__init__.py mahtab/io/tags.py tests/test_io_tags.py
git commit -m "feat(io): add tag constants for message routing"
```

---

## Task 2: TagFilter

**Files:**
- Create: `mahtab/io/filters.py`
- Create: `tests/test_io_filters.py`

**Step 1: Write the failing test**

```python
# tests/test_io_filters.py
"""Tests for logging filters."""

import logging

from mahtab.io.filters import TagFilter


def test_tag_filter_allows_matching_tag():
    f = TagFilter({"user-chat", "assistant-chat"})
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="hello", args=(), exc_info=None
    )
    record.tag = "user-chat"
    assert f.filter(record) is True


def test_tag_filter_blocks_non_matching_tag():
    f = TagFilter({"user-chat"})
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="hello", args=(), exc_info=None
    )
    record.tag = "assistant-chat"
    assert f.filter(record) is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_io_filters.py -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

**Step 3: Write minimal implementation**

```python
# mahtab/io/filters.py
"""Logging filters for tag-based routing."""

from __future__ import annotations

import logging


class TagFilter(logging.Filter):
    """Filter log records by tag attribute."""

    def __init__(self, tags: set[str]) -> None:
        super().__init__()
        self.tags = tags

    def filter(self, record: logging.LogRecord) -> bool:
        return record.tag in self.tags
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_io_filters.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mahtab/io/filters.py tests/test_io_filters.py
git commit -m "feat(io): add TagFilter for log record routing"
```

---

## Task 3: XMLFormatter

**Files:**
- Create: `mahtab/io/formatters.py`
- Create: `tests/test_io_formatters.py`

**Step 1: Write the failing test**

```python
# tests/test_io_formatters.py
"""Tests for logging formatters."""

import logging

from mahtab.io.formatters import XMLFormatter


def test_xml_formatter_wraps_in_tag():
    f = XMLFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="x = 5", args=(), exc_info=None
    )
    record.tag = "user-repl-in"
    result = f.format(record)
    assert result == "<user-repl-in>x = 5</user-repl-in>"


def test_xml_formatter_handles_multiline():
    f = XMLFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="line1\nline2", args=(), exc_info=None
    )
    record.tag = "assistant-chat"
    result = f.format(record)
    assert result == "<assistant-chat>line1\nline2</assistant-chat>"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_io_formatters.py::test_xml_formatter_wraps_in_tag -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

```python
# mahtab/io/formatters.py
"""Logging formatters for different output targets."""

from __future__ import annotations

import logging


class XMLFormatter(logging.Formatter):
    """Wraps log message in XML tag from record.tag attribute."""

    def format(self, record: logging.LogRecord) -> str:
        return f"<{record.tag}>{record.getMessage()}</{record.tag}>"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_io_formatters.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mahtab/io/formatters.py tests/test_io_formatters.py
git commit -m "feat(io): add XMLFormatter for prompt context"
```

---

## Task 4: RichFormatter

**Files:**
- Modify: `mahtab/io/formatters.py`
- Modify: `tests/test_io_formatters.py`

**Step 1: Write the failing test**

Add to `tests/test_io_formatters.py`:

```python
from mahtab.io.formatters import XMLFormatter, RichFormatter


def test_rich_formatter_user_repl_in():
    f = RichFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="x = 5", args=(), exc_info=None
    )
    record.tag = "user-repl-in"
    result = f.format(record)
    assert result == "[bold cyan]>>> [/]x = 5"


def test_rich_formatter_user_chat():
    f = RichFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="hello", args=(), exc_info=None
    )
    record.tag = "user-chat"
    result = f.format(record)
    assert result == "[bold green]You:[/] hello"


def test_rich_formatter_assistant_chat():
    f = RichFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="hi there", args=(), exc_info=None
    )
    record.tag = "assistant-chat"
    result = f.format(record)
    assert result == "[bold blue]Claude:[/] hi there"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_io_formatters.py::test_rich_formatter_user_repl_in -v`
Expected: FAIL with "ImportError" for RichFormatter

**Step 3: Write minimal implementation**

Add to `mahtab/io/formatters.py`:

```python
class RichFormatter(logging.Formatter):
    """Formats log messages with Rich markup based on tag."""

    def format(self, record: logging.LogRecord) -> str:
        content = record.getMessage()

        match record.tag:
            case "user-repl-in":
                return f"[bold cyan]>>> [/]{content}"
            case "user-repl-out":
                return content
            case "assistant-repl-in":
                return f"[bold magenta]>>> [/]{content}"
            case "assistant-repl-out":
                return content
            case "user-chat":
                return f"[bold green]You:[/] {content}"
            case "assistant-chat":
                return f"[bold blue]Claude:[/] {content}"
            case _:
                return content
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_io_formatters.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mahtab/io/formatters.py tests/test_io_formatters.py
git commit -m "feat(io): add RichFormatter for terminal display"
```

---

## Task 5: BytesFormatter

**Files:**
- Modify: `mahtab/io/formatters.py`
- Modify: `tests/test_io_formatters.py`

**Step 1: Write the failing test**

Add to `tests/test_io_formatters.py`:

```python
from mahtab.io.formatters import XMLFormatter, RichFormatter, BytesFormatter


def test_bytes_formatter_returns_bytes():
    f = BytesFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="hello", args=(), exc_info=None
    )
    record.tag = "user-chat"
    result = f.format(record)
    assert result == b"<user-chat>hello</user-chat>"
    assert isinstance(result, bytes)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_io_formatters.py::test_bytes_formatter_returns_bytes -v`
Expected: FAIL with "ImportError" for BytesFormatter

**Step 3: Write minimal implementation**

Add to `mahtab/io/formatters.py`:

```python
class BytesFormatter(logging.Formatter):
    """Formats log messages as UTF-8 encoded XML bytes."""

    def format(self, record: logging.LogRecord) -> bytes:
        return f"<{record.tag}>{record.getMessage()}</{record.tag}>".encode("utf-8")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_io_formatters.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mahtab/io/formatters.py tests/test_io_formatters.py
git commit -m "feat(io): add BytesFormatter for store serialization"
```

---

## Task 6: PromptHandler

**Files:**
- Create: `mahtab/io/handlers.py`
- Create: `tests/test_io_handlers.py`

**Step 1: Write the failing test**

```python
# tests/test_io_handlers.py
"""Tests for logging handlers."""

import logging

from mahtab.io.handlers import PromptHandler


def test_prompt_handler_accumulates_xml():
    handler = PromptHandler()
    log = logging.getLogger("test_prompt")
    log.addHandler(handler)
    log.setLevel(logging.INFO)

    record1 = log.makeRecord("test", logging.INFO, "", 0, "hello", (), None)
    record1.tag = "user-chat"
    handler.emit(record1)

    record2 = log.makeRecord("test", logging.INFO, "", 0, "hi there", (), None)
    record2.tag = "assistant-chat"
    handler.emit(record2)

    context = handler.get_context()
    assert "<user-chat>hello</user-chat>" in context
    assert "<assistant-chat>hi there</assistant-chat>" in context


def test_prompt_handler_clear():
    handler = PromptHandler()

    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="hello", args=(), exc_info=None
    )
    record.tag = "user-chat"
    handler.emit(record)

    assert handler.get_context() != ""
    handler.clear()
    assert handler.get_context() == ""
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_io_handlers.py::test_prompt_handler_accumulates_xml -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

```python
# mahtab/io/handlers.py
"""Logging handlers for message routing."""

from __future__ import annotations

import logging

from mahtab.io.formatters import XMLFormatter


class PromptHandler(logging.Handler):
    """Accumulates XML-formatted messages for Claude's context."""

    def __init__(self) -> None:
        super().__init__()
        self.buffer: list[str] = []
        self.setFormatter(XMLFormatter())

    def emit(self, record: logging.LogRecord) -> None:
        self.buffer.append(self.format(record))

    def get_context(self) -> str:
        return "\n".join(self.buffer)

    def clear(self) -> None:
        self.buffer.clear()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_io_handlers.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mahtab/io/handlers.py tests/test_io_handlers.py
git commit -m "feat(io): add PromptHandler for context accumulation"
```

---

## Task 7: Store Protocol and StoreHandler

**Files:**
- Modify: `mahtab/io/handlers.py`
- Modify: `tests/test_io_handlers.py`

**Step 1: Write the failing test**

Add to `tests/test_io_handlers.py`:

```python
from mahtab.io.handlers import PromptHandler, StoreHandler, Store


class MockStore:
    """Mock store for testing."""

    def __init__(self) -> None:
        self.data = bytearray()

    def append(self, data: bytes) -> None:
        self.data.extend(data)


def test_store_handler_appends_bytes():
    store = MockStore()
    handler = StoreHandler(store)

    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="hello", args=(), exc_info=None
    )
    record.tag = "user-chat"
    handler.emit(record)

    assert store.data == b"<user-chat>hello</user-chat>"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_io_handlers.py::test_store_handler_appends_bytes -v`
Expected: FAIL with "ImportError" for StoreHandler

**Step 3: Write minimal implementation**

Add to `mahtab/io/handlers.py`:

```python
from typing import Protocol

from mahtab.io.formatters import XMLFormatter, BytesFormatter


class Store(Protocol):
    """Protocol for message stores."""

    def append(self, data: bytes) -> None: ...


class StoreHandler(logging.Handler):
    """Appends bytes to a store."""

    def __init__(self, store: Store) -> None:
        super().__init__()
        self.store = store
        self.setFormatter(BytesFormatter())

    def emit(self, record: logging.LogRecord) -> None:
        self.store.append(self.format(record))
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_io_handlers.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mahtab/io/handlers.py tests/test_io_handlers.py
git commit -m "feat(io): add Store protocol and StoreHandler"
```

---

## Task 8: DisplayHandler

**Files:**
- Modify: `mahtab/io/handlers.py`
- Modify: `tests/test_io_handlers.py`

**Step 1: Write the failing test**

Add to `tests/test_io_handlers.py`:

```python
from io import StringIO
from unittest.mock import MagicMock

from rich.console import Console

from mahtab.io.handlers import PromptHandler, StoreHandler, Store, DisplayHandler


def test_display_handler_prints_formatted():
    output = StringIO()
    console = Console(file=output, force_terminal=True)
    handler = DisplayHandler(console)

    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="hello", args=(), exc_info=None
    )
    record.tag = "user-chat"
    handler.emit(record)

    result = output.getvalue()
    assert "You:" in result
    assert "hello" in result


def test_display_handler_streams_tokens():
    console = MagicMock(spec=Console)
    handler = DisplayHandler(console)

    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="tok", args=(), exc_info=None
    )
    record.tag = "assistant-chat-stream"
    handler.emit(record)

    handler.streamer.process_token.assert_called_once_with("tok")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_io_handlers.py::test_display_handler_prints_formatted -v`
Expected: FAIL with "ImportError" for DisplayHandler

**Step 3: Write minimal implementation**

Add to `mahtab/io/handlers.py`:

```python
from rich.console import Console

from mahtab.io.formatters import XMLFormatter, BytesFormatter, RichFormatter
from mahtab.ui.streaming import StreamingHandler


class DisplayHandler(logging.Handler):
    """Routes messages to terminal display."""

    def __init__(self, console: Console) -> None:
        super().__init__()
        self.console = console
        self.setFormatter(RichFormatter())
        self.streamer = StreamingHandler(console)

    def emit(self, record: logging.LogRecord) -> None:
        match record.tag:
            case "assistant-chat-stream":
                self.streamer.process_token(record.getMessage())
            case _:
                self.console.print(self.format(record))
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_io_handlers.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mahtab/io/handlers.py tests/test_io_handlers.py
git commit -m "feat(io): add DisplayHandler with streaming support"
```

---

## Task 9: Response Parser

**Files:**
- Create: `mahtab/io/parser.py`
- Create: `tests/test_io_parser.py`

**Step 1: Write the failing test**

```python
# tests/test_io_parser.py
"""Tests for response parsing."""

from mahtab.io.parser import parse_response


def test_parse_response_extracts_chat():
    response = "<assistant-chat>Hello there!</assistant-chat>"
    result = parse_response(response)
    assert result == [("assistant-chat", "Hello there!")]


def test_parse_response_extracts_repl():
    response = "<assistant-repl-in>x = 5</assistant-repl-in>"
    result = parse_response(response)
    assert result == [("assistant-repl-in", "x = 5")]


def test_parse_response_extracts_multiple():
    response = """<assistant-chat>Let me calculate that.</assistant-chat>
<assistant-repl-in>result = 2 + 2</assistant-repl-in>"""
    result = parse_response(response)
    assert len(result) == 2
    assert result[0] == ("assistant-chat", "Let me calculate that.")
    assert result[1] == ("assistant-repl-in", "result = 2 + 2")


def test_parse_response_multiline_content():
    response = """<assistant-repl-in>def foo():
    return 42</assistant-repl-in>"""
    result = parse_response(response)
    assert result == [("assistant-repl-in", "def foo():\n    return 42")]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_io_parser.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# mahtab/io/parser.py
"""Parse XML-tagged responses from Claude."""

from __future__ import annotations

import re


def parse_response(response: str) -> list[tuple[str, str]]:
    """Extract (tag, content) tuples from XML-tagged response."""
    pattern = r"<(assistant-(?:chat|repl-in|repl-out))>(.*?)</\1>"
    return re.findall(pattern, response, re.DOTALL)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_io_parser.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mahtab/io/parser.py tests/test_io_parser.py
git commit -m "feat(io): add response parser for XML extraction"
```

---

## Task 10: route_response Function

**Files:**
- Modify: `mahtab/io/parser.py`
- Modify: `tests/test_io_parser.py`

**Step 1: Write the failing test**

Add to `tests/test_io_parser.py`:

```python
import logging

from mahtab.io.parser import parse_response, route_response
from mahtab.io.handlers import PromptHandler


def test_route_response_sends_to_logger():
    # Set up logger with prompt handler
    log = logging.getLogger("mahtab")
    log.setLevel(logging.INFO)
    handler = PromptHandler()
    log.addHandler(handler)

    try:
        response = "<assistant-chat>Hello!</assistant-chat>"
        route_response(response)

        context = handler.get_context()
        assert "<assistant-chat>Hello!</assistant-chat>" in context
    finally:
        log.removeHandler(handler)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_io_parser.py::test_route_response_sends_to_logger -v`
Expected: FAIL with "ImportError" for route_response

**Step 3: Write minimal implementation**

Add to `mahtab/io/parser.py`:

```python
import logging


def route_response(response: str) -> None:
    """Parse response and route each tagged section to the logger."""
    log = logging.getLogger("mahtab")
    for tag, content in parse_response(response):
        log.info(content, extra={"tag": tag})
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_io_parser.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mahtab/io/parser.py tests/test_io_parser.py
git commit -m "feat(io): add route_response for message routing"
```

---

## Task 11: Setup Function

**Files:**
- Create: `mahtab/io/setup.py`
- Create: `tests/test_io_setup.py`

**Step 1: Write the failing test**

```python
# tests/test_io_setup.py
"""Tests for logger setup."""

import logging

from mahtab.io.setup import setup_logging
from mahtab.io.handlers import PromptHandler, DisplayHandler, StoreHandler


class MockStore:
    def __init__(self) -> None:
        self.data = bytearray()

    def append(self, data: bytes) -> None:
        self.data.extend(data)


def test_setup_logging_returns_logger_and_prompt():
    store = MockStore()
    log, prompt = setup_logging(store)

    assert isinstance(log, logging.Logger)
    assert log.name == "mahtab"
    assert isinstance(prompt, PromptHandler)


def test_setup_logging_routes_to_all_handlers():
    store = MockStore()
    log, prompt = setup_logging(store)

    # Clear any prior state
    prompt.clear()
    store.data.clear()

    log.info("hello", extra={"tag": "user-chat"})

    # Check prompt handler received it
    assert "<user-chat>hello</user-chat>" in prompt.get_context()

    # Check store received it
    assert b"<user-chat>hello</user-chat>" in store.data
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_io_setup.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# mahtab/io/setup.py
"""Logger setup and wiring."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mahtab.io.filters import TagFilter
from mahtab.io.handlers import DisplayHandler, PromptHandler, StoreHandler
from mahtab.io.tags import COMPLETE_TAGS, STREAM_TAGS
from mahtab.ui.console import console

if TYPE_CHECKING:
    from mahtab.io.handlers import Store


def setup_logging(store: Store) -> tuple[logging.Logger, PromptHandler]:
    """Configure the mahtab logger with all handlers."""
    log = logging.getLogger("mahtab")
    log.setLevel(logging.INFO)

    # Prompt handler: all complete message tags
    prompt = PromptHandler()
    prompt.addFilter(TagFilter(COMPLETE_TAGS))
    log.addHandler(prompt)

    # Display handler: complete tags except assistant-chat (streamed), plus stream tag
    display_tags = (COMPLETE_TAGS - {"assistant-chat"}) | STREAM_TAGS
    display = DisplayHandler(console)
    display.addFilter(TagFilter(display_tags))
    log.addHandler(display)

    # Store handler: all complete message tags
    store_handler = StoreHandler(store)
    store_handler.addFilter(TagFilter(COMPLETE_TAGS))
    log.addHandler(store_handler)

    return log, prompt
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_io_setup.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mahtab/io/setup.py tests/test_io_setup.py
git commit -m "feat(io): add setup_logging for handler wiring"
```

---

## Task 12: Export Public API

**Files:**
- Modify: `mahtab/io/__init__.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_io_tags.py or create tests/test_io_api.py
def test_public_api_exports():
    from mahtab.io import (
        TAGS,
        STREAM_TAGS,
        COMPLETE_TAGS,
        TagFilter,
        XMLFormatter,
        RichFormatter,
        BytesFormatter,
        PromptHandler,
        DisplayHandler,
        StoreHandler,
        Store,
        parse_response,
        route_response,
        setup_logging,
    )
    # Just verify imports work
    assert TAGS is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_io_tags.py::test_public_api_exports -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

```python
# mahtab/io/__init__.py
"""Structured I/O with XML tags and logging-based routing."""

from mahtab.io.filters import TagFilter
from mahtab.io.formatters import BytesFormatter, RichFormatter, XMLFormatter
from mahtab.io.handlers import DisplayHandler, PromptHandler, Store, StoreHandler
from mahtab.io.parser import parse_response, route_response
from mahtab.io.setup import setup_logging
from mahtab.io.tags import COMPLETE_TAGS, STREAM_TAGS, TAGS

__all__ = [
    "TAGS",
    "STREAM_TAGS",
    "COMPLETE_TAGS",
    "TagFilter",
    "XMLFormatter",
    "RichFormatter",
    "BytesFormatter",
    "PromptHandler",
    "DisplayHandler",
    "StoreHandler",
    "Store",
    "parse_response",
    "route_response",
    "setup_logging",
]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_io_tags.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mahtab/io/__init__.py tests/test_io_tags.py
git commit -m "feat(io): export public API from package"
```

---

## Task 13: Run Full Test Suite

**Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests pass

**Step 2: Check for regressions**

Run: `pytest tests/ -v --tb=short`
Expected: No failures

**Step 3: Commit any fixes if needed**

---

## Summary

After completing all tasks, the `mahtab/io/` module provides:

- Tag constants for message routing
- TagFilter for filtering log records
- XMLFormatter, RichFormatter, BytesFormatter for output formatting
- PromptHandler, DisplayHandler, StoreHandler for message destinations
- parse_response and route_response for inbound message handling
- setup_logging for wiring everything together

Integration with existing code (repl_agent.py, graph.py, etc.) is a separate follow-up task.
