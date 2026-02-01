# XML I/O Integration Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire the `mahtab/io/` module into the existing codebase.

**Architecture:** Update system prompt to instruct Claude to use XML tags. Initialize logging at startup. Route parsed responses through the logger. Log user input.

**Tech Stack:** mahtab.io module, existing codebase

---

## Task 1: Create MemoryStore

**Files:**
- Create: `mahtab/io/store.py`
- Create: `tests/test_io_store.py`

**Step 1: Write the failing test**

```python
# tests/test_io_store.py
"""Tests for memory store."""

from mahtab.io.store import MemoryStore


def test_memory_store_append():
    store = MemoryStore()
    store.append(b"hello")
    store.append(b" world")
    assert store.data == b"hello world"


def test_memory_store_clear():
    store = MemoryStore()
    store.append(b"data")
    store.clear()
    assert store.data == b""
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_io_store.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# mahtab/io/store.py
"""In-memory message store."""

from __future__ import annotations


class MemoryStore:
    """Simple in-memory byte store."""

    def __init__(self) -> None:
        self.data = bytearray()

    def append(self, data: bytes) -> None:
        self.data.extend(data)

    def clear(self) -> None:
        self.data.clear()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_io_store.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mahtab/io/store.py tests/test_io_store.py
git commit -m "feat(io): add MemoryStore for session context"
```

---

## Task 2: Export MemoryStore from io package

**Files:**
- Modify: `mahtab/io/__init__.py`

**Step 1: Write the failing test**

Add to `tests/test_io_tags.py`:

```python
def test_memory_store_exported():
    from mahtab.io import MemoryStore
    assert MemoryStore is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_io_tags.py::test_memory_store_exported -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Add to `mahtab/io/__init__.py`:

```python
from mahtab.io.store import MemoryStore

# Add to __all__:
__all__ = [
    # ... existing exports ...
    "MemoryStore",
]
```

**Step 4: Run test to verify it passes**

**Step 5: Commit**

```bash
git add mahtab/io/__init__.py tests/test_io_tags.py
git commit -m "feat(io): export MemoryStore from package"
```

---

## Task 3: Update system prompt for XML tags

**Files:**
- Modify: `mahtab/llm/prompts.py`
- Modify: `tests/test_prompts.py`

**Step 1: Read current prompts.py**

Understand the current system prompt structure.

**Step 2: Add XML tag instructions**

Add to the system prompt telling Claude to wrap responses:
- Natural language in `<assistant-chat>...</assistant-chat>`
- Python code in `<assistant-repl-in>...</assistant-repl-in>`

**Step 3: Write test**

```python
def test_system_prompt_includes_xml_instructions():
    prompt = build_repl_system_prompt()
    assert "<assistant-chat>" in prompt
    assert "<assistant-repl-in>" in prompt
```

**Step 4: Commit**

```bash
git add mahtab/llm/prompts.py tests/test_prompts.py
git commit -m "feat(prompts): add XML tag instructions for structured output"
```

---

## Task 4: Initialize logging in run_repl

**Files:**
- Modify: `mahtab/repl/interactive.py`

**Step 1: Add imports and initialization**

At the top of `run_repl()`:

```python
from mahtab.io import setup_logging, MemoryStore

def run_repl(ns: dict | None = None) -> None:
    # ... existing code ...

    # Initialize structured I/O
    store = MemoryStore()
    log, prompt_handler = setup_logging(store)
```

**Step 2: Store references on session or agent for later use**

The `prompt_handler` is needed to get context. Store it somewhere accessible.

**Step 3: Test manually**

Run `python -m mahtab` and verify no errors.

**Step 4: Commit**

```bash
git add mahtab/repl/interactive.py
git commit -m "feat(repl): initialize structured I/O logging"
```

---

## Task 5: Log user chat input

**Files:**
- Modify: `mahtab/repl/interactive.py`

**Step 1: Log when user sends a prompt**

In the `ask()` function inside `run_repl()`:

```python
import logging

def ask(prompt: str = "") -> None:
    if not prompt:
        return

    # Log user input
    log = logging.getLogger("mahtab")
    log.info(prompt, extra={"tag": "user-chat"})

    # ... rest of existing code ...
```

**Step 2: Test manually**

Run the REPL, send a prompt, verify logging works.

**Step 3: Commit**

```bash
git add mahtab/repl/interactive.py
git commit -m "feat(repl): log user chat input"
```

---

## Task 6: Route assistant response through parser

**Files:**
- Modify: `mahtab/agent/repl_agent.py`

**Step 1: After graph completes, route the response**

In `REPLAgent.ask()`:

```python
from mahtab.io import route_response

async def ask(self, prompt: str, ...) -> str:
    # ... existing code ...

    result = await self._graph.ainvoke(initial_state, config={"callbacks": callbacks})

    final_response = result["current_response"]

    # Route structured response through logger
    route_response(final_response)

    # ... rest of existing code ...
```

**Step 2: Test manually**

Run the REPL, send a prompt, verify assistant response is routed.

**Step 3: Commit**

```bash
git add mahtab/agent/repl_agent.py
git commit -m "feat(agent): route assistant responses through io logger"
```

---

## Task 7: Log code execution

**Files:**
- Modify: `mahtab/agent/graph.py`

**Step 1: Log assistant code and output in execute_node**

```python
import logging

def execute_node(state: AgentState) -> dict:
    from mahtab.core.executor import execute_code

    log = logging.getLogger("mahtab")
    session = state["session"]
    results = []
    on_execution = state.get("on_execution")

    for block in state["code_blocks"]:
        # Log the code being executed
        log.info(block, extra={"tag": "assistant-repl-in"})

        output, is_error = execute_code(block, session)
        results.append((output, is_error))

        # Log the execution output
        log.info(output, extra={"tag": "assistant-repl-out"})

        if on_execution:
            on_execution(output, is_error)

    return {"execution_results": results}
```

**Step 2: Test manually**

Run the REPL, ask Claude to execute code, verify code and output are logged.

**Step 3: Commit**

```bash
git add mahtab/agent/graph.py
git commit -m "feat(graph): log code execution through io logger"
```

---

## Task 8: Stream tokens through logger

**Files:**
- Modify: `mahtab/ui/streaming.py`

**Step 1: Log each token**

In `StreamingHandler.process_token()`:

```python
import logging

def process_token(self, token: str) -> None:
    # Log token for streaming display
    log = logging.getLogger("mahtab")
    log.info(token, extra={"tag": "assistant-chat-stream"})

    # ... rest of existing processing ...
```

**Step 2: Test manually**

Run the REPL, verify tokens stream correctly.

**Step 3: Commit**

```bash
git add mahtab/ui/streaming.py
git commit -m "feat(streaming): log tokens through io logger"
```

---

## Task 9: Run full test suite

**Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests pass

**Step 2: Fix any issues**

**Step 3: Commit fixes if needed**

---

## Summary

After integration:
- `run_repl()` initializes logging with `setup_logging(store)`
- User chat input logged as `user-chat`
- Claude's XML-tagged response parsed and routed
- Code execution logged as `assistant-repl-in` and `assistant-repl-out`
- Streaming tokens logged as `assistant-chat-stream`
