# Streaming Consolidation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Consolidate duplicate agentic loops into single LangGraph-based flow with streaming via LangChain callbacks.

**Architecture:** Wire `run_manager` in `_agenerate` to call `StreamingHandler.process_token()`. REPL passes handler as callback when calling `agent.ask()`.

**Tech Stack:** LangChain callbacks, LangGraph, Rich (streaming UI)

---

### Task 1: Make StreamingHandler a LangChain Callback

**Files:**
- Modify: `mahtab/ui/streaming.py:20-33`
- Test: `tests/test_streaming_callback.py` (new)

**Step 1: Write the failing test**

Create `tests/test_streaming_callback.py`:

```python
"""Tests for StreamingHandler as LangChain callback."""

from langchain_core.callbacks import BaseCallbackHandler

from mahtab.ui.streaming import StreamingHandler


def test_streaming_handler_is_callback():
    """StreamingHandler should be a LangChain BaseCallbackHandler."""
    handler = StreamingHandler()
    assert isinstance(handler, BaseCallbackHandler)


def test_on_llm_new_token_calls_process_token():
    """on_llm_new_token should delegate to process_token."""
    handler = StreamingHandler()
    handler.reset()

    # Call the callback method
    handler.on_llm_new_token("hello")

    # Text should be in the buffer (process_token buffers short text)
    assert "hello" in handler._text_buffer
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_streaming_callback.py -v`
Expected: FAIL - StreamingHandler not a BaseCallbackHandler

**Step 3: Add BaseCallbackHandler inheritance and callback methods**

In `mahtab/ui/streaming.py`, change the import and class definition:

```python
# Add to imports (after line 11)
from langchain_core.callbacks import BaseCallbackHandler

# Change class definition (line 20)
class StreamingHandler(BaseCallbackHandler):
```

Add callback methods after `cleanup()` (after line 178):

```python
    # LangChain callback interface
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called by LangChain when a new token is generated."""
        self.process_token(token)

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        """Called by LangChain when LLM starts generating."""
        self.start_spinner()

    def on_llm_end(self, response, **kwargs) -> None:
        """Called by LangChain when LLM finishes generating."""
        self.flush()
        self.stop_spinner()
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_streaming_callback.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mahtab/ui/streaming.py tests/test_streaming_callback.py
git commit -m "feat: make StreamingHandler a LangChain callback"
```

---

### Task 2: Wire run_manager in _agenerate

**Files:**
- Modify: `mahtab/llm/claude_cli.py:60-79`
- Test: `tests/test_claude_cli_callback.py` (new)

**Step 1: Write the failing test**

Create `tests/test_claude_cli_callback.py`:

```python
"""Tests for ChatClaudeCLI callback integration."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from mahtab.llm.claude_cli import ChatClaudeCLI


@pytest.mark.asyncio
async def test_agenerate_calls_run_manager_on_token():
    """_agenerate should call run_manager.on_llm_new_token for each token."""
    llm = ChatClaudeCLI()

    # Mock run_manager
    run_manager = AsyncMock()

    # Mock _call_claude_async to return a simple response
    llm._call_claude_async = AsyncMock(return_value=("Hello world", {"input_tokens": 10}))

    from langchain_core.messages import HumanMessage
    messages = [HumanMessage(content="test")]

    await llm._agenerate(messages, run_manager=run_manager)

    # run_manager.on_llm_new_token should have been called
    # Since we mock _call_claude_async, we need to test the actual loop
    # This test verifies the integration point exists
    assert True  # Placeholder - real test below
```

Actually, testing this properly requires testing the token loop. Let me write a better test:

```python
"""Tests for ChatClaudeCLI callback integration."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import asyncio

from mahtab.llm.claude_cli import ChatClaudeCLI


@pytest.mark.asyncio
async def test_agenerate_with_run_manager():
    """_agenerate should accept and use run_manager."""
    llm = ChatClaudeCLI()

    run_manager = AsyncMock()
    run_manager.on_llm_new_token = AsyncMock()

    # We need to mock at a lower level - mock the subprocess
    mock_stdout = AsyncMock()
    mock_stdout.__aiter__ = lambda self: self
    mock_stdout.__anext__ = AsyncMock(side_effect=StopAsyncIteration)

    mock_proc = AsyncMock()
    mock_proc.stdout = mock_stdout
    mock_proc.stderr = AsyncMock()
    mock_proc.stderr.read = AsyncMock(return_value=b"")
    mock_proc.wait = AsyncMock()
    mock_proc.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        from langchain_core.messages import HumanMessage
        messages = [HumanMessage(content="test")]

        # Should not raise - run_manager is accepted
        try:
            await llm._agenerate(messages, run_manager=run_manager)
        except Exception:
            pass  # May fail due to empty response, but that's ok for this test
```

**Step 2: Run test to verify baseline**

Run: `uv run pytest tests/test_claude_cli_callback.py -v`
Expected: PASS (run_manager already accepted, just unused)

**Step 3: Wire run_manager in the token loop**

In `mahtab/llm/claude_cli.py`, modify `_agenerate` to pass run_manager to the helper, then modify `_call_claude_async` to accept and use it.

Change `_agenerate` (around line 73):

```python
        result, usage = await self._call_claude_async(prompt, system, run_manager)
```

Change `_call_claude_async` signature (line 181):

```python
    async def _call_claude_async(self, prompt: str, system: str, run_manager=None) -> tuple[str, dict | None]:
```

Add callback invocation inside the token loop (after line 222, inside the `if delta.get("type") == "text_delta":` block):

```python
                            full_response += delta.get("text", "")
                            if run_manager:
                                await run_manager.on_llm_new_token(delta.get("text", ""))
```

**Step 4: Write test that verifies tokens are emitted**

Update `tests/test_claude_cli_callback.py`:

```python
"""Tests for ChatClaudeCLI callback integration."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch

from mahtab.llm.claude_cli import ChatClaudeCLI


@pytest.mark.asyncio
async def test_agenerate_emits_tokens_to_run_manager():
    """_agenerate should call run_manager.on_llm_new_token for each token."""
    llm = ChatClaudeCLI()

    run_manager = AsyncMock()
    tokens_received = []

    async def capture_token(token, **kwargs):
        tokens_received.append(token)

    run_manager.on_llm_new_token = capture_token

    # Simulate streaming JSON output from CLI
    stream_lines = [
        json.dumps({"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}}),
        json.dumps({"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": " world"}}}),
        json.dumps({"type": "result", "result": "Hello world", "usage": {"input_tokens": 10, "output_tokens": 2}}),
    ]

    async def mock_readline():
        for line in stream_lines:
            yield (line + "\n").encode()

    mock_proc = AsyncMock()
    mock_proc.stdout = mock_readline()
    mock_proc.stderr = AsyncMock()
    mock_proc.stderr.read = AsyncMock(return_value=b"")
    mock_proc.wait = AsyncMock()
    mock_proc.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        from langchain_core.messages import HumanMessage
        messages = [HumanMessage(content="test")]

        await llm._agenerate(messages, run_manager=run_manager)

    assert tokens_received == ["Hello", " world"]
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_claude_cli_callback.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add mahtab/llm/claude_cli.py tests/test_claude_cli_callback.py
git commit -m "feat: wire run_manager to emit tokens in _agenerate"
```

---

### Task 3: Pass callbacks through generate_node

**Files:**
- Modify: `mahtab/agent/graph.py:150-176, 259-311`
- Test: `tests/test_graph_nodes.py` (add test)

**Step 1: Write the failing test**

Add to `tests/test_graph_nodes.py`:

```python
@pytest.mark.asyncio
async def test_generate_node_passes_callbacks():
    """generate_node should pass callbacks config to llm.ainvoke."""
    from langchain_core.messages import AIMessage, HumanMessage
    from mahtab.agent.graph import generate_node

    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = AIMessage(content="response")

    mock_callback = MagicMock()

    state = {
        "messages": [HumanMessage(content="test")],
        "system_prompt": "You are helpful.",
        "turn_count": 0,
    }

    await generate_node(state, mock_llm, callbacks=[mock_callback])

    # Verify ainvoke was called with callbacks in config
    call_kwargs = mock_llm.ainvoke.call_args
    assert "config" in call_kwargs.kwargs
    assert call_kwargs.kwargs["config"]["callbacks"] == [mock_callback]
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_graph_nodes.py::test_generate_node_passes_callbacks -v`
Expected: FAIL - generate_node doesn't accept callbacks

**Step 3: Update generate_node to accept and pass callbacks**

In `mahtab/agent/graph.py`, modify `generate_node` (line 150):

```python
async def generate_node(state: AgentState, llm, callbacks=None) -> dict:
```

Modify the ainvoke call (line 170):

```python
    response = await llm.ainvoke(messages, config={"callbacks": callbacks} if callbacks else None)
```

**Step 4: Update build_agent_graph to pass callbacks**

The graph builder needs to pass callbacks from config to the node. Modify the `_generate` wrapper (line 276):

```python
    async def _generate(state, config=None):
        callbacks = config.get("callbacks") if config else None
        return await generate_node(state, llm, callbacks=callbacks)
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_graph_nodes.py::test_generate_node_passes_callbacks -v`
Expected: PASS

**Step 6: Commit**

```bash
git add mahtab/agent/graph.py tests/test_graph_nodes.py
git commit -m "feat: pass callbacks through generate_node"
```

---

### Task 4: REPLAgent.ask() accepts streaming handler

**Files:**
- Modify: `mahtab/agent/repl_agent.py:49-96`
- Test: `tests/test_repl_agent.py` (add test)

**Step 1: Write the failing test**

Add to `tests/test_repl_agent.py`:

```python
@pytest.mark.asyncio
async def test_ask_accepts_streaming_handler(mock_graph, mock_session):
    """ask() should accept and pass streaming_handler as callback."""
    from mahtab.agent.repl_agent import REPLAgent
    from mahtab.ui.streaming import StreamingHandler

    agent = REPLAgent(session=mock_session)
    agent._graph = mock_graph

    handler = StreamingHandler()

    await agent.ask("test prompt", streaming_handler=handler)

    # Verify graph.ainvoke was called with callbacks in config
    call_kwargs = mock_graph.ainvoke.call_args
    assert "config" in call_kwargs.kwargs
    assert handler in call_kwargs.kwargs["config"]["callbacks"]
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_repl_agent.py::test_ask_accepts_streaming_handler -v`
Expected: FAIL - ask() doesn't accept streaming_handler

**Step 3: Update ask() signature and implementation**

In `mahtab/agent/repl_agent.py`, modify `ask()` (line 49):

```python
    async def ask(
        self,
        prompt: str,
        streaming_handler=None,
    ) -> str:
```

Remove the unused callback parameters (on_token, on_code_block, on_execution).

Add callbacks to ainvoke (line 89):

```python
        callbacks = [streaming_handler] if streaming_handler else None
        result = await self._graph.ainvoke(initial_state, config={"callbacks": callbacks})
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_repl_agent.py::test_ask_accepts_streaming_handler -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mahtab/agent/repl_agent.py tests/test_repl_agent.py
git commit -m "feat: REPLAgent.ask() accepts streaming_handler"
```

---

### Task 5: Simplify interactive.py ask() to use agent.ask()

**Files:**
- Modify: `mahtab/repl/interactive.py:116-198`

**Step 1: Read current implementation for context**

The current `ask()` function in interactive.py has its own agentic loop. We're replacing it with a call to `agent.ask()`.

**Step 2: Replace the ask() function**

Replace lines 116-198 with:

```python
    def ask(prompt: str = "") -> None:
        """Ask Claude something. Claude can execute code in your namespace."""
        if not prompt:
            return

        try:
            streaming_handler.reset()
            streaming_handler.start_spinner()

            async def run():
                return await agent.ask(prompt, streaming_handler=streaming_handler)

            result = asyncio.run(run())

            # Print execution outputs (handled by execute_node via callback - TODO)

        except KeyboardInterrupt:
            streaming_handler.cleanup()
            sys.stdout.write("\n\033[33m[cancelled]\033[0m\n")
            sys.stdout.flush()
        finally:
            session.clear_activity()
            print("\033[0m", end="", flush=True)
```

**Step 3: Run manual test**

Run: `uv run python -c "from mahtab.repl.interactive import run_repl; run_repl()"`
Test: Type `ask("say hello")` and verify streaming works.

**Step 4: Commit**

```bash
git add mahtab/repl/interactive.py
git commit -m "refactor: simplify interactive.py to use agent.ask()"
```

---

### Task 6: Add execution output callback

**Files:**
- Modify: `mahtab/agent/graph.py` (execute_node)
- Modify: `mahtab/agent/repl_agent.py` (add on_execution param)
- Modify: `mahtab/repl/interactive.py` (pass callback)

**Step 1: Add on_execution callback to ask()**

In `mahtab/agent/repl_agent.py`:

```python
    async def ask(
        self,
        prompt: str,
        streaming_handler=None,
        on_execution=None,
    ) -> str:
```

Store in state for execute_node to access:

```python
        initial_state: AgentState = {
            ...
            "on_execution": on_execution,
        }
```

**Step 2: Update AgentState TypedDict**

In `mahtab/agent/graph.py`, add to AgentState:

```python
    on_execution: callable | None
```

**Step 3: Call on_execution in execute_node**

In `execute_node`, after each execution result:

```python
        on_execution = state.get("on_execution")
        if on_execution:
            on_execution(output, is_error)
```

**Step 4: Pass callback from interactive.py**

```python
    def ask(prompt: str = "") -> None:
        ...
        def handle_execution(output, is_error):
            print_output_panel(output, is_error)

        result = asyncio.run(agent.ask(
            prompt,
            streaming_handler=streaming_handler,
            on_execution=handle_execution,
        ))
```

**Step 5: Test manually and commit**

```bash
git add mahtab/agent/graph.py mahtab/agent/repl_agent.py mahtab/repl/interactive.py
git commit -m "feat: add on_execution callback for code output"
```

---

### Task 7: Capture usage stats in on_llm_end

**Files:**
- Modify: `mahtab/ui/streaming.py`
- Modify: `mahtab/repl/interactive.py`

**Step 1: Add usage capture to StreamingHandler**

Add to StreamingHandler `__init__`:

```python
        self.last_usage = None
```

Update `on_llm_end`:

```python
    def on_llm_end(self, response, **kwargs) -> None:
        """Called by LangChain when LLM finishes generating."""
        self.flush()
        self.stop_spinner()

        # Capture usage from response if available
        if hasattr(response, "llm_output") and response.llm_output:
            self.last_usage = response.llm_output.get("usage")
```

**Step 2: Record usage in interactive.py**

After `agent.ask()` returns:

```python
            if streaming_handler.last_usage:
                session.usage.record(
                    cost=streaming_handler.last_usage.get("total_cost_usd", 0),
                    input_tokens=streaming_handler.last_usage.get("input_tokens", 0),
                    output_tokens=streaming_handler.last_usage.get("output_tokens", 0),
                )
```

**Step 3: Commit**

```bash
git add mahtab/ui/streaming.py mahtab/repl/interactive.py
git commit -m "feat: capture usage stats via on_llm_end callback"
```

---

### Task 8: Clean up and final testing

**Step 1: Run all tests**

Run: `uv run pytest -v`
Expected: All pass

**Step 2: Run pre-commit**

Run: `uv run pre-commit run --all-files`
Expected: All pass

**Step 3: Manual integration test**

Run: `uv run python -c "from mahtab.repl.interactive import run_repl; run_repl()"`

Test scenarios:
- `ask("say hello")` - should stream text
- `ask("print('hello')")` - should show code panel, execute, show output
- `usage()` - should show accumulated stats
- Ctrl+C during response - should cancel cleanly

**Step 4: Final commit**

```bash
git add -A
git commit -m "chore: streaming consolidation complete"
```
