# Streaming Consolidation Design

## Problem

Two separate agentic loops exist:
1. `REPLAgent.ask()` uses LangGraph but doesn't stream (buffers via `_agenerate`)
2. `interactive.py ask()` has its own loop with streaming but bypasses LangGraph

This creates maintenance burden and inconsistency.

## Solution

Consolidate to a single LangGraph-based loop with streaming via LangChain callbacks.

## Approach: Callback Injection

Wire `run_manager` in `_agenerate` to invoke `StreamingHandler.process_token()`. The REPL passes the handler as a callback when calling `agent.ask()`, and tokens stream as a side effect of normal LangGraph execution.

## Callback Flow

```
REPL creates StreamingHandler
        │
        ▼
agent.ask(prompt, streaming_handler)
        │
        ▼
LangGraph invokes generate_node
        │
        ▼
generate_node calls llm.ainvoke() with callbacks=[streaming_handler]
        │
        ▼
_agenerate receives run_manager (wraps our callback)
        │
        ▼
_agenerate loops through CLI output, calls:
    run_manager.on_llm_new_token(text)
        │
        ▼
StreamingHandler.process_token(text) is invoked
        │
        ▼
UI updates in real-time
```

## Code Changes

### 1. StreamingHandler becomes a LangChain callback

```python
# mahtab/ui/streaming.py
from langchain_core.callbacks import BaseCallbackHandler

class StreamingHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.process_token(token)

    def on_llm_start(self, *args, **kwargs) -> None:
        self.start_spinner()

    def on_llm_end(self, response, **kwargs) -> None:
        self.flush()
        self.stop_spinner()
```

### 2. Wire run_manager in _agenerate

```python
# mahtab/llm/claude_cli.py
# Inside the token loop:
if run_manager:
    await run_manager.on_llm_new_token(text)
```

### 3. Pass callbacks through generate_node

```python
# mahtab/agent/graph.py
async def generate_node(state: AgentState, llm, callbacks=None) -> dict:
    response = await llm.ainvoke(messages, config={"callbacks": callbacks})
```

### 4. REPLAgent.ask() accepts streaming handler

```python
# mahtab/agent/repl_agent.py
async def ask(self, prompt: str, streaming_handler=None) -> str:
    callbacks = [streaming_handler] if streaming_handler else None
    result = await self._graph.ainvoke(initial_state, config={"callbacks": callbacks})
```

### 5. Delete duplicate loop in interactive.py

Replace lines 116-198 with:
```python
def ask(prompt: str = "", max_turns: int = 5) -> None:
    try:
        streaming_handler.reset()
        result = asyncio.run(agent.ask(prompt, streaming_handler))
    except KeyboardInterrupt:
        streaming_handler.cleanup()
        sys.stdout.write("\n\033[33m[cancelled]\033[0m\n")
    finally:
        session.clear_activity()
```

## Edge Cases

### Spinner lifecycle
- `on_llm_start()` calls `start_spinner()`
- `on_llm_end()` calls `stop_spinner()`

### Usage stats
- Capture in `on_llm_end()` which receives full response with metadata

### Execution output
- Add `on_execution` callback parameter to `ask()`
- Called by `execute_node` after each code block

### Keyboard interrupt
- Wrap `agent.ask()` in try/except
- Call `streaming_handler.cleanup()` on interrupt

## Files Modified

1. `mahtab/ui/streaming.py` - Add BaseCallbackHandler, implement callback methods
2. `mahtab/llm/claude_cli.py` - Call run_manager.on_llm_new_token()
3. `mahtab/agent/graph.py` - Pass callbacks through generate_node
4. `mahtab/agent/repl_agent.py` - Accept streaming_handler, pass as callback
5. `mahtab/repl/interactive.py` - Delete duplicate loop, use agent.ask()
