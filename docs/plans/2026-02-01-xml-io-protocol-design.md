# XML I/O Protocol Design

Structured message protocol for Mahtab using XML tags, with logging-based routing to handlers.

## Purpose

Enable integration with:
- CLI (Claude Code) - structured input/output
- LangGraph agents - message routing backbone

## Core Concepts

### One Logger, Six Tag Types

Single logger with message type in record metadata:

```python
log = logging.getLogger("mahtab")
log.info(content, extra={"tag": "user-repl-in"})
```

### Tags

| Tag | Meaning |
|-----|---------|
| `user-repl-in` | User inputs code |
| `user-repl-out` | Output from user's code |
| `assistant-repl-in` | Assistant inputs code |
| `assistant-repl-out` | Output from assistant's code |
| `user-chat` | User natural language |
| `assistant-chat` | Assistant natural language (complete) |
| `assistant-chat-stream` | Assistant tokens (display only) |

### Three Handler Types

| Handler | Formatter | Purpose |
|---------|-----------|---------|
| `PromptHandler` | `XMLFormatter` | Builds Claude's context with XML tags |
| `DisplayHandler` | `RichFormatter` | Terminal output with Rich styling |
| `StoreHandler` | `BytesFormatter` | Serializes to RAM store |

## Formatters

### XMLFormatter

```python
class XMLFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return f"<{record.tag}>{record.getMessage()}</{record.tag}>"
```

### RichFormatter

```python
class RichFormatter(logging.Formatter):
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
```

### BytesFormatter

```python
class BytesFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> bytes:
        return f"<{record.tag}>{record.getMessage()}</{record.tag}>".encode("utf-8")
```

## Handlers

### PromptHandler

Accumulates XML-formatted messages for Claude's context.

```python
class PromptHandler(logging.Handler):
    def __init__(self):
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

### DisplayHandler

Routes to terminal. Delegates streaming to existing StreamingHandler.

```python
class DisplayHandler(logging.Handler):
    def __init__(self, console: Console):
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

### StoreHandler

Appends bytes to RAM store.

```python
class StoreHandler(logging.Handler):
    def __init__(self, store: Store):
        super().__init__()
        self.store = store
        self.setFormatter(BytesFormatter())

    def emit(self, record: logging.LogRecord) -> None:
        self.store.append(self.format(record))
```

## Filters

```python
class TagFilter(logging.Filter):
    def __init__(self, tags: set[str]):
        super().__init__()
        self.tags = tags

    def filter(self, record: logging.LogRecord) -> bool:
        return record.tag in self.tags
```

## Routing Table

| Tag | Display | Prompt | Store |
|-----|---------|--------|-------|
| `user-repl-in` | yes | yes | yes |
| `user-repl-out` | yes | yes | yes |
| `assistant-repl-in` | yes | yes | yes |
| `assistant-repl-out` | yes | yes | yes |
| `user-chat` | yes | yes | yes |
| `assistant-chat` | no (already streamed) | yes | yes |
| `assistant-chat-stream` | yes | no | no |

## Inbound Parsing

Claude sends XML-tagged responses. Parser extracts and routes:

```python
import re

def parse_response(response: str) -> list[tuple[str, str]]:
    """Returns list of (tag, content) tuples."""
    pattern = r"<(assistant-(?:chat|repl-in))>(.*?)</\1>"
    return re.findall(pattern, response, re.DOTALL)

def route_response(response: str) -> None:
    for tag, content in parse_response(response):
        log.info(content, extra={"tag": tag})
```

Streaming displays raw tokens (including XML tags) initially. State machine for pretty streaming is future work.

## File Structure

```
mahtab/
├── io/
│   ├── __init__.py
│   ├── tags.py            # Tag constants
│   ├── formatters.py      # XMLFormatter, RichFormatter, BytesFormatter
│   ├── handlers.py        # PromptHandler, DisplayHandler, StoreHandler
│   ├── filters.py         # TagFilter
│   ├── parser.py          # parse_response(), route_response()
│   └── setup.py           # Logger wiring
├── ui/
│   ├── streaming.py       # StreamingHandler (stays here)
│   └── ...
```

## Setup

```python
# mahtab/io/setup.py
import logging
from mahtab.io.handlers import PromptHandler, DisplayHandler, StoreHandler
from mahtab.io.filters import TagFilter
from mahtab.ui.console import console

def setup_logging(store: Store) -> tuple[logging.Logger, PromptHandler]:
    log = logging.getLogger("mahtab")
    log.setLevel(logging.INFO)

    prompt = PromptHandler()
    prompt.addFilter(TagFilter({
        "user-repl-in", "user-repl-out",
        "assistant-repl-in", "assistant-repl-out",
        "user-chat", "assistant-chat"
    }))

    display = DisplayHandler(console)
    display.addFilter(TagFilter({
        "user-repl-in", "user-repl-out",
        "assistant-repl-in", "assistant-repl-out",
        "user-chat", "assistant-chat-stream"
    }))

    store_handler = StoreHandler(store)
    store_handler.addFilter(TagFilter({
        "user-repl-in", "user-repl-out",
        "assistant-repl-in", "assistant-repl-out",
        "user-chat", "assistant-chat"
    }))

    log.addHandler(prompt)
    log.addHandler(display)
    log.addHandler(store_handler)

    return log, prompt
```

## Integration Points

| Current code | New approach |
|--------------|--------------|
| `repl/interactive.py` user input | `log.info(input, extra={"tag": "user-chat"})` |
| `graph.py` code extraction | `log.info(code, extra={"tag": "assistant-repl-in"})` |
| `graph.py` execution output | `log.info(output, extra={"tag": "assistant-repl-out"})` |
| `executor.py` user code | `log.info(code, extra={"tag": "user-repl-in"})` |
| `executor.py` user output | `log.info(output, extra={"tag": "user-repl-out"})` |
| `streaming.py` tokens | `log.info(token, extra={"tag": "assistant-chat-stream"})` |
| `repl_agent.py` context building | `prompt_handler.get_context()` |

## Future Work

- State machine for pretty streaming (strip XML tags in real-time)
- Custom persistence for Store
