# Textual Refactor Design

## Overview

Refactor Mahtab from a single Rich-based REPL to a Textual TUI application with multiple concurrent sessions, like browser tabs. Each session is still a true Python REPL, but they share a common Store and can spawn child sessions (e.g., for RLM search).

## Core Components

### Store

Giant byte blob. That's it.

```python
class Store:
    def __init__(self):
        self.data: bytes = b""

    def load(self, start: int = 0, end: int | None = None) -> bytes:
        return self.data[start:end]

    def append(self, content: bytes | str) -> None:
        if isinstance(content, str):
            content = content.encode()
        self.data += content
```

All sessions share the same Store instance. Everything gets logged to it - user input, LLM output, code execution, all of it.

Search operations (peek, grep, partition) are LangChain tools that Claude calls. They operate on the store but aren't methods of it.

### Channels via Logging

Each session has multiple channels, implemented as Python loggers:

| Channel | Purpose |
|---------|---------|
| `session.<id>.user.repl` | User's Python input/output |
| `session.<id>.user.chat` | User's prompts to Claude |
| `session.<id>.llm.repl` | Claude's generated code + execution results |
| `session.<id>.llm.chat` | Claude's responses to user |

One source, multiple destinations via handlers:

```python
class WidgetHandler(logging.Handler):
    """Sends log records to a Textual widget."""
    def __init__(self, widget: RichLog):
        super().__init__()
        self.widget = widget

    def emit(self, record):
        self.widget.write(self.format(record))

class StoreHandler(logging.Handler):
    """Appends log records to the Store."""
    def __init__(self, store: Store):
        super().__init__()
        self.store = store

    def emit(self, record):
        self.store.append(self.format(record).encode())
```

Streaming tokens log as they arrive - handler pushes to widget for live display.

### Session

Async REPL session with its own namespace and conversation history.

```python
class Session:
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
        self.namespace: dict = {}
        self.messages: list = []

        # Inherit context from parent
        if context:
            self.namespace.update(context)

        if parent:
            parent.children.append(self)

        # Set up loggers
        self.log_user_repl = logging.getLogger(f"session.{self.id}.user.repl")
        self.log_user_chat = logging.getLogger(f"session.{self.id}.user.chat")
        self.log_llm_repl = logging.getLogger(f"session.{self.id}.llm.repl")
        self.log_llm_chat = logging.getLogger(f"session.{self.id}.llm.chat")

    async def ask(self, prompt: str) -> str:
        """Send prompt to Claude, execute any code, return response."""
        ...

    def spawn(self, context: dict | None = None) -> Session:
        """Create a child session with shared store."""
        return Session(store=self.store, parent=self, context=context)

    def exec(self, code: str) -> Any:
        """Execute Python code in this session's namespace."""
        return exec(code, self.namespace)
```

### REPLWidget

Textual widget for a single session. Two panes (chat + repl) with shared input.

```python
class REPLWidget(Widget):
    def __init__(self, session: Session):
        super().__init__()
        self.session = session

    def compose(self):
        with Horizontal():
            yield RichLog(id="chat")   # Chat history
            yield RichLog(id="repl")   # REPL history
        yield TextArea(id="input", language="python")

    def on_mount(self):
        # Subscribe widgets to session loggers
        chat = self.query_one("#chat", RichLog)
        repl = self.query_one("#repl", RichLog)

        self.session.log_user_chat.addHandler(WidgetHandler(chat))
        self.session.log_llm_chat.addHandler(WidgetHandler(chat))
        self.session.log_user_repl.addHandler(WidgetHandler(repl))
        self.session.log_llm_repl.addHandler(WidgetHandler(repl))

        # Store gets everything
        for logger in [
            self.session.log_user_chat,
            self.session.log_llm_chat,
            self.session.log_user_repl,
            self.session.log_llm_repl,
        ]:
            logger.addHandler(StoreHandler(self.session.store))

    async def on_key(self, event):
        if event.key == "enter" and event.ctrl:
            await self.submit()

    async def submit(self):
        input_widget = self.query_one("#input", TextArea)
        code = input_widget.text
        input_widget.clear()

        if code.startswith("ask("):
            prompt = code[4:-1].strip("\"'")
            self.session.log_user_chat.info(f"You: {prompt}")
            response = await self.session.ask(prompt)
            # Response logged via llm_chat in session.ask()
        else:
            self.session.log_user_repl.info(f">>> {code}")
            try:
                result = self.session.exec(code)
                if result is not None:
                    self.session.log_user_repl.info(repr(result))
            except Exception as e:
                self.session.log_user_repl.error(str(e))
```

### MahtabApp

Main Textual app. Owns the Store, manages sessions as tabs.

```python
class MahtabApp(App):
    CSS = """
    #sessions { height: 1fr; }
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
        pane = TabPane(label, REPLWidget(session), id=session.id)
        tabs.add_pane(pane)

        return session

    def action_new_session(self):
        self.spawn_session()

    def action_close_session(self):
        tabs = self.query_one("#sessions", TabbedContent)
        if len(tabs.children) > 1:
            tabs.remove_pane(tabs.active)
```

### RLM Integration

RLM spawns child sessions. Tools (peek, grep, partition) operate on the shared store.

```python
# Tools Claude can call
def make_tools(store: Store):
    @tool
    def peek(start: int = 0, n: int = 1000) -> str:
        """View n bytes from the store starting at offset."""
        chunk = store.load(start, start + n)
        # Return as hex + ascii, or decoded if valid utf-8
        try:
            return chunk.decode()
        except UnicodeDecodeError:
            return hexdump(chunk)

    @tool
    def grep(pattern: str) -> str:
        """Search the store for pattern. Returns matches with offsets."""
        data = store.load()
        matches = []
        for m in re.finditer(pattern.encode(), data):
            matches.append(f"{m.start()}: {m.group()}")
        return "\n".join(matches[:50])  # Limit results

    @tool
    def partition(n: int) -> str:
        """Split store into n chunks, return size info."""
        total = len(store.data)
        chunk_size = total // n
        return f"{n} chunks of ~{chunk_size} bytes each"

    return [peek, grep, partition]
```

When Claude wants to do RLM-style deep search, it spawns a child session:

```python
@tool
def spawn_search(query: str) -> str:
    """Spawn a child session to search for something in the store."""
    child = app.spawn_session(parent=current_session, context={"query": query})
    # Child session gets its own tab
    # Parent waits for result
    return await child.ask(f"Search the store for: {query}")
```

## Visual Layout

```
┌─────────────────────────────────────────────────────────────┐
│ Mahtab                                              [$0.42] │
├───────────┬───────────┬─────────────────────────────────────┤
│ Session a │ ↳ child b │                                     │
├───────────┴───────────┴─────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────┬─────────────────────────────┐  │
│  │ Chat                    │ REPL                        │  │
│  │                         │                             │  │
│  │ You: help me with X     │ >>> x = 5                   │  │
│  │                         │ >>> def foo():              │  │
│  │ Claude: I'll try...     │ ...     return x * 2        │  │
│  │                         │ >>> foo()                   │  │
│  │ [streaming...]          │ 10                          │  │
│  │                         │                             │  │
│  └─────────────────────────┴─────────────────────────────┘  │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ >>> _                                                   ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Streaming

Streaming handler logs tokens as they arrive:

```python
class StreamingHandler(BaseCallbackHandler):
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.buffer = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.buffer += token
        self.logger.info(token)  # Widget updates live

    def on_llm_end(self, response, **kwargs):
        # Full response in self.buffer
        pass
```

WidgetHandler receives log records and writes to RichLog, which handles the live update.

## Migration Path

1. Add textual dependency
2. Implement Store, Session, REPLWidget, MahtabApp
3. Migrate existing tools (read_file, edit_file, etc.) to work with sessions
4. Update RLM to spawn child sessions instead of recursive function calls
5. Keep `python -m mahtab` entry point, now launches Textual app

## Open Questions

- Mode toggle (chat vs repl) or detect from input?
- Should child tabs auto-close when done?
- How to handle keyboard focus when child spawns?
- Persist store to disk on exit?
