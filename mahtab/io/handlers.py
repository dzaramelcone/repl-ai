"""Logging handlers for message routing."""

import logging
from typing import Protocol

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from mahtab.io.formatters import BytesFormatter, RichFormatter, XMLFormatter
from mahtab.ui.streaming import StreamingHandler


class Store(Protocol):
    """Protocol for message stores."""

    def append(self, _data: bytes) -> None: ...


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


class StoreHandler(logging.Handler):
    """Appends bytes to a store."""

    def __init__(self, store: Store) -> None:
        super().__init__()
        self.store = store
        self.setFormatter(BytesFormatter())

    def emit(self, record: logging.LogRecord) -> None:
        self.store.append(self.format(record))


class DisplayHandler(logging.Handler):
    """Routes messages to terminal display."""

    def __init__(self, console: Console) -> None:
        super().__init__()
        self.console = console
        self.setFormatter(RichFormatter())
        self.streamer = StreamingHandler(console=console, chars_per_second=200.0)

    def emit(self, record: logging.LogRecord) -> None:
        match record.tag:
            case "assistant-chat-stream":
                self.streamer.process_token(record.getMessage())
            case "assistant-chat" | "assistant-repl-in" | "assistant-repl-out":
                # Skip - already shown during streaming / on_execution callback
                pass
            case "user-chat":
                md = Markdown(record.getMessage())
                panel = Panel(md, title="[bold green]You[/]", border_style="green")
                self.console.print(panel)
            case _:
                self.console.print(self.format(record))
