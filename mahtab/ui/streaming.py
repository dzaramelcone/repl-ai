"""Streaming output utilities: typewriter animation and live panels."""

from __future__ import annotations

import sys
import time
from enum import Enum, auto

from langchain_core.callbacks import BaseCallbackHandler
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner

from mahtab.llm import extract_usage
from mahtab.ui.code_panel import CodePanel


class StreamState(Enum):
    """State machine states for XML tag parsing."""

    OUTSIDE = auto()
    IN_CHAT = auto()
    IN_REPL = auto()


class StreamingHandler(BaseCallbackHandler):
    """Handles streaming output with XML tag parsing."""

    # Capture real stdout at class load time, before any redirects
    _real_stdout = sys.stdout

    # Tag patterns
    _OPEN_CHAT = "<assistant-chat>"
    _CLOSE_CHAT = "</assistant-chat>"
    _OPEN_REPL = "<assistant-repl-in>"
    _CLOSE_REPL = "</assistant-repl-in>"

    def __init__(self, console: Console, chars_per_second: float):
        super().__init__()
        self.console = console
        self._char_interval = 1.0 / chars_per_second

        # State machine
        self._state = StreamState.OUTSIDE
        self._buffer = ""

        # UI state
        self._spinner: Live | None = None
        self._code_panel = CodePanel(console)
        self._first_token = True
        self._last_output_time: float = 0.0
        self.last_usage: dict = {}

        # State dispatch table
        self._handlers = {
            StreamState.OUTSIDE: self._handle_outside,
            StreamState.IN_CHAT: self._handle_chat,
            StreamState.IN_REPL: self._handle_repl,
        }

    def _write(self, text: str) -> None:
        """Write text to real stdout."""
        self._real_stdout.write(text)
        self._real_stdout.flush()

    def _write_smooth(self, text: str) -> None:
        """Write text with rate limiting."""
        for char in text:
            now = time.time()
            wait = self._char_interval - (now - self._last_output_time)
            if wait > 0 and self._last_output_time > 0:
                time.sleep(wait)
            self._real_stdout.write(char)
            self._real_stdout.flush()
            self._last_output_time = time.time()

    def start_spinner(self, text: str) -> None:
        """Start a spinner while waiting for response."""
        if self._spinner is None:
            self._spinner = Live(
                Spinner("dots", text=f"[dim]{text}[/]"),
                console=self.console,
                refresh_per_second=10,
            )
            self._spinner.start()
        self._first_token = True

    def stop_spinner(self) -> None:
        """Stop the spinner."""
        if self._spinner:
            self._spinner.stop()
            self._spinner = None

    def _handle_outside(self) -> bool:
        """Handle OUTSIDE state."""
        if self._buffer.startswith(self._OPEN_CHAT):
            self._buffer = self._buffer[len(self._OPEN_CHAT) :]
            self._state = StreamState.IN_CHAT
            return True
        if self._buffer.startswith(self._OPEN_REPL):
            self._buffer = self._buffer[len(self._OPEN_REPL) :]
            self._state = StreamState.IN_REPL
            self._write("\n")
            self._code_panel.start()
            return True
        for tag in (self._OPEN_CHAT, self._OPEN_REPL):
            if tag.startswith(self._buffer):
                return False
        self._buffer = ""
        return False

    def _handle_chat(self) -> bool:
        """Handle IN_CHAT state."""
        if self._CLOSE_CHAT in self._buffer:
            idx = self._buffer.find(self._CLOSE_CHAT)
            self._write_smooth(self._buffer[:idx])
            self._buffer = self._buffer[idx + len(self._CLOSE_CHAT) :]
            self._state = StreamState.OUTSIDE
            return True
        if "</" in self._buffer:
            idx = self._buffer.find("</")
            if idx > 0:
                self._write_smooth(self._buffer[:idx])
                self._buffer = self._buffer[idx:]
            return False
        self._write_smooth(self._buffer)
        self._buffer = ""
        return False

    def _handle_repl(self) -> bool:
        """Handle IN_REPL state."""
        if self._CLOSE_REPL in self._buffer:
            idx = self._buffer.find(self._CLOSE_REPL)
            self._code_panel.append(self._buffer[:idx])
            self._code_panel.finish()
            self._buffer = self._buffer[idx + len(self._CLOSE_REPL) :]
            self._state = StreamState.OUTSIDE
            return True
        if "</" in self._buffer:
            idx = self._buffer.find("</")
            if idx > 0:
                self._code_panel.append(self._buffer[:idx])
                self._buffer = self._buffer[idx:]
            self._code_panel.update()
            return False
        self._code_panel.append(self._buffer)
        self._buffer = ""
        self._code_panel.update()
        return False

    def process_token(self, token: str) -> None:
        """Process a streaming token."""
        if self._first_token:
            self.stop_spinner()
            self._first_token = False
        self._buffer += token
        while self._buffer:
            if not self._handlers[self._state]():
                break

    def flush(self) -> None:
        """Flush remaining buffered content."""
        if self._state == StreamState.IN_CHAT and self._buffer:
            self._write_smooth(self._buffer)
        elif self._state == StreamState.IN_REPL and self._code_panel.is_active:
            self._code_panel.append(self._buffer)
            self._code_panel.finish()
        self._buffer = ""
        self._write("\n")

    def reset(self) -> None:
        """Reset state for a new streaming session."""
        self._state = StreamState.OUTSIDE
        self._buffer = ""
        self._first_token = True
        self._last_output_time = 0.0
        self.last_usage = {}

    def cleanup(self) -> None:
        """Clean up any active UI elements."""
        self.stop_spinner()
        self._code_panel.cleanup()

    def on_llm_new_token(self, token: str, **_kwargs) -> None:
        """Called by LangChain when a new token is generated."""
        self.process_token(token)

    def on_llm_start(self, _serialized, _prompts, **_kwargs) -> None:
        """Called by LangChain when LLM starts generating."""
        self.start_spinner("thinking...")

    def on_llm_end(self, response, **_kwargs) -> None:
        """Called by LangChain when LLM finishes generating."""
        self.flush()
        self.stop_spinner()
        self.last_usage = extract_usage(response)
