"""Streaming output utilities: typewriter animation and live panels."""

import re
import sys
import time
from enum import Enum, auto

from langchain_core.callbacks import BaseCallbackHandler
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner

from mahtab.llm import extract_usage
from mahtab.ui.buffer_parser import find_close_tag, find_partial_close
from mahtab.ui.code_panel import CodePanel
from mahtab.ui.console import format_elapsed
from mahtab.ui.markdown_panel import MarkdownPanel
from mahtab.ui.xml_panel import XmlPanel


class StreamState(Enum):
    """State machine states for XML tag parsing."""

    OUTSIDE = auto()
    IN_CHAT = auto()
    IN_REPL = auto()
    IN_XML = auto()


class StreamingHandler(BaseCallbackHandler):
    """Handles streaming output with XML tag parsing."""

    _real_stdout = sys.stdout
    _OPEN_CHAT = "<assistant-chat>"
    _CLOSE_CHAT = "</assistant-chat>"
    _OPEN_REPL = "<assistant-repl-in>"
    _CLOSE_REPL = "</assistant-repl-in>"

    def __init__(self, console: Console, chars_per_second: float):
        super().__init__()
        self.console = console
        self._char_interval = 1.0 / chars_per_second
        self._state = StreamState.OUTSIDE
        self._buffer = ""
        self._spinner: Live | None = None
        self._spinner_start_time: float = 0.0
        self._code_panel = CodePanel(console)
        self._xml_panel = XmlPanel(console)
        self._chat_panel = MarkdownPanel(console)
        self._first_token = True
        self._last_output_time: float = 0.0
        self.last_usage: dict = {}
        self._handlers = {
            StreamState.OUTSIDE: self._handle_outside,
            StreamState.IN_CHAT: self._handle_chat,
            StreamState.IN_REPL: self._handle_repl,
            StreamState.IN_XML: self._handle_xml,
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
            self._spinner_start_time = time.time()
        self._first_token = True

    def stop_spinner(self) -> None:
        """Stop the spinner and show elapsed time."""
        if self._spinner:
            elapsed = time.time() - self._spinner_start_time
            self._spinner.stop()
            self._spinner = None
            self.console.print(f"[dim]thought for {format_elapsed(elapsed)}[/]")

    def _try_known_tags(self) -> bool | None:
        """Try to match known tags. Returns True/False if matched, None if not."""
        if self._buffer.startswith(self._OPEN_CHAT):
            self._buffer = self._buffer[len(self._OPEN_CHAT) :]
            self._state = StreamState.IN_CHAT
            self._chat_panel.start()
            return True
        if self._buffer.startswith(self._OPEN_REPL):
            self._buffer = self._buffer[len(self._OPEN_REPL) :]
            self._state = StreamState.IN_REPL
            self._code_panel.start(title="REPL", color="magenta")
            return True
        for tag in (self._OPEN_CHAT, self._OPEN_REPL):
            if tag.startswith(self._buffer):
                return False
        return None

    def _try_generic_xml(self) -> bool | None:
        """Try to match generic XML tag. Returns True/False if matched, None if not."""
        xml_match = re.match(r"<([a-z_-]+)>", self._buffer)
        if xml_match:
            self._buffer = self._buffer[xml_match.end() :]
            self._state = StreamState.IN_XML
            self._xml_panel.start(xml_match.group(1))
            return True
        if re.match(r"<[a-z_-]*$", self._buffer):
            return False
        return None

    def _handle_outside(self) -> bool:
        """Handle OUTSIDE state."""
        # Skip markdown code fences that wrap XML (e.g. ```xml\n<reflection>...)
        fence_match = re.match(r"```\w*\n+", self._buffer)
        if fence_match:
            self._buffer = self._buffer[fence_match.end() :]
            return True
        # Wait if we might be starting a code fence
        if re.match(r"```\w*$", self._buffer):
            return False
        result = self._try_known_tags()
        if result is not None:
            return result
        result = self._try_generic_xml()
        if result is not None:
            return result
        # Output unrecognized content up to next '<' or '`'
        next_special = len(self._buffer)
        for char in "<`":
            pos = self._buffer.find(char, 1)
            if pos > 0:
                next_special = min(next_special, pos)
        if next_special < len(self._buffer):
            self._write_smooth(self._buffer[:next_special])
            self._buffer = self._buffer[next_special:]
            return True
        if self._buffer.endswith("<") or self._buffer.endswith("`"):
            self._write_smooth(self._buffer[:-1])
            self._buffer = self._buffer[-1:]
        else:
            self._write_smooth(self._buffer)
            self._buffer = ""
        return False

    def _handle_chat(self) -> bool:
        """Handle IN_CHAT state."""
        content, remaining = find_close_tag(self._buffer, self._CLOSE_CHAT)
        if content is not None:
            self._chat_panel.append(content)
            self._chat_panel.finish()
            self._buffer = remaining
            self._state = StreamState.OUTSIDE
            return True
        before, partial = find_partial_close(self._buffer)
        if partial.startswith("</"):
            if before:
                self._chat_panel.append(before)
            self._buffer = partial
            self._chat_panel.update()
            return False
        self._chat_panel.append(self._buffer)
        self._buffer = ""
        self._chat_panel.update()
        return False

    def _handle_repl(self) -> bool:
        """Handle IN_REPL state."""
        content, remaining = find_close_tag(self._buffer, self._CLOSE_REPL)
        if content is not None:
            self._code_panel.append(content)
            self._code_panel.finish()
            self._buffer = remaining
            self._state = StreamState.OUTSIDE
            return True
        before, partial = find_partial_close(self._buffer)
        if before:
            self._code_panel.append(before)
            self._buffer = partial
        else:
            self._code_panel.append(self._buffer)
            self._buffer = ""
        self._code_panel.update()
        return False

    def _handle_xml(self) -> bool:
        """Handle IN_XML state for generic XML blocks."""
        close_tag = f"</{self._xml_panel.tag}>"
        content, remaining = find_close_tag(self._buffer, close_tag)
        if content is not None:
            self._xml_panel.append(content)
            self._xml_panel.finish()
            self._buffer = remaining
            self._state = StreamState.OUTSIDE
            return True
        before, partial = find_partial_close(self._buffer)
        if before:
            self._xml_panel.append(before)
            self._buffer = partial
        else:
            self._xml_panel.append(self._buffer)
            self._buffer = ""
        self._xml_panel.update()
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
        if self._state == StreamState.IN_CHAT and self._chat_panel.is_active:
            self._chat_panel.append(self._buffer)
            self._chat_panel.finish()
        elif self._state == StreamState.IN_REPL and self._code_panel.is_active:
            self._code_panel.append(self._buffer)
            self._code_panel.finish()
        elif self._state == StreamState.IN_XML and self._xml_panel.is_active:
            self._xml_panel.append(self._buffer)
            self._xml_panel.finish()
        self._buffer = ""

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
        self._chat_panel.cleanup()
        self._code_panel.cleanup()
        self._xml_panel.cleanup()

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
