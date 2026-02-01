"""Streaming output utilities: typewriter animation and live panels."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax

from mahtab.ui.console import console as default_console

if TYPE_CHECKING:
    from rich.console import Console


class StreamingHandler:
    """Handles streaming output with code panel detection.

    This class manages the streaming output experience including:
    - Spinner while waiting for first token
    - Direct token output (no complex animation)
    - Live-updating code panels as code streams in

    Attributes:
        console: Rich console for output.
    """

    def __init__(self, console: Console | None = None):
        self.console = console or default_console

        # Internal state
        self._spinner: Live | None = None
        self._code_live: Live | None = None
        self._in_code_block = False
        self._text_buffer = ""
        self._code_buffer = ""
        self._first_token = True

    def _write(self, text: str) -> None:
        """Write text to stdout."""
        sys.stdout.write(text)
        sys.stdout.flush()

    def _make_code_panel(self, code: str, done: bool = False) -> Panel:
        """Create a code panel for display."""
        title = "[cyan]Code[/]" if done else "[dim cyan]Writing...[/]"
        return Panel(
            Syntax(code or " ", "python", theme="monokai", line_numbers=True),
            title=title,
            border_style="cyan" if done else "dim",
        )

    def start_spinner(self, text: str = "thinking...") -> None:
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

    def process_token(self, token: str) -> None:
        """Process a streaming token.

        Handles state machine for code block detection and output.
        """
        # Stop spinner on first token
        if self._first_token:
            self.stop_spinner()
            self._first_token = False

        for char in token:
            if self._in_code_block:
                self._code_buffer += char
                # Update live code panel
                if self._code_live and not self._code_buffer.endswith("```"):
                    self._code_live.update(self._make_code_panel(self._code_buffer.rstrip("`")))
                # Check for closing ```
                if self._code_buffer.endswith("```"):
                    final_code = self._code_buffer[:-3]
                    if self._code_live:
                        self._code_live.update(self._make_code_panel(final_code, done=True))
                        self._code_live.stop()
                        self._code_live = None
                    self._in_code_block = False
                    self._code_buffer = ""
            else:
                self._text_buffer += char
                if "```python\n" in self._text_buffer or "```python\r\n" in self._text_buffer:
                    # Flush text before code block
                    idx = self._text_buffer.find("```python")
                    if idx > 0:
                        self._write(self._text_buffer[:idx])
                    self._text_buffer = ""
                    self._code_buffer = ""
                    self._in_code_block = True
                    # Start live code panel
                    self._write("\n")
                    self._code_live = Live(
                        self._make_code_panel(""),
                        console=self.console,
                        refresh_per_second=15,
                    )
                    self._code_live.start()
                elif len(self._text_buffer) > 20 and "```" not in self._text_buffer:
                    # Flush buffered text
                    self._write(self._text_buffer)
                    self._text_buffer = ""

    def flush(self) -> None:
        """Flush any remaining buffered text."""
        if self._text_buffer and not self._in_code_block:
            self._write(self._text_buffer)
            self._text_buffer = ""
        self._write("\n")

    def reset(self) -> None:
        """Reset state for a new streaming session."""
        self._text_buffer = ""
        self._code_buffer = ""
        self._in_code_block = False
        self._first_token = True

    def cleanup(self) -> None:
        """Clean up any active UI elements."""
        self.stop_spinner()
        if self._code_live:
            self._code_live.stop()
            self._code_live = None
