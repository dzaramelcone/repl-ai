"""Streaming output utilities: typewriter animation and live panels."""

from __future__ import annotations

import sys
import time
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
    - Smooth rate-limited text output for consistent streaming feel
    - Live-updating code panels as code streams in

    Attributes:
        console: Rich console for output.
        chars_per_second: Target output rate for smooth streaming.
    """

    def __init__(self, console: Console | None = None, chars_per_second: float = 200.0):
        self.console = console or default_console
        self.chars_per_second = chars_per_second

        # Internal state
        self._spinner: Live | None = None
        self._code_live: Live | None = None
        self._in_code_block = False
        self._text_buffer = ""
        self._code_buffer = ""
        self._first_token = True

        # Smooth streaming state
        self._last_output_time: float = 0.0
        self._last_code_update_time: float = 0.0
        self._code_update_interval: float = 1.0 / 30.0  # 30 updates per second max

    def _write(self, text: str) -> None:
        """Write text to stdout."""
        sys.stdout.write(text)
        sys.stdout.flush()

    def _write_smooth(self, text: str) -> None:
        """Write text with rate-limited smooth streaming.

        Outputs characters at a consistent rate to avoid choppy bursts.
        """
        if not text:
            return

        now = time.time()
        char_interval = 1.0 / self.chars_per_second

        for char in text:
            # Calculate time since last output
            elapsed = now - self._last_output_time

            # If we're behind schedule, catch up gradually (don't sleep)
            # If we're ahead, add a small delay
            if elapsed < char_interval and self._last_output_time > 0:
                time.sleep(char_interval - elapsed)

            sys.stdout.write(char)
            sys.stdout.flush()
            self._last_output_time = time.time()
            now = self._last_output_time

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
                # Check for closing ```
                if self._code_buffer.endswith("```"):
                    final_code = self._code_buffer[:-3]
                    if self._code_live:
                        self._code_live.update(self._make_code_panel(final_code, done=True))
                        self._code_live.stop()
                        self._code_live = None
                    self._in_code_block = False
                    self._code_buffer = ""
                # Rate-limited update of live code panel
                elif self._code_live:
                    now = time.time()
                    if now - self._last_code_update_time >= self._code_update_interval:
                        self._code_live.update(self._make_code_panel(self._code_buffer.rstrip("`")))
                        self._last_code_update_time = now
            else:
                self._text_buffer += char
                if "```python\n" in self._text_buffer or "```python\r\n" in self._text_buffer:
                    # Flush text before code block
                    idx = self._text_buffer.find("```python")
                    if idx > 0:
                        self._write_smooth(self._text_buffer[:idx])
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
                    # Flush buffered text with smooth streaming
                    self._write_smooth(self._text_buffer)
                    self._text_buffer = ""

    def flush(self) -> None:
        """Flush any remaining buffered text."""
        if self._text_buffer and not self._in_code_block:
            self._write_smooth(self._text_buffer)
            self._text_buffer = ""
        self._write("\n")

    def reset(self) -> None:
        """Reset state for a new streaming session."""
        self._text_buffer = ""
        self._code_buffer = ""
        self._in_code_block = False
        self._first_token = True
        self._last_output_time = 0.0
        self._last_code_update_time = 0.0

    def cleanup(self) -> None:
        """Clean up any active UI elements."""
        self.stop_spinner()
        if self._code_live:
            self._code_live.stop()
            self._code_live = None
