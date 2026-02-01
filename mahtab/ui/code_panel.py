"""Live code panel for streaming display."""

from __future__ import annotations

import time

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax


class CodePanel:
    """Manages a live-updating code panel for streaming code display."""

    def __init__(self, console: Console, update_interval: float = 0.0333):
        self.console = console
        self._update_interval = update_interval
        self._live: Live | None = None
        self._buffer = ""
        self._language = "python"
        self._last_update_time: float = 0.0

    def _make_panel(self, code: str, done: bool) -> Panel:
        """Create a code panel for display."""
        syntax = Syntax(
            code or " ",
            self._language,
            theme="monokai",
            line_numbers=True,
            indent_guides=True,
        )
        title = self._language if self._language != "python" else "Code"
        if done:
            return Panel(
                syntax,
                title=f"[bold cyan]{title}[/]",
                border_style="cyan",
            )
        return Panel(
            syntax,
            title=f"[dim cyan]{title}...[/]",
            border_style="dim",
        )

    def start(self, language: str = "python") -> None:
        """Start the live code panel."""
        self._buffer = ""
        self._language = language
        self._live = Live(
            self._make_panel("", done=False),
            console=self.console,
            refresh_per_second=15,
        )
        self._live.start()

    def append(self, text: str) -> None:
        """Append text to the code buffer."""
        self._buffer += text

    def update(self) -> None:
        """Rate-limited update of live code panel."""
        if self._live:
            now = time.time()
            if now - self._last_update_time >= self._update_interval:
                self._live.update(self._make_panel(self._buffer, done=False))
                self._last_update_time = now

    def finish(self) -> None:
        """Finalize the code panel."""
        if self._live:
            self._live.update(self._make_panel(self._buffer.strip(), done=True))
            self._live.stop()
            self._live = None
        self._buffer = ""

    def cleanup(self) -> None:
        """Clean up if panel is still open."""
        if self._live:
            self._live.stop()
            self._live = None

    @property
    def is_active(self) -> bool:
        """Check if panel is currently active."""
        return self._live is not None

    @property
    def buffer(self) -> str:
        """Get current buffer contents."""
        return self._buffer
