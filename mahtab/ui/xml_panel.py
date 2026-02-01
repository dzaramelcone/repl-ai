"""Live XML panel for streaming display."""

from __future__ import annotations

from rich.console import Console
from rich.live import Live
from rich.panel import Panel


class XmlPanel:
    """Manages a live-updating panel for streaming XML content."""

    def __init__(self, console: Console):
        self.console = console
        self._live: Live | None = None
        self._tag = ""
        self._buffer = ""

    def _make_panel(self, done: bool) -> Panel:
        """Create a panel for display."""
        title = f"[bold magenta]{self._tag}[/]" if done else f"[dim magenta]{self._tag}[/]"
        border = "magenta" if done else "dim"
        return Panel(self._buffer.strip() or " ", title=title, border_style=border)

    def start(self, tag: str) -> None:
        """Start the live XML panel."""
        self._tag = tag
        self._buffer = ""
        self._live = Live(
            self._make_panel(done=False),
            console=self.console,
            refresh_per_second=15,
        )
        self._live.start()

    def append(self, text: str) -> None:
        """Append text to the buffer."""
        self._buffer += text

    def update(self) -> None:
        """Update the live panel."""
        if self._live:
            self._live.update(self._make_panel(done=False))

    def finish(self) -> None:
        """Finalize the panel."""
        if self._live:
            self._live.update(self._make_panel(done=True))
            self._live.stop()
            self._live = None
        self._buffer = ""
        self._tag = ""

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
    def tag(self) -> str:
        """Get current tag name."""
        return self._tag

    @property
    def buffer(self) -> str:
        """Get current buffer contents."""
        return self._buffer
