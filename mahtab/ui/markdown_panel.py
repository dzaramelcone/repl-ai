"""Live markdown panel for streaming display."""

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel


class MarkdownPanel:
    """Manages a live-updating panel for streaming markdown content."""

    def __init__(self, console: Console):
        self.console = console
        self._live: Live | None = None
        self._buffer = ""

    def _make_panel(self, done: bool) -> Panel:
        """Create a panel for display."""
        # During streaming, show raw text; when done, render as Markdown
        content: Markdown | str
        if done and self._buffer.strip():
            content = Markdown(self._buffer.strip())
        else:
            content = self._buffer.strip() or " "
        title = "[bold blue]Claude[/]" if done else "[dim blue]Claude...[/]"
        border = "blue" if done else "dim"
        return Panel(content, title=title, border_style=border)

    def start(self) -> None:
        """Start the live markdown panel."""
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
        """Finalize the panel with markdown rendering."""
        if self._live:
            self._live.update(self._make_panel(done=True))
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
