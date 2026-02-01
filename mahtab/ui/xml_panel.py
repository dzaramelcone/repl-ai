"""Live XML panel for streaming display."""

import xml.etree.ElementTree as ET

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.pretty import Pretty


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "…"


def _xml_to_dict(element: ET.Element, max_len: int = 10) -> dict | str:
    """Convert XML element to dict recursively, truncating long values."""
    children = list(element)
    if not children:
        text = element.text.strip() if element.text else ""
        return _truncate(text, max_len)
    return {child.tag: _xml_to_dict(child, max_len) for child in children}


class XmlPanel:
    """Manages a live-updating panel for streaming XML content."""

    def __init__(self, console: Console):
        self.console = console
        self._live: Live | None = None
        self._tag = ""
        self._buffer = ""

    def _format_content(self, content: str, done: bool) -> Pretty | str:
        """Format content as pretty dict if valid XML, else raw text."""
        if not done:
            return content.strip() or " "
        try:
            xml_str = f"<{self._tag}>{content}</{self._tag}>"
            root = ET.fromstring(xml_str)
            data = _xml_to_dict(root)
            return Pretty(data, indent_guides=True, expand_all=True)
        except ET.ParseError:
            return content.strip() or " "

    def _make_panel(self, done: bool) -> Panel:
        """Create a panel for display."""
        title = f"[bold magenta]{self._tag}[/]" if done else f"[dim magenta]{self._tag}[/]"
        border = "magenta" if done else "dim"
        content = self._format_content(self._buffer, done)
        return Panel(content, title=title, border_style=border)

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
