"""Rich console utilities and singleton."""

from __future__ import annotations

from rich.console import Console

# Global console instance
console = Console()


def get_console() -> Console:
    """Get the global console instance."""
    return console
