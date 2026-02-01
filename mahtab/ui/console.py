"""Rich console utilities and singleton."""

from rich.console import Console

# Global console instance
console = Console()


def get_console() -> Console:
    """Get the global console instance."""
    return console


def format_elapsed(seconds: float) -> str:
    """Format elapsed time as human-readable string."""
    if seconds >= 1.0:
        return f"{seconds:.1f}s"
    return f"{seconds * 1000:.0f}ms"
