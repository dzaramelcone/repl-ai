"""Store: Giant byte blob shared across sessions."""

from __future__ import annotations


class Store:
    """Giant byte blob. All sessions share one instance."""

    def __init__(self):
        self.data: bytes = b""

    def load(self, start: int = 0, end: int | None = None) -> bytes:
        """Read a slice of the blob."""
        return self.data[start:end]

    def append(self, content: bytes | str) -> None:
        """Add to the blob."""
        if isinstance(content, str):
            content = content.encode()
        self.data += content
