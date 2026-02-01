"""In-memory message store."""

from __future__ import annotations


class MemoryStore:
    """Simple in-memory byte store."""

    def __init__(self) -> None:
        self.data = bytearray()

    def append(self, data: bytes) -> None:
        self.data.extend(data)

    def clear(self) -> None:
        self.data.clear()
