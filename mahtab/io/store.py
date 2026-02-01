"""In-memory message store."""


class MemoryStore:
    """Simple in-memory byte store."""

    def __init__(self) -> None:
        self.data = bytearray()

    def append(self, data: bytes) -> None:
        self.data.extend(data)

    def clear(self) -> None:
        self.data.clear()

    def find_records(self, tags: list[str]) -> list[tuple[int, str]]:
        """Find all XML records for given tags, sorted by position."""
        data = self.data.decode("utf-8", errors="replace")
        matches = []
        for tag in tags:
            open_tag, close_tag = f"<{tag}>", f"</{tag}>"
            start = 0
            while (open_pos := data.find(open_tag, start)) != -1:
                close_pos = data.find(close_tag, open_pos)
                if close_pos == -1:
                    break
                matches.append((open_pos, data[open_pos : close_pos + len(close_tag)]))
                start = close_pos + len(close_tag)
        matches.sort(key=lambda x: x[0])
        return matches
