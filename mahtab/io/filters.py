"""Logging filters for tag-based routing."""

import logging


class TagFilter(logging.Filter):
    """Filter log records by tag attribute."""

    def __init__(self, tags: set[str]) -> None:
        super().__init__()
        self.tags = tags

    def filter(self, record: logging.LogRecord) -> bool:
        return record.tag in self.tags
