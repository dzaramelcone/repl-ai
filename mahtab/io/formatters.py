"""Logging formatters for different output targets."""

from __future__ import annotations

import logging


class XMLFormatter(logging.Formatter):
    """Wraps log message in XML tag from record.tag attribute."""

    def format(self, record: logging.LogRecord) -> str:
        return f"<{record.tag}>{record.getMessage()}</{record.tag}>"
