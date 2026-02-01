"""Structured I/O with XML tags and logging-based routing."""

from mahtab.io.filters import TagFilter
from mahtab.io.formatters import BytesFormatter, RichFormatter, XMLFormatter
from mahtab.io.handlers import DisplayHandler, PromptHandler, Store, StoreHandler
from mahtab.io.parser import parse_response, route_response
from mahtab.io.setup import setup_logging
from mahtab.io.tags import COMPLETE_TAGS, STREAM_TAGS, TAGS

__all__ = [
    "TAGS",
    "STREAM_TAGS",
    "COMPLETE_TAGS",
    "TagFilter",
    "XMLFormatter",
    "RichFormatter",
    "BytesFormatter",
    "PromptHandler",
    "DisplayHandler",
    "StoreHandler",
    "Store",
    "parse_response",
    "route_response",
    "setup_logging",
]
