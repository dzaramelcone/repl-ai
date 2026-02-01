"""Logger setup and wiring."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mahtab.io.filters import TagFilter
from mahtab.io.handlers import DisplayHandler, PromptHandler, StoreHandler
from mahtab.io.tags import COMPLETE_TAGS, STREAM_TAGS
from mahtab.ui.console import console

if TYPE_CHECKING:
    from mahtab.io.handlers import Store


def setup_logging(store: Store) -> tuple[logging.Logger, PromptHandler]:
    """Configure the mahtab logger with all handlers."""
    log = logging.getLogger("mahtab")
    log.setLevel(logging.INFO)

    # Prompt handler: all complete message tags
    prompt = PromptHandler()
    prompt.addFilter(TagFilter(COMPLETE_TAGS))
    log.addHandler(prompt)

    # Display handler: complete tags except assistant-chat (streamed), plus stream tag
    display_tags = (COMPLETE_TAGS - {"assistant-chat"}) | STREAM_TAGS
    display = DisplayHandler(console)
    display.addFilter(TagFilter(display_tags))
    log.addHandler(display)

    # Store handler: all complete message tags
    store_handler = StoreHandler(store)
    store_handler.addFilter(TagFilter(COMPLETE_TAGS))
    log.addHandler(store_handler)

    return log, prompt
