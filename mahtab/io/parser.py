"""Parse XML-tagged responses from Claude."""

import logging
import re


def parse_response(response: str) -> list[tuple[str, str]]:
    """Extract (tag, content) tuples from XML-tagged response."""
    pattern = r"<(assistant-(?:chat|repl-in|repl-out))>(.*?)</\1>"
    return re.findall(pattern, response, re.DOTALL)


def route_response(response: str) -> None:
    """Parse response and route each tagged section to the logger."""
    log = logging.getLogger("mahtab")
    for tag, content in parse_response(response):
        log.info(content, extra={"tag": tag})
