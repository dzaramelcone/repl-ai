"""Parse XML-tagged responses from Claude."""

from __future__ import annotations

import re


def parse_response(response: str) -> list[tuple[str, str]]:
    """Extract (tag, content) tuples from XML-tagged response."""
    pattern = r"<(assistant-(?:chat|repl-in|repl-out))>(.*?)</\1>"
    return re.findall(pattern, response, re.DOTALL)
