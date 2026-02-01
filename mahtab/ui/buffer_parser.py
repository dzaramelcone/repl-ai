"""Buffer parsing utilities for streaming output."""

from __future__ import annotations

import re


def find_close_tag(buffer: str, close_tag: str) -> tuple[str | None, str]:
    """Find close tag in buffer.

    Returns (content_before_tag, remaining_buffer) if found,
    or (None, buffer) if not found.
    """
    if close_tag in buffer:
        idx = buffer.find(close_tag)
        return buffer[:idx], buffer[idx + len(close_tag) :]
    return None, buffer


def find_partial_close(buffer: str) -> tuple[str, str]:
    """Check for partial close tag '</'.

    Returns (content_before, remaining) where remaining starts with '</' if found.
    """
    if "</" in buffer:
        idx = buffer.find("</")
        if idx > 0:
            return buffer[:idx], buffer[idx:]
    return "", buffer


def find_code_fence_start(buffer: str) -> tuple[str | None, str]:
    """Find markdown code fence start (```lang).

    Returns (language, remaining_buffer) if found,
    or (None, buffer) if not found.
    """
    code_match = re.match(r"```(\w*)\n", buffer)
    if code_match:
        lang = code_match.group(1) or "text"
        return lang, buffer[code_match.end() :]
    return None, buffer


def has_partial_backticks(buffer: str) -> tuple[str, str]:
    """Check for trailing partial backticks.

    Returns (content_before_backticks, backticks) if buffer ends with 1-2 backticks,
    or (buffer, "") if no trailing backticks.
    """
    if buffer.endswith("`") or buffer.endswith("``"):
        idx = buffer.rfind("`")
        while idx > 0 and buffer[idx - 1] == "`":
            idx -= 1
        return buffer[:idx], buffer[idx:]
    return buffer, ""


def find_code_fence_end(buffer: str) -> tuple[str | None, str]:
    """Find markdown code fence end.

    Returns (content, remaining_buffer) if found,
    or (None, buffer) if not found.
    """
    if "\n```" in buffer:
        idx = buffer.find("\n```")
        return buffer[:idx], buffer[idx + 4 :]
    if buffer.startswith("```"):
        return "", buffer[3:]
    return None, buffer
