"""Text exploration tools for navigating large data.

These tools are shared between the main REPL and the RLM search algorithm.
"""

from __future__ import annotations

import re

from langchain_core.tools import tool


@tool
def peek(text: str, n: int = 2000) -> str:
    """Return first n characters of text.

    Useful for getting an initial view of a large text to understand its structure.

    Args:
        text: The text to peek at.
        n: Number of characters to return. Default 2000.

    Returns:
        The first n characters of the text.
    """
    return text[:n]


@tool
def grep(text: str, pattern: str) -> list[str]:
    """Return lines matching a regex pattern (case-insensitive).

    Useful for finding specific content within large text.

    Args:
        text: The text to search.
        pattern: Regex pattern to match.

    Returns:
        List of lines that match the pattern.
    """
    return [line for line in text.split("\n") if re.search(pattern, line, re.IGNORECASE)]


@tool
def partition(text: str, n: int = 10) -> list[str]:
    """Split text into n roughly equal chunks.

    Useful for dividing large text into manageable pieces for exploration.

    Args:
        text: The text to partition.
        n: Number of chunks to create. Default 10.

    Returns:
        List of text chunks.
    """
    chunk_size = max(1, len(text) // n)
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


# Also provide non-decorated versions for direct use in RLM sandbox
def peek_raw(text: str, n: int) -> str:
    """Raw peek function without tool decoration."""
    return text[:n]


def grep_raw(text: str, pattern: str) -> list[str]:
    """Raw grep function without tool decoration."""
    return [line for line in text.split("\n") if re.search(pattern, line, re.IGNORECASE)]


def partition_raw(text: str, n: int) -> list[str]:
    """Raw partition function without tool decoration."""
    chunk_size = max(1, len(text) // n)
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
