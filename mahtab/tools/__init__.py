"""Tools for file operations, text exploration, and skills management."""

from mahtab.tools.files import create_file, edit_file, open_in_editor, read_file
from mahtab.tools.skills import load_claude_sessions, load_skill, load_skill_descriptions
from mahtab.tools.text import grep, grep_raw, partition, partition_raw, peek, peek_raw

__all__ = [
    # Text tools (with @tool decorator)
    "peek",
    "grep",
    "partition",
    # Text tools (raw, for sandbox use)
    "peek_raw",
    "grep_raw",
    "partition_raw",
    # File tools
    "read_file",
    "edit_file",
    "create_file",
    "open_in_editor",
    # Skills
    "load_skill",
    "load_skill_descriptions",
    "load_claude_sessions",
]
