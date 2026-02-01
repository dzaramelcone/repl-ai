"""File operation tools for reading, editing, and creating files."""

from __future__ import annotations

import os
from pathlib import Path

from langchain_core.tools import tool

from mahtab.core.namespace import ensure_cwd_in_path, reload_module_if_imported


@tool
def read_file(file_path: str, start: int = 1, end: int | None = None) -> str:
    """Read a file and return its contents with line numbers.

    Args:
        file_path: Path to the file to read.
        start: Starting line number (1-indexed). Default 1.
        end: Ending line number (inclusive). Default None (read to end).

    Returns:
        File contents with line numbers, or error message.
    """
    path = Path(file_path).expanduser()

    if not path.exists():
        return f"Error: {path} does not exist"

    lines = path.read_text().split("\n")

    if end is None:
        end = len(lines)

    start = max(1, start)
    end = min(len(lines), end)

    result = []
    for i in range(start - 1, end):
        result.append(f"{i + 1:4d}│ {lines[i]}")

    return "\n".join(result)


@tool
def edit_file(file_path: str, old: str, new: str) -> str:
    """Edit a file by replacing old text with new text.

    If the file is a Python module that's already imported, it will be reloaded.

    Args:
        file_path: Path to the file to edit.
        old: Text to find and replace.
        new: Replacement text.

    Returns:
        Status message indicating success or error.
    """
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        return f"Error: {path} does not exist"

    content = path.read_text()

    if old not in content:
        # Show a preview of the file to help debug
        lines = content.split("\n")
        preview = "\n".join(lines[:20])
        return f"Error: old text not found in {path}\n\nFirst 20 lines:\n{preview}"

    count = content.count(old)
    if count > 1:
        return f"Error: old text appears {count} times in {path}. Make it more specific."

    new_content = content.replace(old, new, 1)
    path.write_text(new_content)

    # Try to reload if it's an imported module
    reloaded, error = reload_module_if_imported(path)
    if error:
        return f"OK: edited {path} (reload failed: {error})"
    if reloaded:
        return f"OK: edited and reloaded {path}"
    return f"OK: edited {path}"


@tool
def create_file(name: str, content: str = "") -> str:
    """Create a new Python module that can be imported.

    Args:
        name: Module name (e.g. "utils" creates utils.py, "foo.bar" creates foo/bar.py).
        content: Initial content for the module. Default: empty with docstring.

    Returns:
        Status message with import instruction.
    """
    # Handle dotted names (foo.bar -> foo/bar.py)
    parts = name.split(".")
    if len(parts) > 1:
        # Create parent directories
        dir_path = Path(os.getcwd()) / "/".join(parts[:-1])
        dir_path.mkdir(parents=True, exist_ok=True)
        # Create __init__.py files for packages
        for i in range(len(parts) - 1):
            init_path = Path(os.getcwd()) / "/".join(parts[: i + 1]) / "__init__.py"
            if not init_path.exists():
                init_path.write_text("")
        file_path = dir_path / f"{parts[-1]}.py"
    else:
        file_path = Path(os.getcwd()) / f"{name}.py"

    if file_path.exists():
        return f"Error: {file_path} already exists"

    # Default content with docstring
    if not content:
        content = f'"""{name} module."""\n'

    file_path.write_text(content)

    # Ensure cwd is in sys.path for imports
    ensure_cwd_in_path()

    return f"OK: created {file_path}\n→ import {name}"


def open_in_editor(content: str = "", path: str | None = None, suffix: str = ".py", history: list | None = None) -> str:
    """Edit text in $EDITOR, return the result.

    Args:
        content: Initial content to edit (ignored if path is provided).
        path: If provided, edit this file directly instead of a temp file.
        suffix: File extension for temp file (default: .py for syntax highlighting).
        history: Optional conversation history for context.

    Returns:
        The edited content as a string.
    """
    import subprocess
    import tempfile

    editor = os.environ.get("EDITOR", "vim")

    if path:
        # Edit existing file directly
        file_path = Path(path).expanduser()
        subprocess.call([editor, str(file_path)])
        return file_path.read_text()

    # Build header with help and context
    header_lines = [
        "# ╭─────────────────────────────────────────────────────────────╮",
        "# │  VIM: i=insert  Esc=normal  :wq=save+quit  :q!=quit no save │",
        "# │  Delete these comments. Only text below --- is returned.   │",
        "# ╰─────────────────────────────────────────────────────────────╯",
    ]

    # Add last assistant message as context
    if history:
        for msg in reversed(history):
            if hasattr(msg, "type") and msg.type == "ai":
                last_response = msg.content
                # Truncate and format as comments
                preview = last_response[:500]
                if len(last_response) > 500:
                    preview += "..."
                header_lines.append("#")
                header_lines.append("# Last response from Claude:")
                for line in preview.split("\n"):
                    header_lines.append(f"#   {line}")
                break

    header_lines.append("#")
    header_lines.append("# --- YOUR MESSAGE BELOW (everything above this line is ignored) ---")
    header_lines.append("")

    header = "\n".join(header_lines)
    full_content = header + content

    # Use temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
        f.write(full_content)
        temp_path = f.name

    try:
        subprocess.call([editor, temp_path])
        with open(temp_path) as f:
            result = f.read()
    finally:
        os.unlink(temp_path)

    # Strip header - find the marker line and return everything after
    marker = "# --- YOUR MESSAGE BELOW"
    if marker in result:
        _, _, after = result.partition(marker)
        # Skip the rest of the marker line and the blank line after
        lines = after.split("\n", 2)
        if len(lines) >= 2:
            result = lines[2] if len(lines) > 2 else ""
        else:
            result = ""

    return result.strip()
