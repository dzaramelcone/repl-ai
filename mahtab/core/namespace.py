"""Namespace management utilities."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mahtab.core.state import SessionState


def init_namespace(session: SessionState, globals_dict: dict | None = None, locals_dict: dict | None = None) -> None:
    """Initialize session with caller's namespace.

    Args:
        session: The session state to initialize.
        globals_dict: Caller's globals() dict. If None, uses empty dict.
        locals_dict: Caller's locals() dict. If None, uses globals_dict.
    """
    session.init_namespace(globals_dict, locals_dict)


def reload_module_if_imported(path: Path) -> tuple[bool, str | None]:
    """Try to reload a Python module if it's already imported.

    Args:
        path: Path to the Python file.

    Returns:
        Tuple of (was_reloaded, error_message).
        If reloaded successfully, returns (True, None).
        If not a Python file or not imported, returns (False, None).
        If reload failed, returns (False, error_message).
    """
    if path.suffix != ".py":
        return False, None

    # Find module name from path
    for _name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        mod_file = getattr(mod, "__file__", None)
        if mod_file and Path(mod_file).resolve() == path.resolve():
            try:
                importlib.reload(mod)
                return True, None
            except Exception as e:
                return False, str(e)

    return False, None


def ensure_cwd_in_path() -> None:
    """Ensure current working directory is in sys.path for imports."""
    import os

    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
