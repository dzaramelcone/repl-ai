"""Code execution utilities with output limiting."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mahtab.core.state import SessionState


class LimitedOutput:
    """StringIO-like wrapper that raises error if output exceeds limit."""

    def __init__(self, limit: int = 10000):
        self.limit = limit
        self.buffer: list[str] = []
        self.size = 0

    def write(self, s: str) -> int:
        self.size += len(s)
        if self.size > self.limit:
            raise RuntimeError(f"Output too large (>{self.limit} chars). Use slicing or summarize.")
        self.buffer.append(s)
        return len(s)

    def flush(self) -> None:
        pass

    def getvalue(self) -> str:
        return "".join(self.buffer)


def execute_code(
    code: str,
    session: SessionState,
    output_limit: int = 10000,
) -> tuple[str, bool]:
    """Execute code in the session's namespace.

    Args:
        code: Python code to execute.
        session: Session containing the namespace.
        output_limit: Maximum output size in characters.

    Returns:
        Tuple of (output_string, is_error).
    """
    old_stdout = sys.stdout
    sys.stdout = captured = LimitedOutput(limit=output_limit)

    try:
        # Try eval first (expression)
        try:
            result = eval(code, session.globals_ns, session.locals_ns)
            if result is not None:
                print(repr(result))
        except SyntaxError:
            # Fall back to exec (statement)
            exec(code, session.globals_ns, session.locals_ns)

        output = captured.getvalue()
        return output if output else "(no output)", False
    except Exception as e:
        return f"Error: {e}", True
    finally:
        sys.stdout = old_stdout


def execute_sandboxed(
    code: str,
    local_vars: dict[str, Any],
    output_limit: int = 10000,
) -> tuple[str, list[str], Exception | None]:
    """Execute code in a sandboxed environment with limited builtins.

    Used by RLM for safe code execution.

    Args:
        code: Python code to execute.
        local_vars: Local variables available to the code.
        output_limit: Maximum output size.

    Returns:
        Tuple of (final_result_or_none, output_lines, exception_or_none).
    """
    output_buffer: list[str] = []
    output_size = 0

    def capture_print(*args: Any, **_: Any) -> None:
        nonlocal output_size
        text = " ".join(str(a) for a in args)
        output_size += len(text)
        if output_size > output_limit:
            raise RuntimeError(f"Output too large (>{output_limit} chars). Use slicing or summarize.")
        output_buffer.append(text)

    # Add capture_print to local_vars
    exec_vars = {**local_vars, "print": capture_print}

    try:
        exec(code, {"__builtins__": {}}, exec_vars)
        return exec_vars.get("_final"), output_buffer, None
    except Exception as e:
        return None, output_buffer, e
