"""Interactive REPL implementation."""

from __future__ import annotations

import asyncio
import re
import readline
import sys

from mahtab.agent.repl_agent import create_repl_agent
from mahtab.core.state import SessionState
from mahtab.rlm.search import rlm
from mahtab.tools.files import create_file, open_in_editor, read_file
from mahtab.tools.skills import load_claude_sessions, load_skill
from mahtab.tools.text import grep_raw, partition_raw, peek_raw
from mahtab.ui.console import console
from mahtab.ui.panels import print_banner, print_usage_panel
from mahtab.ui.streaming import StreamingHandler

# ANSI codes for prompt
NUM = "\033[38;5;117m"  # bright blue-ish
DIM = "\033[38;5;242m"  # grey
RESET = "\033[0m"
CYAN = "\033[36m"


def _approx_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def _format_tokens(n: int) -> str:
    """Format token count with units."""
    if n >= 1_000_000_000:
        return f"{NUM}{n / 1_000_000_000:.1f}{DIM}Gt"
    elif n >= 1_000_000:
        return f"{NUM}{n / 1_000_000:.1f}{DIM}Mt"
    elif n >= 1_000:
        return f"{NUM}{n / 1_000:.1f}{DIM}kt"
    return f"{NUM}{n}{DIM}t"


class DynamicPrompt:
    """Dynamic prompt that shows memory, context, and cost info."""

    def __init__(self, session: SessionState, ns: dict):
        self.session = session
        self.ns = ns
        self._last_history_len = readline.get_current_history_length()

    def __str__(self) -> str:
        # Capture any new readline history entries (user input)
        current_len = readline.get_current_history_length()
        while self._last_history_len < current_len:
            self._last_history_len += 1
            item = readline.get_history_item(self._last_history_len)
            if item and not item.startswith("ask("):
                self.session.record_activity(f">>> {item}")

        parts = []

        # Memory
        try:
            import resource

            mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
            parts.append(f"{NUM}{mem_mb:.0f}{DIM}MB")
        except Exception:
            pass

        # Context size (if 'context' var exists)
        ctx = self.ns.get("context")
        if ctx and isinstance(ctx, str) and len(ctx) > 0:
            toks = _approx_tokens(ctx)
            parts.append(f"{DIM}ctx:{_format_tokens(toks)}")

        # History size
        if self.session.messages:
            hist_chars = sum(len(str(m.content)) for m in self.session.messages)
            hist_toks = hist_chars // 4
            parts.append(f"{DIM}hist:{_format_tokens(hist_toks)}")

        # Usage stats (cost)
        if self.session.usage.num_calls > 0:
            cost = self.session.usage.total_cost_usd
            parts.append(f"{DIM}${NUM}{cost:.2f}{RESET}")

        info = " ".join(parts)
        return f"{DIM}{info}{RESET} {CYAN}◈{RESET} " if info else f"{CYAN}◈{RESET} "


def run_repl(ns: dict | None = None) -> None:
    """Run the interactive REPL.

    Args:
        ns: Namespace dict to use. If None, uses caller's globals.
    """
    import inspect

    if ns is None:
        frame = inspect.currentframe()
        if frame and frame.f_back:
            ns = frame.f_back.f_globals
        else:
            ns = {}

    # Create session and agent
    session = SessionState()
    session.init_namespace(ns, ns)
    session.ensure_skills_dir()

    agent = create_repl_agent(session=session, console=console)
    streaming_handler = StreamingHandler(console=console)

    # Create wrapper functions for the namespace
    def ask(prompt: str = "") -> None:
        """Ask Claude something. Claude can execute code in your namespace."""
        if not prompt:
            return

        try:
            streaming_handler.reset()

            async def run():
                return await agent.ask(prompt, streaming_handler=streaming_handler)

            asyncio.run(run())

        except KeyboardInterrupt:
            streaming_handler.cleanup()
            sys.stdout.write("\n\033[33m[cancelled]\033[0m\n")
            sys.stdout.flush()
        finally:
            session.clear_activity()
            print("\033[0m", end="", flush=True)

    def clear() -> None:
        """Clear conversation history."""
        session.clear_history()
        console.print("[dim]History cleared.[/]")

    def usage() -> None:
        """Show cumulative usage stats for this session."""
        print_usage_panel(session.usage.model_dump())

    def ed(content: str = "", path: str | None = None, suffix: str = ".py") -> str:
        """Edit text in $EDITOR, return the result."""
        return open_in_editor(content, path, suffix, session.messages)

    def read(file_path: str, start: int = 1, end: int | None = None) -> str:
        """Read a file with line numbers."""
        return read_file.invoke({"file_path": file_path, "start": start, "end": end})

    def edit(file_path: str, old: str, new: str) -> str:
        """Replace text in a file."""
        from mahtab.tools.files import edit_file

        return edit_file.invoke({"file_path": file_path, "old": old, "new": new})

    def create(name: str, content: str = "") -> str:
        """Create a new Python module."""
        return create_file.invoke({"name": name, "content": content})

    def skill(name: str, args: str = "") -> str:
        """Load a skill."""
        return load_skill.invoke({"name": name, "args": args})

    # Add functions to namespace
    ns.update(
        {
            "ask": ask,
            "clear": clear,
            "usage": usage,
            "ed": ed,
            "read": read,
            "edit": edit,
            "create": create,
            "skill": skill,
            "peek": peek_raw,
            "grep": grep_raw,
            "partition": partition_raw,
            "rlm": rlm,
            "load_claude_sessions": load_claude_sessions,
            "re": re,
        }
    )

    # Set up dynamic prompt
    sys.ps1 = DynamicPrompt(session, ns)
    sys.ps2 = "\033[2m⋮\033[0m "

    # Print banner
    print_banner()
