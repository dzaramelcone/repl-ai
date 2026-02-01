"""Interactive REPL implementation."""

from __future__ import annotations

import asyncio
import code
import re
import readline
import rlcompleter
import sys

from mahtab.agent.repl_agent import create_repl_agent
from mahtab.core.state import SessionState
from mahtab.io import MemoryStore, setup_logging
from mahtab.rlm.search import rlm
from mahtab.tools.files import create_file, open_in_editor, read_file
from mahtab.tools.skills import load_claude_sessions, load_skill
from mahtab.tools.text import grep_raw, partition_raw, peek_raw
from mahtab.ui.console import console
from mahtab.ui.panels import print_banner, print_output_panel, print_usage_panel
from mahtab.ui.streaming import StreamingHandler

# ANSI codes for prompt - wrapped in \001 \002 for readline length calculation
NUM = "\001\033[38;5;117m\002"  # bright blue-ish
DIM = "\001\033[38;5;242m\002"  # grey
RESET = "\001\033[0m\002"
CYAN = "\001\033[36m\002"
GREEN = "\001\033[32m\002"


def _approx_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def is_toggle_command(text: str) -> bool:
    """Check if input is the mode toggle command.

    Args:
        text: The user input text.

    Returns:
        True if this is a toggle command (single slash).
    """
    return text.strip() == "/"


def should_route_to_chat(text: str, mode: str) -> bool:
    """Check if input should be routed to chat based on mode.

    Args:
        text: The user input text.
        mode: Current input mode ("repl" or "chat").

    Returns:
        True if input should go to ask(), False for Python interpreter.
    """
    if not text.strip():
        return False
    return mode == "chat"


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
        self.input_mode = "repl"

    def toggle_mode(self) -> None:
        """Toggle between REPL and CHAT modes."""
        self.input_mode = "chat" if self.input_mode == "repl" else "repl"

    def __str__(self) -> str:
        # Capture any new readline history entries (user input)
        current_len = readline.get_current_history_length()
        while self._last_history_len < current_len:
            self._last_history_len += 1
            item = readline.get_history_item(self._last_history_len)
            if item and not item.startswith("ask("):
                self.session.record_activity(f"<user-repl-in>{item}</user-repl-in>")

        parts = []

        # Memory
        import resource

        mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
        parts.append(f"{NUM}{mem_mb:.0f}{DIM}MB")

        # Context size (if 'context' var exists)
        ctx = self.ns.get("context")
        if ctx and isinstance(ctx, str) and len(ctx) > 0:
            toks = _approx_tokens(ctx)
            parts.append(f"{DIM}ctx:{_format_tokens(toks)}")

        # History size
        if self.session.messages:
            hist_chars = sum(len(str(m.content)) for m in self.session.messages)
            hist_toks = hist_chars // 4
            parts.append(_format_tokens(hist_toks))

        # Usage stats (cost)
        if self.session.usage.num_calls > 0:
            cost = self.session.usage.total_cost_usd
            parts.append(f"{DIM}${NUM}{cost:.2f}{RESET}")

        info = " ".join(parts)
        if self.input_mode == "chat":
            mode_indicator = f"{GREEN}◇ ai{RESET}"
        else:
            mode_indicator = f"{CYAN}◈ py{RESET}"
        return f"{info}{RESET} {mode_indicator} " if info else f"{mode_indicator} "


class InteractiveREPL(code.InteractiveConsole):
    """REPL with mode switching between Python and chat.

    Type '/' on its own line to toggle between modes:
    - REPL mode (cyan diamond): Execute Python code directly
    - CHAT mode (green diamond): Send input to Claude

    Attributes:
        prompt_obj: DynamicPrompt instance for visual feedback.
        ask_func: Function to call for chat mode input.
    """

    def __init__(self, locals: dict, prompt_obj: DynamicPrompt, ask_func):
        super().__init__(locals)
        self.prompt_obj = prompt_obj
        self.ask_func = ask_func

        # Enable tab completion
        if locals:
            readline.set_completer(rlcompleter.Completer(locals).complete)
            readline.parse_and_bind("tab: complete")

    def runsource(self, source: str, filename: str = "<input>", symbol: str = "single") -> bool:
        """Execute source code or send to chat.

        Args:
            source: The source code or chat prompt.
            filename: The filename for error messages.
            symbol: The symbol for compilation.

        Returns:
            True if more input is needed, False otherwise.
        """
        source = source.rstrip()

        # Toggle mode with '/'
        if is_toggle_command(source):
            self.prompt_obj.toggle_mode()
            mode_name = "[green]chat[/]" if self.prompt_obj.input_mode == "chat" else "[cyan]repl[/]"
            console.print(f"[dim]switched to {mode_name} mode[/]")
            return False

        if not source:
            return False

        # Route based on mode
        if should_route_to_chat(source, self.prompt_obj.input_mode):
            self.ask_func(source)
            return False
        else:
            # Normal Python execution
            return super().runsource(source, filename, symbol)

    def interact(self, banner: str = "", exitmsg: str = "") -> None:
        """Custom interact loop with dynamic prompts.

        Args:
            banner: Banner to display at start.
            exitmsg: Message to display on exit.
        """
        if banner:
            self.write(f"{banner}\n")

        more = False
        while True:
            try:
                if more:
                    prompt = sys.ps2
                else:
                    prompt = str(self.prompt_obj)
                line = input(prompt)
                more = self.push(line)
            except KeyboardInterrupt:
                console.print("\n[dim]KeyboardInterrupt[/]")
                self.resetbuffer()
                more = False
            except EOFError:
                break
            except SystemExit:
                # exit() should exit the whole application, not just the eval loop
                sys.exit(0)

        if exitmsg:
            self.write(f"{exitmsg}\n")


def run_repl(ns: dict) -> None:
    """Run the interactive REPL.

    Args:
        ns: Namespace dict to use.
    """

    # Create session and agent
    session = SessionState()
    session.init_namespace(ns, ns)
    session.ensure_skills_dir()

    # Initialize structured I/O
    store = MemoryStore()
    log, prompt_handler = setup_logging(store)

    agent = create_repl_agent(session=session, model="claude-haiku-4-5-20251001", max_turns=5)
    streaming_handler = StreamingHandler(console=console, chars_per_second=200.0)

    # Create wrapper functions for the namespace
    def ask(prompt: str) -> None:
        """Ask Claude something. Claude can execute code in your namespace."""
        if not prompt:
            return

        # Log user input
        log.info(prompt, extra={"tag": "user-chat"})

        streaming_handler.reset()

        def handle_execution(output, is_error):
            print_output_panel(output, is_error, title="", console=console)

        async def run():
            return await agent.ask(
                prompt,
                streaming_handler=streaming_handler,
                on_execution=handle_execution,
            )

        try:
            asyncio.run(run())
        except KeyboardInterrupt:
            streaming_handler.cleanup()
            sys.stdout.write("\n\033[33m[cancelled]\033[0m\n")
            sys.stdout.flush()
            session.clear_activity()
            print("\033[0m", end="", flush=True)
            return

        # Record usage stats
        usage = streaming_handler.last_usage
        session.usage.record(
            cost=usage["total_cost_usd"],
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            cache_read=usage["cache_read_input_tokens"],
            cache_create=usage["cache_creation_input_tokens"],
        )

        session.clear_activity()
        print("\033[0m", end="", flush=True)

    def clear() -> None:
        """Clear conversation history."""
        session.clear_history()
        console.print("[dim]History cleared.[/]")

    def usage() -> None:
        """Show cumulative usage stats for this session."""
        print_usage_panel(session.usage.model_dump(), console=console)

    def ed(content: str, path: str, suffix: str) -> str:
        """Edit text in $EDITOR, return the result."""
        return open_in_editor(content, path, suffix, session.messages)

    def read(file_path: str, start: int, end: int) -> str:
        """Read a file with line numbers."""
        return read_file.invoke({"file_path": file_path, "start": start, "end": end})

    def edit(file_path: str, old: str, new: str) -> str:
        """Replace text in a file."""
        from mahtab.tools.files import edit_file

        return edit_file.invoke({"file_path": file_path, "old": old, "new": new})

    def create(name: str, content: str) -> str:
        """Create a new Python module."""
        return create_file.invoke({"name": name, "content": content})

    def skill(name: str, args: str) -> str:
        """Load a skill."""
        return load_skill.invoke({"name": name, "args": args, "skills_dir": session.skills_dir})

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

    # Set up prompts
    prompt_obj = DynamicPrompt(session, ns)
    sys.ps1 = prompt_obj
    sys.ps2 = "\033[2m⋮\033[0m "

    # Print banner
    print_banner(console=console)

    # Create and run modal REPL
    repl = InteractiveREPL(locals=ns, prompt_obj=prompt_obj, ask_func=ask)
    repl.interact()
