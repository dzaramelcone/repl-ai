"""Modal REPL implementation with backtick mode switching."""

from __future__ import annotations

import code as code_module
import readline
import rlcompleter

from rich.console import Console

from mahtab.agent.repl_agent import create_repl_agent
from mahtab.core.state import SessionState
from mahtab.rlm.search import rlm
from mahtab.tools.skills import load_claude_sessions
from mahtab.ui.console import console as default_console
from mahtab.ui.panels import print_modal_banner


class ModalREPL(code_module.InteractiveConsole):
    """REPL that toggles between ask mode (Claude) and python mode.

    Use backtick (`) to toggle between modes:
    - Python mode (cyan prompt): Execute Python code directly
    - Ask mode (magenta prompt): Send input to Claude

    Attributes:
        session: The session state.
        console: Rich console for output.
        ask_mode: Whether currently in ask mode.
    """

    def __init__(
        self,
        session: SessionState | None = None,
        console: Console | None = None,
        locals: dict | None = None,
    ):
        super().__init__(locals)

        self.session = session or SessionState()
        self.console = console or default_console
        self.ask_mode = False
        self._agent = None

        # Enable tab completion
        if locals:
            readline.set_completer(rlcompleter.Completer(locals).complete)
            readline.parse_and_bind("tab: complete")

    @property
    def agent(self):
        """Lazy-load the agent."""
        if self._agent is None:
            self._agent = create_repl_agent(session=self.session, console=self.console)
        return self._agent

    @property
    def prompt(self) -> str:
        """Get the current prompt based on mode."""
        if self.ask_mode:
            return "\033[35m◈\033[0m "  # Magenta for ask mode
        else:
            return "\033[36m◈\033[0m "  # Cyan for python mode

    @property
    def prompt2(self) -> str:
        """Get the continuation prompt."""
        return "\033[2m⋮\033[0m "

    def runsource(self, source: str, filename: str = "<input>", symbol: str = "single") -> bool:
        """Execute source code or send to Claude.

        Args:
            source: The source code or prompt.
            filename: The filename for error messages.
            symbol: The symbol for compilation.

        Returns:
            True if more input is needed, False otherwise.
        """
        source = source.rstrip()

        # Toggle mode with backtick
        if source == "`":
            self.ask_mode = not self.ask_mode
            mode = "[magenta]ask[/]" if self.ask_mode else "[cyan]python[/]"
            self.console.print(f"[dim]switched to {mode} mode[/]")
            return False

        if not source:
            return False

        if self.ask_mode:
            # Send to Claude
            self.agent.ask_sync(source)
            return False
        else:
            # Normal Python execution
            return super().runsource(source, filename, symbol)

    def interact(self, banner: str | None = None, exitmsg: str | None = None) -> None:
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
                    prompt = self.prompt2
                else:
                    prompt = self.prompt
                try:
                    line = input(prompt)
                except EOFError:
                    self.write("\n")
                    break
                more = self.push(line)
            except KeyboardInterrupt:
                self.console.print("\n[dim]KeyboardInterrupt[/]")
                self.resetbuffer()
                more = False

        if exitmsg:
            self.write(f"{exitmsg}\n")


def run_modal_repl(ns: dict | None = None, console: Console | None = None) -> None:
    """Start the modal REPL with the given namespace.

    Args:
        ns: Namespace dict to use. If None, uses empty dict.
        console: Rich console for output.
    """
    if ns is None:
        ns = {}

    if console is None:
        console = default_console

    # Create session
    session = SessionState()
    session.init_namespace(ns, ns)
    session.ensure_skills_dir()

    # Create modal REPL
    repl = ModalREPL(session=session, console=console, locals=ns)

    # Add helper functions to namespace
    def clear() -> None:
        """Clear conversation history."""
        session.clear_history()
        console.print("[dim]History cleared.[/]")

    ns.update({
        "clear": clear,
        "rlm": rlm,
        "load_claude_sessions": load_claude_sessions,
    })

    # Print banner
    print_modal_banner(console)

    # Start REPL
    repl.interact()
