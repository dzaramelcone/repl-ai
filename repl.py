"""
Shared REPL with Claude.
Claude can execute code in your namespace.
"""
import asyncio
import re
import json

from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.panel import Panel
from rich.text import Text

console = Console()

# Will be set to caller's globals/locals
_globals = None
_locals = None
_history = []


def init(g=None, l=None):
    """Initialize with your namespace. Call as: init(globals(), locals())"""
    global _globals, _locals
    _globals = g if g is not None else {}
    _locals = l if l is not None else _globals


def _print_code(code: str, title: str = "Code"):
    """Print syntax-highlighted Python code in a panel."""
    syntax = Syntax(code, "python", theme="monokai", line_numbers=True, indent_guides=True)
    console.print(Panel(syntax, title=f"[bold cyan]{title}[/]", border_style="cyan"))


def _print_output(output: str, is_error: bool = False):
    """Print execution output."""
    style = "red" if is_error else "green"
    title = "Error" if is_error else "Output"
    console.print(Panel(output, title=f"[bold {style}]{title}[/]", border_style=style))


async def ask(prompt: str, max_turns: int = 5) -> str:
    """
    Ask Claude something. Claude can execute code in your namespace.
    Streams response to stdout. Returns final text response.
    """
    global _history

    if _globals is None:
        raise RuntimeError("Call init(globals(), locals()) first")

    # Build context about available variables
    var_summary = _summarize_namespace()

    system = f"""You're in a shared Python REPL with the user. You can see and modify their namespace.

Available variables:
{var_summary}

File tools:
  read(path, start=1, end=None) -> str   # Read file with line numbers
  edit(path, old, new) -> str            # Replace old text with new text in file

Large text exploration:
  load_claude_sessions() -> str     # Load all ~/.claude/projects/*.jsonl into a string
  rlm(query, context) -> str        # Recursive LLM search over large context

Inside rlm, you have: peek(n), grep(pattern), partition(n), FINAL(answer)

When you want to run code, output a fenced python block. The code will execute in the user's namespace and you'll see the output. You can run multiple code blocks in one response.

When you're done and have a final answer, just respond with text (no code block).

Keep responses concise."""

    _history.append({"role": "user", "content": prompt})

    for _ in range(max_turns):
        # Call Claude with streaming
        response = await _call_claude_stream(system, _history)

        # Extract code blocks
        code_blocks = re.findall(r'```python\n(.*?)```', response, re.DOTALL)

        if not code_blocks:
            # No code, just a text response - we're done
            _history.append({"role": "assistant", "content": response})
            return response

        # Execute code blocks and collect output
        # (code already shown during streaming, just show execution output)
        outputs = []
        for i, code in enumerate(code_blocks):
            code = code.strip()
            output = _exec_code(code)
            outputs.append(output)
            is_error = output.startswith("Error:")
            _print_output(output, is_error)

        # Add assistant response and execution results to history
        _history.append({"role": "assistant", "content": response})

        exec_report = "\n\n".join(
            f"Code block {i+1} output:\n{out}"
            for i, out in enumerate(outputs)
        )
        _history.append({"role": "user", "content": f"[Execution results]\n{exec_report}"})

    return "(max turns reached)"


def _summarize_namespace() -> str:
    """Summarize variables in the namespace."""
    if not _globals:
        return "(empty)"

    lines = []
    for name, val in _globals.items():
        if name.startswith('_'):
            continue
        try:
            typ = type(val).__name__
            if isinstance(val, (int, float, str, bool, type(None))):
                rep = repr(val)[:50]
            elif isinstance(val, (list, dict, set, tuple)):
                rep = f"{typ} with {len(val)} items"
            else:
                rep = typ
            lines.append(f"  {name}: {rep}")
        except:
            lines.append(f"  {name}: <unknown>")

    return "\n".join(lines[:30]) or "(no user variables)"


class _LimitedOutput:
    """StringIO wrapper that raises error if output exceeds limit."""
    def __init__(self, limit: int = 10000):
        self.limit = limit
        self.buffer = []
        self.size = 0

    def write(self, s: str):
        self.size += len(s)
        if self.size > self.limit:
            raise RuntimeError(f"Output too large (>{self.limit} chars). Use slicing or summarize.")
        self.buffer.append(s)

    def flush(self):
        pass

    def getvalue(self) -> str:
        return "".join(self.buffer)


def _exec_code(code: str) -> str:
    """Execute code in the shared namespace, return output."""
    import sys

    old_stdout = sys.stdout
    sys.stdout = captured = _LimitedOutput(limit=10000)

    try:
        # Try eval first (expression)
        try:
            result = eval(code, _globals, _locals)
            if result is not None:
                print(repr(result))
        except SyntaxError:
            # Fall back to exec (statement)
            exec(code, _globals, _locals)

        output = captured.getvalue()
        return output if output else "(no output)"
    except Exception as e:
        return f"Error: {e}"
    finally:
        sys.stdout = old_stdout


async def _call_claude_stream(system: str, messages: list) -> str:
    """Call Claude CLI with streaming output.

    Features:
    - Spinner while waiting for first token
    - Smooth typewriter animation for text
    - Live-updating code panels as code streams in
    """
    import sys
    from collections import deque

    # Build conversation as a single prompt
    prompt_parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            prompt_parts.append(f"User: {content}")
        else:
            prompt_parts.append(f"Assistant: {content}")

    full_prompt = "\n\n".join(prompt_parts)

    proc = await asyncio.create_subprocess_exec(
        "claude", "-p", full_prompt,
        "--system-prompt", system,
        "--setting-sources", "",
        "--output-format", "stream-json",
        "--include-partial-messages",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd="/tmp",
    )

    full_response = ""
    first_token = True
    spinner = None

    # State machine for code block detection
    in_code_block = False
    text_buffer = ""  # Buffer to detect ```python
    code_buffer = ""  # Buffer for code block content
    code_live = None  # Live display for code

    # Typewriter animation buffer
    output_queue = deque()
    typing_done = asyncio.Event()

    def _write_char(char: str):
        """Write a single character to stdout."""
        sys.stdout.write(char)
        sys.stdout.flush()

    def _make_code_panel(code: str, done: bool = False) -> Panel:
        title = "[cyan]Code[/]" if done else "[dim cyan]Writing...[/]"
        return Panel(
            Syntax(code or " ", "python", theme="monokai", line_numbers=True),
            title=title,
            border_style="cyan" if done else "dim"
        )

    async def typewriter():
        """Drain output queue with smooth animation."""
        chars_per_tick = 3  # Characters to output per tick
        tick_delay = 0.012  # 12ms between ticks (~250 chars/sec)

        while not typing_done.is_set() or output_queue:
            if output_queue:
                for _ in range(min(chars_per_tick, len(output_queue))):
                    if output_queue:
                        _write_char(output_queue.popleft())
            await asyncio.sleep(tick_delay)

    # Start typewriter task
    typewriter_task = asyncio.create_task(typewriter())

    try:
        # Show spinner while waiting
        spinner = Live(Spinner("dots", text="[dim]thinking...[/]"), console=console, refresh_per_second=10)
        spinner.start()

        async for line in proc.stdout:
            line = line.decode().strip()
            if not line:
                continue
            try:
                data = json.loads(line)

                # Handle streaming deltas
                if data.get("type") == "stream_event":
                    event = data.get("event", {})
                    if event.get("type") == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")

                            # Stop spinner on first token
                            if first_token and spinner:
                                spinner.stop()
                                spinner = None
                                first_token = False

                            full_response += text

                            # State machine: handle code blocks with live display
                            for char in text:
                                if in_code_block:
                                    code_buffer += char
                                    # Update live code panel
                                    if code_live and not code_buffer.endswith("```"):
                                        code_live.update(_make_code_panel(code_buffer.rstrip('`')))
                                    # Check for closing ```
                                    if code_buffer.endswith("```"):
                                        # Finalize code panel
                                        final_code = code_buffer[:-3]
                                        if code_live:
                                            code_live.update(_make_code_panel(final_code, done=True))
                                            code_live.stop()
                                            code_live = None
                                        in_code_block = False
                                        code_buffer = ""
                                else:
                                    text_buffer += char
                                    if "```python\n" in text_buffer or "```python\r\n" in text_buffer:
                                        # Flush text before code block
                                        idx = text_buffer.find("```python")
                                        if idx > 0:
                                            for c in text_buffer[:idx]:
                                                output_queue.append(c)
                                        # Wait for pending text to flush
                                        while output_queue:
                                            await asyncio.sleep(0.01)
                                        text_buffer = ""
                                        code_buffer = ""
                                        in_code_block = True
                                        # Start live code panel
                                        sys.stdout.write("\n")
                                        code_live = Live(_make_code_panel(""), console=console, refresh_per_second=15)
                                        code_live.start()
                                    elif len(text_buffer) > 20 and "```" not in text_buffer:
                                        for c in text_buffer:
                                            output_queue.append(c)
                                        text_buffer = ""

                # Handle final result
                elif data.get("type") == "result":
                    if not full_response:
                        full_response = data.get("result", "")

            except json.JSONDecodeError:
                pass

        # Clean up any open code panel
        if code_live:
            code_live.stop()

        # Flush remaining text buffer (only if not in code block)
        if text_buffer and not in_code_block:
            for c in text_buffer:
                output_queue.append(c)

        # Signal typewriter to finish and wait
        typing_done.set()
        await typewriter_task

        _write_char("\n")  # newline after stream
        await proc.wait()

    except asyncio.CancelledError:
        if spinner:
            spinner.stop()
        if code_live:
            code_live.stop()
        typing_done.set()
        typewriter_task.cancel()
        proc.terminate()
        raise
    finally:
        if spinner:
            spinner.stop()

    if proc.returncode != 0:
        stderr = await proc.stderr.read()
        raise RuntimeError(f"claude failed: {stderr.decode()}")

    return full_response.strip()


def clear():
    """Clear conversation history."""
    global _history
    _history = []
    console.print("[dim]History cleared.[/]")


def edit(file_path: str, old: str, new: str) -> str:
    """
    Edit a file by replacing old text with new text.
    If the file is a Python module that's already imported, it will be reloaded.
    Returns a status message.
    """
    from pathlib import Path
    import sys
    import importlib

    path = Path(file_path).expanduser().resolve()

    # Don't allow editing ourselves
    self_path = Path(__file__).resolve()
    if path == self_path:
        return f"Error: cannot edit {path} while it's running"

    if not path.exists():
        return f"Error: {path} does not exist"

    content = path.read_text()

    if old not in content:
        # Show a preview of the file to help debug
        lines = content.split('\n')
        preview = '\n'.join(lines[:20])
        return f"Error: old text not found in {path}\n\nFirst 20 lines:\n{preview}"

    count = content.count(old)
    if count > 1:
        return f"Error: old text appears {count} times in {path}. Make it more specific."

    new_content = content.replace(old, new, 1)
    path.write_text(new_content)

    # Try to reload if it's an imported module
    reloaded = False
    if path.suffix == '.py':
        # Find module name from path
        for name, mod in list(sys.modules.items()):
            if mod is None:
                continue
            mod_file = getattr(mod, '__file__', None)
            if mod_file and Path(mod_file).resolve() == path:
                try:
                    importlib.reload(mod)
                    reloaded = True
                    break
                except Exception as e:
                    return f"OK: edited {path} (reload failed: {e})"

    if reloaded:
        return f"OK: edited and reloaded {path}"
    return f"OK: edited {path}"


def read(file_path: str, start: int = 1, end: int = None) -> str:
    """
    Read a file and return its contents with line numbers.
    Optionally specify start and end line numbers.
    """
    from pathlib import Path

    path = Path(file_path).expanduser()

    if not path.exists():
        return f"Error: {path} does not exist"

    lines = path.read_text().split('\n')

    if end is None:
        end = len(lines)

    start = max(1, start)
    end = min(len(lines), end)

    result = []
    for i in range(start - 1, end):
        result.append(f"{i + 1:4d}│ {lines[i]}")

    return '\n'.join(result)


def load_claude_sessions(projects_path: str = "~/.claude/projects") -> str:
    """Load all JSONL files from Claude projects into one big context."""
    from pathlib import Path

    path = Path(projects_path).expanduser()
    chunks = []

    for jsonl_file in sorted(path.rglob("*.jsonl")):
        rel_path = jsonl_file.relative_to(path)
        content = jsonl_file.read_text()
        chunks.append(f"\n\n=== FILE: {rel_path} ===\n{content}")

    result = "".join(chunks)

    # Token estimate (~4 chars per token)
    tokens = len(result) // 4
    if tokens >= 1_000_000:
        tok_str = f"{tokens/1_000_000:.1f}Mt"
    elif tokens >= 1_000:
        tok_str = f"{tokens/1_000:.1f}kt"
    else:
        tok_str = f"{tokens}t"

    console.print(f"[dim]Loaded[/] [bold cyan]{tok_str}[/] [dim]from {len(chunks)} files[/]")
    return result


def q(prompt: str):
    """
    Fire-and-forget ask. Returns a task you can await later.
    Usage: t = q("do something"); ... ; await t
    """
    return asyncio.create_task(ask(prompt))


def ask_sync(prompt: str, max_turns: int = 5) -> None:
    """Synchronous version of ask() - blocks until complete, streams to stdout."""
    import sys
    try:
        asyncio.run(ask(prompt, max_turns))
    except KeyboardInterrupt:
        sys.stdout.write("\n\033[33m[cancelled]\033[0m\n")
        sys.stdout.flush()
    # Print empty line and reset ANSI to ensure prompt appears correctly
    print("\033[0m", end="", flush=True)


# Modal REPL
import code
import readline
import rlcompleter

class ModalREPL(code.InteractiveConsole):
    """REPL that toggles between ask mode (Claude) and python mode."""

    def __init__(self, locals=None):
        super().__init__(locals)
        self.ask_mode = False
        self.python_buffer = []  # For multi-line python input

        # Enable tab completion
        if locals:
            readline.set_completer(rlcompleter.Completer(locals).complete)
            readline.parse_and_bind("tab: complete")

    @property
    def prompt(self):
        if self.ask_mode:
            return "\033[35m◈\033[0m "  # Magenta for ask mode
        else:
            return "\033[36m◈\033[0m "  # Cyan for python mode

    @property
    def prompt2(self):
        return "\033[2m⋮\033[0m "

    def runsource(self, source, filename="<input>", symbol="single"):
        source = source.rstrip()

        # Toggle mode with backtick
        if source == '`':
            self.ask_mode = not self.ask_mode
            mode = "[magenta]ask[/]" if self.ask_mode else "[cyan]python[/]"
            console.print(f"[dim]switched to {mode} mode[/]")
            return False

        if not source:
            return False

        if self.ask_mode:
            # Send to Claude
            ask_sync(source)
            return False
        else:
            # Normal Python execution
            return super().runsource(source, filename, symbol)

    def interact(self, banner=None, exitmsg=None):
        """Custom interact loop with dynamic prompts."""
        import sys

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
                console.print("\n[dim]KeyboardInterrupt[/]")
                self.resetbuffer()
                more = False

        if exitmsg:
            self.write(f"{exitmsg}\n")


def run_modal_repl(ns: dict):
    """Start the modal REPL with the given namespace."""
    from rich.text import Text

    repl = ModalREPL(locals=ns)
    init(ns, ns)

    # Print banner
    console.print()
    console.print(Panel(
        Text.from_markup('''[bold cyan]`[/]                    [dim]→ toggle between ask/python mode[/]
[bold cyan]clear[/][dim]()[/]             [dim]→ clear conversation history[/]

[bold magenta]load_claude_sessions[/][dim]() → load ~/.claude/projects/*.jsonl[/]
[bold magenta]rlm[/][dim](query, context)[/]    [dim]→ recursive LLM search[/]

[dim]In ask mode, just type naturally. In python mode, write code.[/]'''),
        title='[bold white]mahtab[/]',
        subtitle='[dim]Ctrl+C to cancel • Ctrl+D to exit[/]',
        border_style='bright_black'
    ))
    console.print()

    repl.interact()


# Background task support
import threading
import concurrent.futures

_bg_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
_bg_tasks: list[concurrent.futures.Future] = []


def bg(prompt: str, max_turns: int = 5) -> concurrent.futures.Future:
    """
    Run ask() in background thread. Returns a Future.

    Usage:
        t = bg("do something")  # starts immediately
        # ... do other stuff ...
        t.result()  # wait for result (or t.done() to check)
    """
    def run():
        return asyncio.run(ask(prompt, max_turns))

    future = _bg_executor.submit(run)
    _bg_tasks.append(future)
    console.print(f"[dim]Started background task #{len(_bg_tasks)}[/]")
    return future


def tasks():
    """Show status of background tasks."""
    if not _bg_tasks:
        console.print("[dim]No background tasks.[/]")
        return

    for i, t in enumerate(_bg_tasks, 1):
        if t.done():
            if t.exception():
                console.print(f"  [red]#{i}: error - {t.exception()}[/]")
            else:
                console.print(f"  [green]#{i}: done[/]")
        elif t.running():
            console.print(f"  [yellow]#{i}: running...[/]")
        else:
            console.print(f"  [dim]#{i}: pending[/]")


# Convenience: auto-init if imported interactively
if __name__ != "__main__":
    import inspect
    frame = inspect.currentframe()
    if frame and frame.f_back:
        init(frame.f_back.f_globals, frame.f_back.f_locals)
