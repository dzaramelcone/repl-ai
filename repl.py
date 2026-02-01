"""
Shared REPL with Claude.
Claude can execute code in your namespace.
"""

import asyncio
import code as code_module
import json
import re
import readline
import rlcompleter
import sys
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.text import Text

from rlm import rlm

console = Console()

# Will be set to caller's globals/locals
_globals = None
_locals = None
_history = []

# Capture REPL activity between ask() calls
_repl_activity = []
_original_displayhook = None

# Usage tracking (session stats)
_usage_stats = {
    "total_cost_usd": 0.0,
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0,
    "num_calls": 0,
}

# Skills directory
SKILLS_DIR = Path("~/.mahtab/skills").expanduser()
LAST_SESSION_FILE = Path("~/.mahtab/last_session.json").expanduser()


def _save_last_session(user_msg: str, assistant_msg: str):
    """Save last exchange for next session."""
    import json
    from datetime import datetime

    LAST_SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "timestamp": datetime.now().isoformat(),
        "user": user_msg[:2000],  # Truncate to save space
        "assistant": assistant_msg[:2000],
    }
    LAST_SESSION_FILE.write_text(json.dumps(data, indent=2))


def _load_last_session() -> str:
    """Load last session context as XML."""
    if not LAST_SESSION_FILE.exists():
        return ""
    try:
        data = json.loads(LAST_SESSION_FILE.read_text())
        return f"""<prior_session timestamp="{data.get("timestamp", "unknown")}">
<human>{data.get("user", "")}</human>
<assistant>{data.get("assistant", "")}</assistant>
</prior_session>"""
    except Exception:
        return ""


def _capture_displayhook(value):
    """Capture REPL output for context."""
    if value is not None:
        _repl_activity.append(f">>> {repr(value)}")
    # Call original displayhook
    if _original_displayhook:
        _original_displayhook(value)


def _install_repl_capture():
    """Install hooks to capture REPL activity."""
    global _original_displayhook
    if _original_displayhook is None:
        _original_displayhook = sys.displayhook
        sys.displayhook = _capture_displayhook


def record_input(code: str):
    """Record user input to REPL activity. Call this from the REPL."""
    _repl_activity.append(f">>> {code}")


def _get_repl_context(max_chars: int = 4000) -> str:
    """Get recent REPL activity, truncated to max_chars."""
    if not _repl_activity:
        return ""
    text = "\n".join(_repl_activity)
    if len(text) > max_chars:
        text = "...\n" + text[-max_chars:]
    return text


def _clear_repl_activity():
    """Clear REPL activity buffer after ask()."""
    _repl_activity.clear()


def _load_skill_descriptions() -> str:
    """Load skill descriptions from SKILLS_DIR."""
    if not SKILLS_DIR.exists():
        return ""

    descriptions = []
    for skill_file in sorted(SKILLS_DIR.glob("*.md")):
        content = skill_file.read_text()
        name = skill_file.stem

        # Parse YAML frontmatter
        desc = name  # Default description
        if content.startswith("---"):
            try:
                end = content.index("---", 3)
                frontmatter = content[3:end].strip()
                for line in frontmatter.split("\n"):
                    if line.startswith("description:"):
                        desc = line.split(":", 1)[1].strip().strip("\"'")
                        break
            except ValueError:
                pass

        descriptions.append(f"  {name}: {desc}")

    if descriptions:
        return "Skills (use skill(name) to invoke):\n" + "\n".join(descriptions)
    return ""


def skill(name: str, args: str = "") -> str:
    """Load and return a skill's full content."""
    skill_file = SKILLS_DIR / f"{name}.md"
    if not skill_file.exists():
        return f"Error: skill '{name}' not found in {SKILLS_DIR}"

    content = skill_file.read_text()

    # Strip frontmatter
    if content.startswith("---"):
        try:
            end = content.index("---", 3)
            content = content[end + 3 :].strip()
        except ValueError:
            pass

    # Replace $ARGUMENTS placeholder
    content = content.replace("$ARGUMENTS", args)

    return content


# Text exploration tools (from RLM paper)
def peek(text: str, n: int = 2000) -> str:
    """Return first n characters of text."""
    return text[:n]


def grep(text: str, pattern: str) -> list[str]:
    """Return lines matching regex pattern (case-insensitive)."""
    import re

    return [line for line in text.split("\n") if re.search(pattern, line, re.IGNORECASE)]


def partition(text: str, n: int = 10) -> list[str]:
    """Split text into n roughly equal chunks."""
    chunk_size = max(1, len(text) // n)
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def init(g=None, loc=None):
    """Initialize with your namespace. Call as: init(globals(), locals())"""
    global _globals, _locals
    _globals = g if g is not None else {}
    _locals = loc if loc is not None else _globals
    # Install REPL capture hooks
    _install_repl_capture()


def _print_code(code: str, title: str = "Code"):
    """Print syntax-highlighted Python code in a panel."""
    syntax = Syntax(code, "python", theme="monokai", line_numbers=True, indent_guides=True)
    console.print(Panel(syntax, title=f"[bold cyan]{title}[/]", border_style="cyan"))


def _print_output(output: str, is_error: bool = False):
    """Print execution output."""
    style = "red" if is_error else "green"
    title = "Error" if is_error else "Output"
    console.print(Panel(output, title=f"[bold {style}]{title}[/]", border_style=style))


def ask(prompt: str = "", max_turns: int = 5) -> str:
    """
    Ask Claude something. Claude can execute code in your namespace.
    Streams response to stdout. Returns final text response.
    """
    try:
        return asyncio.run(_ask_async(prompt, max_turns))
    except KeyboardInterrupt:
        import sys

        sys.stdout.write("\n\033[33m[cancelled]\033[0m\n")
        sys.stdout.flush()
        return "(cancelled)"
    finally:
        _clear_repl_activity()
        print("\033[0m", end="", flush=True)


async def _ask_async(prompt: str, max_turns: int) -> str:
    """Async implementation of ask()."""
    global _history

    if _globals is None:
        raise RuntimeError("Call init(globals(), locals()) first")

    # Build context about available variables
    var_summary = _summarize_namespace()
    skill_descriptions = _load_skill_descriptions()
    repl_context = _get_repl_context()
    prior_session = _load_last_session()

    system = f"""You're in a shared Python REPL with the user. You can see and modify their namespace.

{prior_session}

To search through past conversations, use:
  sessions = load_claude_sessions()  # Load all ~/.claude/projects/*.jsonl
  grep(sessions, "pattern")          # Find relevant lines

Available variables:
{var_summary}

File tools:
  read(path, start=1, end=None) -> str   # Read file with line numbers
  edit(path, old, new) -> str            # Replace old text with new text in file

Text exploration (for large strings):
  peek(text, n=2000) -> str        # First n chars
  grep(text, pattern) -> list[str] # Lines matching regex
  partition(text, n=10) -> list[str] # Split into n chunks
  rlm(query, context) -> str       # Recursive LLM search

{skill_descriptions}

Other:
  load_claude_sessions() -> str    # Load ~/.claude/projects/*.jsonl

When you want to run code, output a fenced python block. The code will execute in the user's namespace and you'll see the output. You can run multiple code blocks in one response.

{f"<repl_activity>{chr(10)}{repl_context}{chr(10)}</repl_activity>" if repl_context else ""}

When you're done and have a final answer, just respond with text (no code block).

Keep responses concise. Do NOT generate conversation transcripts or include "User:", "A:", "Assistant:" labels - just respond directly."""

    _history.append({"role": "user", "content": prompt})

    for _ in range(max_turns):
        # Call Claude with streaming
        response = await _call_claude_stream(system, _history)

        # Extract code blocks
        code_blocks = re.findall(r"```python\n(.*?)```", response, re.DOTALL)

        if not code_blocks:
            # No code, just a text response - we're done
            _history.append({"role": "assistant", "content": response})
            _save_last_session(prompt, response)
            return response

        # Execute code blocks and collect output
        # (code already shown during streaming, just show execution output)
        outputs = []
        for block in code_blocks:
            block = block.strip()
            output = _exec_code(block)
            outputs.append(output)
            is_error = output.startswith("Error:")
            _print_output(output, is_error)

        # Add assistant response and execution results to history
        _history.append({"role": "assistant", "content": response})

        exec_report = "\n\n".join(f"Code block {i + 1} output:\n{out}" for i, out in enumerate(outputs))
        _history.append({"role": "user", "content": f"<execution>\n{exec_report}\n</execution>"})

    console.print(f"[yellow]⚠ Max turns ({max_turns}) reached. Use ask() again to continue.[/]")
    # Save last exchange before returning
    if len(_history) >= 2:
        last_user = next((m["content"] for m in reversed(_history) if m["role"] == "user"), "")
        last_asst = next((m["content"] for m in reversed(_history) if m["role"] == "assistant"), "")
        _save_last_session(last_user, last_asst)
    return "(max turns reached)"


def _summarize_namespace() -> str:
    """Summarize variables in the namespace."""
    if not _globals:
        return "(empty)"

    lines = []
    for name, val in _globals.items():
        if name.startswith("_"):
            continue
        try:
            typ = type(val).__name__
            if isinstance(val, int | float | str | bool | type(None)):
                rep = repr(val)[:50]
            elif isinstance(val, list | dict | set | tuple):
                rep = f"{typ} with {len(val)} items"
            else:
                rep = typ
            lines.append(f"  {name}: {rep}")
        except Exception:
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

    # Build conversation as structured XML
    prompt_parts = ["<conversation>"]
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        tag = "human" if role == "user" else "assistant"
        prompt_parts.append(f"<{tag}>{content}</{tag}>")
    prompt_parts.append("</conversation>")

    full_prompt = "\n".join(prompt_parts)

    proc = await asyncio.create_subprocess_exec(
        "claude",
        "-p",
        full_prompt,
        "--system-prompt",
        system,
        "--setting-sources",
        "",
        "--output-format",
        "stream-json",
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
            border_style="cyan" if done else "dim",
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
                                        code_live.update(_make_code_panel(code_buffer.rstrip("`")))
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
                    # Track usage stats
                    _usage_stats["num_calls"] += 1
                    _usage_stats["total_cost_usd"] += data.get("total_cost_usd", 0)
                    usage_data = data.get("usage", {})
                    _usage_stats["input_tokens"] += usage_data.get("input_tokens", 0)
                    _usage_stats["output_tokens"] += usage_data.get("output_tokens", 0)
                    _usage_stats["cache_read_input_tokens"] += usage_data.get("cache_read_input_tokens", 0)
                    _usage_stats["cache_creation_input_tokens"] += usage_data.get("cache_creation_input_tokens", 0)

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
    import importlib
    import sys
    from pathlib import Path

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
    reloaded = False
    if path.suffix == ".py":
        # Find module name from path
        for _name, mod in list(sys.modules.items()):
            if mod is None:
                continue
            mod_file = getattr(mod, "__file__", None)
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

    lines = path.read_text().split("\n")

    if end is None:
        end = len(lines)

    start = max(1, start)
    end = min(len(lines), end)

    result = []
    for i in range(start - 1, end):
        result.append(f"{i + 1:4d}│ {lines[i]}")

    return "\n".join(result)


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
        tok_str = f"{tokens / 1_000_000:.1f}Mt"
    elif tokens >= 1_000:
        tok_str = f"{tokens / 1_000:.1f}kt"
    else:
        tok_str = f"{tokens}t"

    console.print(f"[dim]Loaded[/] [bold cyan]{tok_str}[/] [dim]from {len(chunks)} files[/]")
    return result


# Modal REPL
class ModalREPL(code_module.InteractiveConsole):
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
        if source == "`":
            self.ask_mode = not self.ask_mode
            mode = "[magenta]ask[/]" if self.ask_mode else "[cyan]python[/]"
            console.print(f"[dim]switched to {mode} mode[/]")
            return False

        if not source:
            return False

        if self.ask_mode:
            # Send to Claude
            ask(source)
            return False
        else:
            # Normal Python execution
            return super().runsource(source, filename, symbol)

    def interact(self, banner=None, exitmsg=None):
        """Custom interact loop with dynamic prompts."""

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
    console.print(
        Panel(
            Text.from_markup("""[bold cyan]`[/]                    [dim]→ toggle between ask/python mode[/]
[bold cyan]clear[/][dim]()[/]             [dim]→ clear conversation history[/]

[bold magenta]load_claude_sessions[/][dim]() → load ~/.claude/projects/*.jsonl[/]
[bold magenta]rlm[/][dim](query, context)[/]    [dim]→ recursive LLM search[/]

[dim]In ask mode, just type naturally. In python mode, write code.[/]"""),
            title="[bold white]mahtab[/]",
            subtitle="[dim]Ctrl+C to cancel • Ctrl+D to exit[/]",
            border_style="bright_black",
        )
    )
    console.print()

    repl.interact()


def usage():
    """Show cumulative usage stats for this session."""
    s = _usage_stats
    console.print(
        Panel(
            f"""[bold]Session Usage Stats[/]

Calls:          {s["num_calls"]}
Total Cost:     ${s["total_cost_usd"]:.4f}

[dim]Tokens:[/]
  Input:        {s["input_tokens"]:,}
  Output:       {s["output_tokens"]:,}
  Cache Read:   {s["cache_read_input_tokens"]:,}
  Cache Create: {s["cache_creation_input_tokens"]:,}""",
            title="[cyan]usage()[/]",
            border_style="dim",
        )
    )


# Convenience: auto-init if imported interactively
if __name__ != "__main__":
    import inspect

    frame = inspect.currentframe()
    if frame and frame.f_back:
        init(frame.f_back.f_globals, frame.f_back.f_locals)


# Ensure skills directory exists
SKILLS_DIR.mkdir(parents=True, exist_ok=True)

# Track readline history position to capture input
_last_history_len = readline.get_current_history_length()

# ANSI codes
NUM = "\033[38;5;117m"  # bright blue-ish
DIM = "\033[38;5;242m"  # grey
RESET = "\033[0m"
CYAN = "\033[36m"


def _approx_tokens(text):
    """Rough token estimate: ~4 chars per token"""
    return len(text) // 4


def _format_tokens(n):
    if n >= 1_000_000_000:
        return f"{NUM}{n / 1_000_000_000:.1f}{DIM}Gt"
    elif n >= 1_000_000:
        return f"{NUM}{n / 1_000_000:.1f}{DIM}Mt"
    elif n >= 1_000:
        return f"{NUM}{n / 1_000:.1f}{DIM}kt"
    return f"{NUM}{n}{DIM}t"


class DynamicPrompt:
    def __str__(self):
        global _last_history_len

        # Capture any new readline history entries (user input)
        current_len = readline.get_current_history_length()
        while _last_history_len < current_len:
            _last_history_len += 1
            item = readline.get_history_item(_last_history_len)
            if item and not item.startswith("ask("):
                record_input(item)

        parts = []

        # Memory
        try:
            import resource

            mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
            parts.append(f"{NUM}{mem_mb:.0f}{DIM}MB")
        except Exception:
            pass

        # Context size (if 'context' var exists)
        ctx = ns.get("context")
        if ctx and isinstance(ctx, str) and len(ctx) > 0:
            toks = _approx_tokens(ctx)
            parts.append(f"{DIM}ctx:{_format_tokens(toks)}")

        # History size
        if _history:
            hist_chars = sum(len(m.get("content", "")) for m in _history)
            hist_toks = hist_chars // 4  # ~4 chars per token
            parts.append(f"{DIM}hist:{_format_tokens(hist_toks)}")

        info = " ".join(parts)
        return f"{DIM}{info}{RESET} {CYAN}◈{RESET} " if info else f"{CYAN}◈{RESET} "


sys.ps1 = DynamicPrompt()
sys.ps2 = "\033[2m⋮\033[0m "

ns = globals()
ns.update(
    {
        "ask": ask,
        "clear": clear,
        "re": re,
        "rlm": rlm,
        "load_claude_sessions": load_claude_sessions,
        "edit": edit,
        "read": read,
        "skill": skill,
        "peek": peek,
        "grep": grep,
        "partition": partition,
        "usage": usage,
    }
)
init(ns, ns)

console.print()
console.print(
    Panel(
        Text.from_markup("""[bold cyan]ask[/][dim](\"prompt\")[/]        [dim]→ ask Claude (streams response)[/]
[bold cyan]clear[/][dim]()[/]             [dim]→ clear conversation history[/]

[bold yellow]read[/][dim](path)[/]           [dim]→ read file with line numbers[/]
[bold yellow]edit[/][dim](path, old, new)[/] [dim]→ replace text in file[/]
[bold yellow]skill[/][dim](name)[/]          [dim]→ invoke a skill from ~/.mahtab/skills/[/]

[bold green]peek[/][dim](text, n)[/]        [dim]→ first n chars of text[/]
[bold green]grep[/][dim](text, pattern)[/]  [dim]→ lines matching regex[/]
[bold green]partition[/][dim](text, n)[/]   [dim]→ split text into n chunks[/]
[bold green]rlm[/][dim](query, context)[/]  [dim]→ recursive LLM search[/]"""),
        title="[bold white]mahtab[/]",
        subtitle="[dim]Ctrl+C to cancel[/]",
        border_style="bright_black",
    )
)
console.print()
