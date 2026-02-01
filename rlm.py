"""
Minimal RLM - Recursive Language Model

The LLM writes code to explore context. That's it.
"""

import re
import traceback

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax

from auth import messages_create

console = Console()

SYSTEM = """You explore data by writing Python code.

You have access to:
  context: str           # The full data (~{size} chars). DO NOT print it all.

Tools (these are functions you can call):
  peek(n=2000) -> str              # First n chars of context
  grep(pattern) -> list[str]       # Lines matching regex pattern
  partition(n=10) -> list[str]     # Split context into n chunks
  rlm(query, subset) -> str        # Recursively explore a subset

Termination:
  FINAL(answer)                    # Return answer and stop

Strategy:
1. peek() to understand structure
2. grep() to find relevant sections
3. partition() + rlm() for large sections
4. FINAL() when you have the answer

Write ONLY Python code. No markdown. No explanation. No print() unless debugging."""


def rlm(query: str, context: str, depth: int = 0, max_iters: int = 10) -> str:
    """
    Recursive Language Model.
    LLM generates code to explore context via peek/grep/partition/rlm.
    """
    history = ""
    output_buffer = []

    # Tools available to the LLM's code
    def peek(n: int = 2000) -> str:
        return context[:n]

    def grep(pattern: str) -> list[str]:
        return [line for line in context.split("\n") if re.search(pattern, line, re.IGNORECASE)]

    def partition(n: int = 10) -> list[str]:
        chunk_size = len(context) // n
        return [context[i : i + chunk_size] for i in range(0, len(context), chunk_size)]

    def recurse(q: str, subset: str) -> str:
        if depth >= 3:
            return f"[max depth reached, subset is {len(subset)} chars]"
        return rlm(q, subset, depth=depth + 1, max_iters=max_iters)

    result = {"_final": None}

    def FINAL(answer):
        result["_final"] = str(answer)
        return answer

    output_size = [0]  # Use list for mutability in closure
    OUTPUT_LIMIT = 10000

    def capture_print(*args, **kwargs):
        text = " ".join(str(a) for a in args)
        output_size[0] += len(text)
        if output_size[0] > OUTPUT_LIMIT:
            raise RuntimeError(f"Output too large (>{OUTPUT_LIMIT} chars). Use slicing or summarize.")
        output_buffer.append(text)

    for i in range(max_iters):
        prompt = f"""Query: {query}
Context size: {len(context):,} chars
Depth: {depth}

{f"History:{history}" if history else "(first iteration)"}

Write Python code:"""

        console.print(f"\n[dim][depth={depth} iter={i + 1}][/]")

        # Stream code into a live-updating panel
        code_buffer: list[str] = []

        def make_panel(code_str: str, done: bool = False) -> Panel:
            title = "[cyan]Generated Code[/]" if done else "[dim cyan]Generating...[/]"
            return Panel(
                Syntax(code_str or " ", "python", theme="monokai", line_numbers=True),
                title=title,
                border_style="cyan" if done else "dim",
            )

        with Live(make_panel(""), console=console, refresh_per_second=15) as live:

            def on_token(token: str, buf: list[str] = code_buffer, lv: Live = live):
                buf.append(token)
                lv.update(make_panel("".join(buf)))

            response = messages_create(
                model="claude-opus-4-20250514",
                max_tokens=2000,
                system=SYSTEM.format(size=f"{len(context):,}"),
                messages=[{"role": "user", "content": prompt}],
                on_token=on_token,
            )
            # Final update with "done" styling
            code = response["content"][0]["text"].strip()
            code = re.sub(r"^```python\n?", "", code)
            code = re.sub(r"\n?```$", "", code)
            live.update(make_panel(code, done=True))

        output_buffer.clear()

        local_vars = {
            "context": context,
            "peek": peek,
            "grep": grep,
            "partition": partition,
            "rlm": recurse,
            "FINAL": FINAL,
            "print": capture_print,
            "re": re,
            "len": len,
        }

        try:
            exec(code, {"__builtins__": {}}, local_vars)
        except SyntaxError as e:
            # Syntax errors have line/offset info
            error_msg = f"SyntaxError at line {e.lineno}: {e.msg}"
            if e.text:
                error_msg += f"\n  {e.text.strip()}"
                if e.offset:
                    error_msg += f"\n  {' ' * (e.offset - 1)}^"
            console.print(Panel(error_msg, title="[red]Syntax Error[/]", border_style="red"))
            history += f"\n---\nCode:\n{code}\n\nError: {error_msg}\nFix the syntax error and try again.\n"
            continue
        except NameError as e:
            error_msg = f"NameError: {e}\n\nAvailable names: context, peek, grep, partition, rlm, FINAL, print, re, len"
            console.print(Panel(error_msg, title="[red]Name Error[/]", border_style="red"))
            history += f"\n---\nCode:\n{code}\n\nError: {error_msg}\nUse only the available functions listed above.\n"
            continue
        except Exception as e:
            # Get the traceback for the exec'd code
            tb = traceback.format_exc()
            # Find the line in the generated code that caused the error
            error_lines = []
            for line in tb.split("\n"):
                if "<string>" in line or "exec(" not in line:
                    error_lines.append(line)
            error_msg = f"{type(e).__name__}: {e}\n\n" + "\n".join(error_lines[-5:])
            console.print(Panel(error_msg, title="[red]Runtime Error[/]", border_style="red"))
            history += f"\n---\nCode:\n{code}\n\nError: {error_msg}\nFix the error and try again.\n"
            continue

        if result["_final"] is not None:
            console.print(
                Panel(
                    result["_final"][:500] + ("..." if len(result["_final"]) > 500 else ""),
                    title=f"[green]FINAL (depth={depth})[/]",
                    border_style="green",
                )
            )
            return result["_final"]

        output = "\n".join(output_buffer) if output_buffer else "(no output)"
        console.print(Panel(output[:1000], title="[cyan]Output[/]", border_style="dim"))
        history += f"\n---\nCode:\n{code}\nOutput:\n{output}\n"

    console.print(f"[yellow][depth={depth}] Max iterations reached[/]")
    return f"[depth={depth}] Max iterations reached"
