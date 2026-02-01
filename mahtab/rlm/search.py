"""Recursive Language Model (RLM) search algorithm.

The LLM writes code to explore large contexts by using peek/grep/partition/rlm tools.
"""

from __future__ import annotations

import re
import traceback
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax

from mahtab.llm.claude_cli import ChatClaudeCLI
from mahtab.llm.prompts import build_rlm_iteration_prompt, build_rlm_system_prompt
from mahtab.tools.text import grep_raw, partition_raw, peek_raw
from mahtab.ui.console import console as default_console


def rlm(
    query: str,
    context: str,
    depth: int = 0,
    max_iters: int = 10,
    max_depth: int = 3,
    model: str = "claude-opus-4-20250514",
    console: Console | None = None,
) -> str:
    """Recursive Language Model search.

    LLM generates code to explore context via peek/grep/partition/rlm.
    The generated code runs in a sandboxed environment with limited builtins.

    Args:
        query: The search query.
        context: The text context to search.
        depth: Current recursion depth.
        max_iters: Maximum iterations per depth level.
        max_depth: Maximum recursion depth.
        model: Claude model to use.
        console: Rich console for output.

    Returns:
        The answer found by the LLM, or error message.
    """
    if console is None:
        console = default_console

    llm = ChatClaudeCLI(model=model)
    history = ""
    output_buffer: list[str] = []

    # Tools available to the LLM's code
    def peek(n: int = 2000) -> str:
        return peek_raw(context, n)

    def grep(pattern: str) -> list[str]:
        return grep_raw(context, pattern)

    def partition(n: int = 10) -> list[str]:
        return partition_raw(context, n)

    def recurse(q: str, subset: str) -> str:
        if depth >= max_depth:
            return f"[max depth reached, subset is {len(subset)} chars]"
        return rlm(q, subset, depth=depth + 1, max_iters=max_iters, max_depth=max_depth, model=model, console=console)

    result: dict[str, Any] = {"_final": None}

    def FINAL(answer: Any) -> Any:
        result["_final"] = str(answer)
        return answer

    output_size = 0
    OUTPUT_LIMIT = 10000

    def capture_print(*args: Any, **_: Any) -> None:
        nonlocal output_size
        text = " ".join(str(a) for a in args)
        output_size += len(text)
        if output_size > OUTPUT_LIMIT:
            raise RuntimeError(f"Output too large (>{OUTPUT_LIMIT} chars). Use slicing or summarize.")
        output_buffer.append(text)

    def make_panel(code_str: str, done: bool = False) -> Panel:
        title = "[cyan]Generated Code[/]" if done else "[dim cyan]Generating...[/]"
        return Panel(
            Syntax(code_str or " ", "python", theme="monokai", line_numbers=True),
            title=title,
            border_style="cyan" if done else "dim",
        )

    system = build_rlm_system_prompt(len(context))

    for i in range(max_iters):
        prompt = build_rlm_iteration_prompt(
            query=query,
            context_size=len(context),
            depth=depth,
            history=history,
        )

        console.print(f"\n[dim][depth={depth} iter={i + 1}][/]")

        # Stream code into a live-updating panel
        code_buffer: list[str] = []

        with Live(make_panel(""), console=console, refresh_per_second=15) as live:
            # Call LLM synchronously with streaming
            from langchain_core.messages import HumanMessage, SystemMessage

            messages = [
                SystemMessage(content=system),
                HumanMessage(content=prompt),
            ]

            full_code = ""
            for chunk in llm.stream(messages):
                token = chunk.content
                if token:
                    full_code += token
                    code_buffer.append(token)
                    live.update(make_panel("".join(code_buffer)))

            # Clean up code
            code = full_code.strip()
            code = re.sub(r"^```python\n?", "", code)
            code = re.sub(r"\n?```$", "", code)
            live.update(make_panel(code, done=True))

        output_buffer.clear()
        output_size = 0

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
            tb = traceback.format_exc()
            error_lines = []
            for line in tb.split("\n"):
                if "<string>" in line or "exec(" not in line:
                    error_lines.append(line)
            error_msg = f"{type(e).__name__}: {e}\n\n" + "\n".join(error_lines[-5:])
            console.print(Panel(error_msg, title="[red]Runtime Error[/]", border_style="red"))
            history += f"\n---\nCode:\n{code}\n\nError: {error_msg}\nFix the error and try again.\n"
            continue

        if result["_final"] is not None:
            final_result = result["_final"]
            console.print(
                Panel(
                    final_result[:500] + ("..." if len(final_result) > 500 else ""),
                    title=f"[green]FINAL (depth={depth})[/]",
                    border_style="green",
                )
            )
            return final_result

        output = "\n".join(output_buffer) if output_buffer else "(no output)"
        console.print(Panel(output[:1000], title="[cyan]Output[/]", border_style="dim"))
        history += f"\n---\nCode:\n{code}\nOutput:\n{output}\n"

    console.print(f"[yellow][depth={depth}] Max iterations reached[/]")
    return f"[depth={depth}] Max iterations reached"
