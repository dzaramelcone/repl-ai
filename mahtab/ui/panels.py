"""Panel rendering utilities for code and output display."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text


def print_code_panel(
    code: str,
    title: str,
    console: Console,
    done: bool,
) -> None:
    """Print syntax-highlighted Python code in a panel.

    Args:
        code: The Python code to display.
        title: Panel title.
        console: Console to use. Defaults to global console.
        done: Whether the code is complete (affects styling).
    """
    syntax = Syntax(
        code or " ",
        "python",
        theme="monokai",
        line_numbers=True,
        indent_guides=True,
    )

    if done:
        panel = Panel(
            syntax,
            title=f"[bold cyan]{title}[/]",
            border_style="cyan",
        )
    else:
        panel = Panel(
            syntax,
            title=f"[dim cyan]{title}[/]",
            border_style="dim",
        )

    console.print(panel)


def print_output_panel(
    output: str,
    is_error: bool,
    title: str,
    console: Console,
) -> None:
    """Print execution output in a panel.

    Args:
        output: The output text to display.
        is_error: Whether this is an error output.
        title: Panel title. Empty string defaults to "Error" or "Output".
        console: Console to use. Defaults to global console.
    """
    style = "red" if is_error else "green"
    if not title:
        title = "Error" if is_error else "Output"

    console.print(
        Panel(
            output,
            title=f"[bold {style}]{title}[/]",
            border_style=style,
        )
    )


def print_final_panel(
    result: str,
    depth: int,
    max_chars: int,
    console: Console,
) -> None:
    """Print a final result panel (used in RLM).

    Args:
        result: The result to display.
        depth: Current recursion depth.
        max_chars: Maximum characters to show.
        console: Console to use. Defaults to global console.
    """
    truncated = result[:max_chars]
    if len(result) > max_chars:
        truncated += "..."

    console.print(
        Panel(
            truncated,
            title=f"[green]FINAL (depth={depth})[/]",
            border_style="green",
        )
    )


def print_banner(console: Console) -> None:
    """Print the REPL welcome banner.

    Args:
        console: Console to use. Defaults to global console.
    """
    console.print()
    console.print(
        Panel(
            Text.from_markup(
                """[bold cyan]ask[/][dim]("prompt")[/]       [dim]→ ask Claude (streams response)[/]
[bold cyan]ed[/][dim]()[/]                [dim]→ edit in $EDITOR, return text (try: ask(ed()))[/]
[bold cyan]clear[/][dim]()[/]             [dim]→ clear conversation history[/]
[bold cyan]usage[/][dim]()[/]             [dim]→ show session costs[/]"""
            ),
            title="[bold white]your tools[/]",
            subtitle="[dim]Ctrl+C to cancel[/]",
            border_style="cyan",
        )
    )
    console.print(
        Panel(
            Text.from_markup(
                """[bold yellow]create[/][dim](name)[/]         [dim]→ create new module (e.g. "utils" or "foo.bar")[/]
[bold yellow]read[/][dim](path)[/]           [dim]→ read file with line numbers[/]
[bold yellow]edit[/][dim](path, old, new)[/] [dim]→ replace text in file[/]
[bold yellow]skill[/][dim](name)[/]          [dim]→ invoke a skill from ~/.mahtab/skills/[/]

[bold green]peek[/][dim](text, n)[/]        [dim]→ first n chars of text[/]
[bold green]grep[/][dim](text, pattern)[/]  [dim]→ lines matching regex[/]
[bold green]partition[/][dim](text, n)[/]   [dim]→ split text into n chunks[/]
[bold green]rlm[/][dim](query, context)[/]  [dim]→ recursive LLM search[/]"""
            ),
            title="[dim]mahtab's tools[/]",
            border_style="dim",
        )
    )
    console.print()


def print_modal_banner(console: Console) -> None:
    """Print the modal REPL banner.

    Args:
        console: Console to use. Defaults to global console.
    """
    console.print()
    console.print(
        Panel(
            Text.from_markup(
                """[bold cyan]`[/]                    [dim]→ toggle between ask/python mode[/]
[bold cyan]clear[/][dim]()[/]             [dim]→ clear conversation history[/]

[bold magenta]load_claude_sessions[/][dim]() → load ~/.claude/projects/*.jsonl[/]
[bold magenta]rlm[/][dim](query, context)[/]    [dim]→ recursive LLM search[/]

[dim]In ask mode, just type naturally. In python mode, write code.[/]"""
            ),
            title="[bold white]mahtab[/]",
            subtitle="[dim]Ctrl+C to cancel • Ctrl+D to exit[/]",
            border_style="bright_black",
        )
    )
    console.print()


def print_usage_panel(usage_stats: dict, console: Console) -> None:
    """Print usage statistics panel.

    Args:
        usage_stats: Dictionary of usage stats from UsageStats.model_dump().
        console: Console to use. Defaults to global console.
    """
    s = usage_stats
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
