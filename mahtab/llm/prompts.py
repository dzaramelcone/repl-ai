"""System prompts and prompt templates for the REPL agent."""

from __future__ import annotations

# Main REPL system prompt template
REPL_SYSTEM_TEMPLATE = """You're in a shared Python REPL with the user. You can see and modify their namespace.

Available variables:
{var_summary}

When you want to run code, output a fenced python block. The code will execute in the user's namespace and you'll see the output. You can run multiple code blocks in one response.

IMPORTANT: Code and execution output are displayed in a separate REPL pane. Do NOT repeat or echo execution output in your text responses - the user already sees it. Just provide brief commentary if needed.

Keep responses concise."""


def build_repl_system_prompt(
    var_summary: str = "(empty)",
) -> str:
    """Build the complete system prompt for the REPL agent.

    Args:
        var_summary: Summary of variables in the namespace.

    Returns:
        Formatted system prompt string.
    """
    return REPL_SYSTEM_TEMPLATE.format(var_summary=var_summary)


# RLM (Recursive Language Model) system prompt
RLM_SYSTEM_TEMPLATE = """You explore data by writing Python code.

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


def build_rlm_system_prompt(context_size: int) -> str:
    """Build the system prompt for RLM with context size.

    Args:
        context_size: Size of the context in characters.

    Returns:
        Formatted system prompt string.
    """
    return RLM_SYSTEM_TEMPLATE.format(size=f"{context_size:,}")


# Prompt for building RLM iteration query
RLM_ITERATION_TEMPLATE = """Query: {query}
Context size: {context_size:,} chars
Depth: {depth}

{history}

Write Python code:"""


def build_rlm_iteration_prompt(
    query: str,
    context_size: int,
    depth: int,
    history: str = "",
) -> str:
    """Build the prompt for an RLM iteration.

    Args:
        query: The user's query.
        context_size: Size of the context in characters.
        depth: Current recursion depth.
        history: Previous iteration history.

    Returns:
        Formatted prompt string.
    """
    history_str = f"History:{history}" if history else "(first iteration)"
    return RLM_ITERATION_TEMPLATE.format(
        query=query,
        context_size=context_size,
        depth=depth,
        history=history_str,
    )


# Reflection prompt for evaluating code execution
REFLECTION_PROMPT_TEMPLATE = """Evaluate whether the code execution satisfied the user's request.

## Original Request
{original_prompt}

## Code Executed
{code_blocks}

## Execution Output
{execution_results}

## Instructions
Evaluate:
1. CORRECTNESS: Did the code run without errors? If there were errors, are they blocking the task?
2. COMPLETENESS: Does the output satisfy what the user asked for?

Respond with ONLY valid JSON (no markdown, no explanation):
{{"is_complete": true/false, "reasoning": "brief explanation", "next_action": "what to do next" or null}}"""


def build_reflection_prompt(
    original_prompt: str,
    code_blocks: list[str],
    execution_results: list[tuple[str, bool]],
) -> str:
    """Build the prompt for reflection evaluation.

    Args:
        original_prompt: The user's original request.
        code_blocks: List of code blocks that were executed.
        execution_results: List of (output, is_error) tuples.

    Returns:
        Formatted reflection prompt string.
    """
    # Format code blocks
    code_str = "\n\n".join(f"```python\n{block}\n```" for block in code_blocks)

    # Format execution results with error markers
    results_parts = []
    for i, (output, is_error) in enumerate(execution_results, 1):
        status = "[ERROR]" if is_error else "[OK]"
        results_parts.append(f"Block {i} {status}:\n{output}")
    results_str = "\n\n".join(results_parts)

    return REFLECTION_PROMPT_TEMPLATE.format(
        original_prompt=original_prompt,
        code_blocks=code_str,
        execution_results=results_str,
    )
