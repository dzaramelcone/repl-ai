"""System prompts and prompt templates for the REPL agent."""

from __future__ import annotations

from langchain_core.prompts import SystemMessagePromptTemplate

# Main REPL system prompt template
REPL_SYSTEM_TEMPLATE = """You're in a shared Python REPL with the user. You can see and modify their namespace.

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

{skills_description}

Other:
  load_claude_sessions() -> str    # Load ~/.claude/projects/*.jsonl

When you want to run code, output a fenced python block. The code will execute in the user's namespace and you'll see the output. You can run multiple code blocks in one response.

{repl_context}

When you're done and have a final answer, just respond with text (no code block).

Keep responses concise. Do NOT generate conversation transcripts or include "User:", "A:", "Assistant:" labels - just respond directly."""

REPL_SYSTEM_PROMPT = SystemMessagePromptTemplate.from_template(REPL_SYSTEM_TEMPLATE)


def build_repl_system_prompt(
    var_summary: str = "(empty)",
    skills_description: str = "",
    repl_context: str = "",
    prior_session: str = "",
) -> str:
    """Build the complete system prompt for the REPL agent.

    Args:
        var_summary: Summary of variables in the namespace.
        skills_description: Description of available skills.
        repl_context: Recent REPL activity for context.
        prior_session: Prior session context XML.

    Returns:
        Formatted system prompt string.
    """
    # Format REPL context if present
    formatted_repl_context = ""
    if repl_context:
        formatted_repl_context = f"<repl_activity>\n{repl_context}\n</repl_activity>"

    return REPL_SYSTEM_TEMPLATE.format(
        var_summary=var_summary,
        skills_description=skills_description,
        repl_context=formatted_repl_context,
        prior_session=prior_session,
    )


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
