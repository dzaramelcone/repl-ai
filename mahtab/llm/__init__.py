"""LLM integration: Claude CLI wrapper and prompt templates."""

from mahtab.llm.claude_cli import ChatClaudeCLI
from mahtab.llm.prompts import (
    REPL_SYSTEM_TEMPLATE,
    RLM_SYSTEM_TEMPLATE,
    build_repl_system_prompt,
    build_rlm_iteration_prompt,
    build_rlm_system_prompt,
)


def extract_usage(response) -> dict:
    """Extract usage dict from LLM response.

    Args:
        response: LLMResult or ChatResult from LangChain.

    Returns:
        Usage dict with input_tokens, output_tokens, total_cost_usd, etc.
    """
    gen = response.generations[0][0]
    return gen.generation_info["usage"]


__all__ = [
    "ChatClaudeCLI",
    "REPL_SYSTEM_TEMPLATE",
    "RLM_SYSTEM_TEMPLATE",
    "build_repl_system_prompt",
    "build_rlm_system_prompt",
    "build_rlm_iteration_prompt",
    "extract_usage",
]
