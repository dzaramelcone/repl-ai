"""LLM integration: Claude CLI wrapper and prompt templates."""

from mahtab.llm.claude_cli import ChatClaudeCLI
from mahtab.llm.prompts import (
    REPL_SYSTEM_TEMPLATE,
    RLM_SYSTEM_TEMPLATE,
    build_repl_system_prompt,
    build_rlm_iteration_prompt,
    build_rlm_system_prompt,
)


def extract_usage(response) -> dict | None:
    """Extract usage dict from LLM response if available.

    Args:
        response: LLMResult or ChatResult from LangChain.

    Returns:
        Usage dict with input_tokens, output_tokens, total_cost_usd, etc.
        Returns None if usage is not available.
    """
    if not response or not hasattr(response, "generations") or not response.generations:
        return None
    gen = response.generations[0]
    if not hasattr(gen, "generation_info") or not gen.generation_info:
        return None
    return gen.generation_info.get("usage")


__all__ = [
    "ChatClaudeCLI",
    "REPL_SYSTEM_TEMPLATE",
    "RLM_SYSTEM_TEMPLATE",
    "build_repl_system_prompt",
    "build_rlm_system_prompt",
    "build_rlm_iteration_prompt",
    "extract_usage",
]
