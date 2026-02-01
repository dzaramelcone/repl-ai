"""LangGraph-based agent implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from pydantic import BaseModel

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

    from mahtab.core.state import SessionState


class ReflectionResult(BaseModel):
    """Result of the reflection node's evaluation."""

    is_complete: bool
    reasoning: str
    next_action: str | None = None


class AgentState(TypedDict, total=False):
    """State that flows through the agent graph.

    Attributes:
        messages: Conversation history (LangChain messages).
        system_prompt: The system prompt built at start.
        original_prompt: User's initial request for reflection.
        current_response: Claude's latest response text.
        code_blocks: Extracted Python code blocks.
        execution_results: List of (output, is_error) tuples.
        turn_count: Number of generate cycles completed.
        session: The SessionState for namespace and persistence.
        reflection: Result of the last reflection (if any).
    """

    messages: list[BaseMessage]
    system_prompt: str
    original_prompt: str
    current_response: str
    code_blocks: list[str]
    execution_results: list[tuple[str, bool]]
    turn_count: int
    session: SessionState
    reflection: ReflectionResult | None
