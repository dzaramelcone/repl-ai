"""LangGraph agent state definitions."""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(dict):
    """State for the REPL agent graph.

    This state is passed between nodes in the LangGraph and contains:
    - messages: Conversation history with automatic message merging
    - loaded_skills: List of skills that have been loaded in this session
    - namespace: Reference to the shared Python namespace
    - pending_code: Code blocks awaiting execution (if any)

    Using dict subclass for LangGraph compatibility while providing
    type hints for IDE support.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    loaded_skills: list[str]
    namespace: dict[str, Any]
    pending_code: str | None


# Type alias for use in type hints
AgentStateDict = dict[str, Any]


def create_initial_state(
    messages: list[BaseMessage] | None = None,
    namespace: dict[str, Any] | None = None,
) -> AgentStateDict:
    """Create an initial agent state.

    Args:
        messages: Initial messages. Defaults to empty list.
        namespace: Python namespace dict. Defaults to empty dict.

    Returns:
        Initial state dictionary for the agent graph.
    """
    return {
        "messages": messages or [],
        "loaded_skills": [],
        "namespace": namespace or {},
        "pending_code": None,
    }
