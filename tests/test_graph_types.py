"""Tests for graph type definitions."""

from mahtab.agent.graph import AgentState, ReflectionResult


def test_reflection_result_complete():
    result = ReflectionResult(
        is_complete=True,
        reasoning="Task accomplished",
        next_action=None,
    )
    assert result.is_complete is True
    assert result.next_action is None


def test_reflection_result_incomplete():
    result = ReflectionResult(
        is_complete=False,
        reasoning="Missing step",
        next_action="Add error handling",
    )
    assert result.is_complete is False
    assert result.next_action == "Add error handling"


def test_agent_state_typing():
    """Verify AgentState has required fields."""
    assert "messages" in AgentState.__annotations__
    assert "original_prompt" in AgentState.__annotations__
    assert "code_blocks" in AgentState.__annotations__
    assert "execution_results" in AgentState.__annotations__
    assert "turn_count" in AgentState.__annotations__
