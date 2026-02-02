"""Tests for graph type definitions."""

from mahtab.agent.graph import AgentState, ReflectionResult


def test_reflection_result_complete():
    result = ReflectionResult(is_complete=True, reasoning="Task accomplished")
    assert result.is_complete is True
    assert result.reasoning == "Task accomplished"


def test_reflection_result_incomplete():
    result = ReflectionResult(is_complete=False, reasoning="Missing step")
    assert result.is_complete is False
    assert result.reasoning == "Missing step"


def test_agent_state_typing():
    """Verify AgentState has required fields."""
    assert "messages" in AgentState.__annotations__
    assert "original_prompt" in AgentState.__annotations__
    assert "code_blocks" in AgentState.__annotations__
    assert "execution_results" in AgentState.__annotations__
    assert "turn_count" in AgentState.__annotations__
