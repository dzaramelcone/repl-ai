"""Tests for LangGraph agent."""

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
    # TypedDict doesn't enforce at runtime, just check keys exist
    assert "messages" in AgentState.__annotations__
    assert "original_prompt" in AgentState.__annotations__
    assert "code_blocks" in AgentState.__annotations__
    assert "execution_results" in AgentState.__annotations__
    assert "turn_count" in AgentState.__annotations__


def test_extract_code_node_single_block():
    from mahtab.agent.graph import extract_code_node

    state: AgentState = {
        "current_response": "Here's the code:\n```python\nprint('hello')\n```\nDone.",
        "code_blocks": [],
    }
    result = extract_code_node(state)
    assert result["code_blocks"] == ["print('hello')"]


def test_extract_code_node_multiple_blocks():
    from mahtab.agent.graph import extract_code_node

    state: AgentState = {
        "current_response": "```python\nx = 1\n```\nThen:\n```python\ny = 2\n```",
        "code_blocks": [],
    }
    result = extract_code_node(state)
    assert result["code_blocks"] == ["x = 1", "y = 2"]


def test_extract_code_node_no_blocks():
    from mahtab.agent.graph import extract_code_node

    state: AgentState = {
        "current_response": "No code here, just text.",
        "code_blocks": [],
    }
    result = extract_code_node(state)
    assert result["code_blocks"] == []
