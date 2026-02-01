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


def test_execute_node_success():
    from mahtab.agent.graph import execute_node
    from mahtab.core.state import SessionState

    session = SessionState()
    state: AgentState = {
        "code_blocks": ["x = 42", "print(x)"],
        "execution_results": [],
        "session": session,
    }
    result = execute_node(state)
    assert len(result["execution_results"]) == 2
    assert result["execution_results"][1][0] == "42\n"  # print output
    assert result["execution_results"][1][1] is False  # no error


def test_execute_node_error():
    from mahtab.agent.graph import execute_node
    from mahtab.core.state import SessionState

    session = SessionState()
    state: AgentState = {
        "code_blocks": ["1/0"],
        "execution_results": [],
        "session": session,
    }
    result = execute_node(state)
    assert len(result["execution_results"]) == 1
    assert "division by zero" in result["execution_results"][0][0].lower()
    assert result["execution_results"][0][1] is True  # is error


def test_execute_node_updates_namespace():
    from mahtab.agent.graph import execute_node
    from mahtab.core.state import SessionState

    session = SessionState()
    state: AgentState = {
        "code_blocks": ["my_var = 'hello'"],
        "execution_results": [],
        "session": session,
    }
    execute_node(state)
    # exec(code, globals, locals) puts assignments in locals_ns
    assert session.locals_ns.get("my_var") == "hello"


def test_reflect_node_parses_complete():
    from mahtab.agent.graph import _parse_reflection_response

    response = '{"is_complete": true, "reasoning": "Task done", "next_action": null}'
    result = _parse_reflection_response(response)
    assert result.is_complete is True
    assert result.reasoning == "Task done"


def test_reflect_node_parses_incomplete():
    from mahtab.agent.graph import _parse_reflection_response

    response = '{"is_complete": false, "reasoning": "Need more", "next_action": "Add validation"}'
    result = _parse_reflection_response(response)
    assert result.is_complete is False
    assert result.next_action == "Add validation"


def test_reflect_node_handles_malformed_json():
    from mahtab.agent.graph import _parse_reflection_response

    response = "This is not JSON at all"
    result = _parse_reflection_response(response)
    # Should default to incomplete on parse failure
    assert result.is_complete is False
    assert "parse" in result.reasoning.lower() or "invalid" in result.reasoning.lower()
