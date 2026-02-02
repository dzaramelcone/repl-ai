"""Tests for graph routing and building."""

from unittest.mock import MagicMock

from mahtab.agent.graph import AgentState, ReflectionResult


def test_should_execute_with_code():
    from mahtab.agent.graph import should_execute

    state: AgentState = {"code_blocks": ["x = 1"]}
    assert should_execute(state) == "execute"


def test_should_execute_no_code():
    from mahtab.agent.graph import should_execute

    state: AgentState = {"code_blocks": []}
    assert should_execute(state) == "end"


def test_should_continue_complete():
    from mahtab.agent.graph import should_continue

    state: AgentState = {
        "reflection": ReflectionResult(is_complete=True, reasoning="Done"),
        "turn_count": 1,
    }
    assert should_continue(state, max_turns=5) == "end"


def test_should_continue_incomplete_under_limit():
    from mahtab.agent.graph import should_continue

    state: AgentState = {
        "reflection": ReflectionResult(is_complete=False, reasoning="Need more"),
        "turn_count": 2,
    }
    assert should_continue(state, max_turns=5) == "generate"


def test_should_continue_incomplete_at_limit():
    from mahtab.agent.graph import should_continue

    state: AgentState = {
        "reflection": ReflectionResult(is_complete=False, reasoning="Need more"),
        "turn_count": 5,
    }
    assert should_continue(state, max_turns=5) == "end"


def test_build_agent_graph_creates_graph():
    from mahtab.agent.graph import build_agent_graph

    mock_llm = MagicMock()
    graph = build_agent_graph(llm=mock_llm, max_turns=3)

    assert hasattr(graph, "invoke")
    assert hasattr(graph, "ainvoke")
