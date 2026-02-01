"""Integration tests for the agent graph."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from mahtab.agent.graph import AgentState, build_agent_graph
from mahtab.core.state import SessionState


class MockLLM:
    """Mock LLM that returns predefined responses."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0

    async def ainvoke(self, _messages, **_kwargs):
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return AIMessage(content=response)


@pytest.mark.asyncio
async def test_graph_text_only_response():
    """Test that text-only response goes directly to END."""
    llm = MockLLM(["<assistant-chat>This is just text, no code.</assistant-chat>"])
    graph = build_agent_graph(llm=llm, max_turns=5)

    initial_state: AgentState = {
        "messages": [HumanMessage(content="hello")],
        "system_prompt": "You are helpful.",
        "original_prompt": "hello",
        "turn_count": 0,
        "session": SessionState(),
        "on_execution": None,
    }

    result = await graph.ainvoke(initial_state)

    assert result["current_response"] == "<assistant-chat>This is just text, no code.</assistant-chat>"
    assert result["code_blocks"] == []
    assert result["turn_count"] == 1


@pytest.mark.asyncio
async def test_graph_code_then_complete():
    """Test code execution followed by reflection saying complete."""
    llm = MockLLM(
        [
            "<assistant-chat>Here's the code:</assistant-chat><assistant-repl-in>x = 42\nprint(x)</assistant-repl-in>",
            '{"is_complete": true, "reasoning": "Variable set and printed", "next_action": null}',
        ]
    )
    graph = build_agent_graph(llm=llm, max_turns=5)

    session = SessionState()
    initial_state: AgentState = {
        "messages": [HumanMessage(content="set x to 42 and print it")],
        "system_prompt": "You are helpful.",
        "original_prompt": "set x to 42 and print it",
        "turn_count": 0,
        "session": session,
        "on_execution": None,
    }

    result = await graph.ainvoke(initial_state)

    assert result["reflection"].is_complete is True
    assert session.globals_ns.get("x") == 42 or session.locals_ns.get("x") == 42
    assert result["turn_count"] == 1


@pytest.mark.asyncio
async def test_graph_multi_turn():
    """Test reflection triggering another generation."""
    llm = MockLLM(
        [
            "<assistant-chat>First:</assistant-chat><assistant-repl-in>x = 1</assistant-repl-in>",
            '{"is_complete": false, "reasoning": "Need to also print", "next_action": "Print x"}',
            "<assistant-chat>Now print:</assistant-chat><assistant-repl-in>print(x)</assistant-repl-in>",
            '{"is_complete": true, "reasoning": "Done", "next_action": null}',
        ]
    )
    graph = build_agent_graph(llm=llm, max_turns=5)

    session = SessionState()
    initial_state: AgentState = {
        "messages": [HumanMessage(content="set x and print it")],
        "system_prompt": "You are helpful.",
        "original_prompt": "set x and print it",
        "turn_count": 0,
        "session": session,
        "on_execution": None,
    }

    result = await graph.ainvoke(initial_state)

    # Should have gone through 2 generate cycles
    assert result["turn_count"] == 2
    assert result["reflection"].is_complete is True


@pytest.mark.asyncio
async def test_graph_max_turns_limit():
    """Test that max_turns stops infinite loops."""
    # LLM always says incomplete
    llm = MockLLM(
        [
            "<assistant-repl-in>x = 1</assistant-repl-in>",
            '{"is_complete": false, "reasoning": "Never done", "next_action": "Keep going"}',
        ]
    )
    graph = build_agent_graph(llm=llm, max_turns=2)

    initial_state: AgentState = {
        "messages": [HumanMessage(content="infinite task")],
        "system_prompt": "You are helpful.",
        "original_prompt": "infinite task",
        "turn_count": 0,
        "session": SessionState(),
        "on_execution": None,
    }

    result = await graph.ainvoke(initial_state)

    # Should stop at max_turns
    assert result["turn_count"] == 2
    assert result["reflection"].is_complete is False
