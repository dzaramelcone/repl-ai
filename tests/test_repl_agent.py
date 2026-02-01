"""Tests for REPLAgent using the graph."""

from unittest.mock import AsyncMock, patch

import pytest

from mahtab.agent.repl_agent import REPLAgent
from mahtab.session import Session
from mahtab.store import Store


def make_session():
    """Create a session for testing."""
    return Session(store=Store())


@pytest.mark.asyncio
async def test_repl_agent_ask_uses_graph():
    """Test that REPLAgent.ask() invokes the graph."""
    session = make_session()
    agent = REPLAgent(session=session)

    # Mock the graph
    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {
        "current_response": "Hello!",
        "code_blocks": [],
        "turn_count": 1,
        "messages": session.messages,
    }

    with patch.object(agent, "_graph", mock_graph):
        result = await agent.ask("hi")

    assert result == "Hello!"
    mock_graph.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_repl_agent_ask_updates_session_messages():
    """Test that ask() updates session messages from graph result."""
    session = make_session()
    agent = REPLAgent(session=session)

    from langchain_core.messages import AIMessage, HumanMessage

    final_messages = [
        HumanMessage(content="hi"),
        AIMessage(content="Hello there!"),
    ]

    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {
        "current_response": "Hello there!",
        "code_blocks": [],
        "turn_count": 1,
        "messages": final_messages,
    }

    with patch.object(agent, "_graph", mock_graph):
        await agent.ask("hi")

    assert session.messages == final_messages


def test_repl_agent_has_graph_after_init():
    """Test that REPLAgent builds the graph on initialization."""
    session = make_session()
    agent = REPLAgent(session=session)

    # Graph should be built
    assert agent._graph is not None
    # Should have invoke and ainvoke methods (compiled graph)
    assert hasattr(agent._graph, "invoke")
    assert hasattr(agent._graph, "ainvoke")


def test_repl_agent_clear_history():
    """Test that clear_history clears session messages."""
    session = make_session()
    from langchain_core.messages import AIMessage, HumanMessage

    session.messages.append(HumanMessage(content="hello"))
    session.messages.append(AIMessage(content="hi"))

    agent = REPLAgent(session=session)
    assert len(session.messages) == 2

    agent.clear_history()
    assert len(session.messages) == 0


@pytest.mark.asyncio
async def test_ask_accepts_streaming_handler():
    """ask() should accept and pass streaming_handler as callback."""
    from mahtab.ui.handlers import SessionStreamingHandler

    session = make_session()
    agent = REPLAgent(session=session)

    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {
        "current_response": "Hello!",
        "code_blocks": [],
        "turn_count": 1,
        "messages": session.messages,
    }

    handler = SessionStreamingHandler(session)

    with patch.object(agent, "_graph", mock_graph):
        await agent.ask("test prompt", streaming_handler=handler)

    # Verify graph.ainvoke was called with callbacks in config
    call_kwargs = mock_graph.ainvoke.call_args
    assert "config" in call_kwargs.kwargs
    assert handler in call_kwargs.kwargs["config"]["callbacks"]


@pytest.mark.asyncio
async def test_ask_passes_on_execution_callback():
    """ask() should pass on_execution callback in initial state."""
    session = make_session()
    agent = REPLAgent(session=session)

    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {
        "current_response": "Hello!",
        "code_blocks": [],
        "turn_count": 1,
        "messages": session.messages,
    }

    callback_calls = []

    def on_execution(output, is_error):
        callback_calls.append((output, is_error))

    with patch.object(agent, "_graph", mock_graph):
        await agent.ask("test prompt", on_execution=on_execution)

    # Verify graph.ainvoke was called with on_execution in initial state
    call_args = mock_graph.ainvoke.call_args
    initial_state = call_args[0][0]
    assert "on_execution" in initial_state
    assert initial_state["on_execution"] is on_execution
