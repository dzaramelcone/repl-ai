"""Tests for REPLAgent using the graph."""

from unittest.mock import AsyncMock, patch

import pytest

from mahtab.agent.repl_agent import REPLAgent, create_repl_agent
from mahtab.core.state import SessionState


@pytest.mark.asyncio
async def test_repl_agent_ask_uses_graph():
    """Test that REPLAgent.ask() invokes the graph."""
    session = SessionState()
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
    session = SessionState()
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
    session = SessionState()
    agent = REPLAgent(session=session)

    # Graph should be built
    assert agent._graph is not None
    # Should have invoke and ainvoke methods (compiled graph)
    assert hasattr(agent._graph, "invoke")
    assert hasattr(agent._graph, "ainvoke")


def test_create_repl_agent_defaults():
    """Test create_repl_agent with default arguments."""
    agent = create_repl_agent()

    assert agent.session is not None
    assert agent.max_turns == 5
    assert agent._graph is not None


def test_create_repl_agent_custom_max_turns():
    """Test create_repl_agent with custom max_turns."""
    agent = create_repl_agent(max_turns=10)

    assert agent.max_turns == 10


def test_repl_agent_clear_history():
    """Test that clear_history delegates to session."""
    session = SessionState()
    session.add_user_message("hello")
    session.add_assistant_message("hi")

    agent = REPLAgent(session=session)
    assert len(session.messages) == 2

    agent.clear_history()
    assert len(session.messages) == 0


def test_repl_agent_ask_sync():
    """Test synchronous ask_sync wrapper."""
    session = SessionState()
    agent = REPLAgent(session=session)

    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {
        "current_response": "Sync response!",
        "code_blocks": [],
        "turn_count": 1,
        "messages": [],
    }

    with patch.object(agent, "_graph", mock_graph):
        result = agent.ask_sync("sync test")

    assert result == "Sync response!"
