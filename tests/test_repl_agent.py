"""Tests for REPLAgent using the graph."""

from unittest.mock import AsyncMock, patch

import pytest

from mahtab.agent.repl_agent import REPLAgent, create_repl_agent
from mahtab.core.state import SessionState
from mahtab.io.handlers import PromptHandler


@pytest.mark.asyncio
async def test_repl_agent_ask_uses_graph():
    """Test that REPLAgent.ask() invokes the graph."""
    session = SessionState()
    prompt_handler = PromptHandler()
    agent = REPLAgent(session=session, prompt_handler=prompt_handler)

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
    agent = REPLAgent(session=session, prompt_handler=PromptHandler())

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
    agent = REPLAgent(session=session, prompt_handler=PromptHandler())

    # Graph should be built
    assert agent._graph is not None
    # Should have invoke and ainvoke methods (compiled graph)
    assert hasattr(agent._graph, "invoke")
    assert hasattr(agent._graph, "ainvoke")


def test_create_repl_agent_with_all_args():
    """Test create_repl_agent with explicit arguments."""
    session = SessionState()
    agent = create_repl_agent(
        session=session, prompt_handler=PromptHandler(), model="claude-haiku-4-5-20251001", max_turns=5
    )

    assert agent.session is session
    assert agent.max_turns == 5
    assert agent._graph is not None


def test_create_repl_agent_custom_max_turns():
    """Test create_repl_agent with custom max_turns."""
    session = SessionState()
    agent = create_repl_agent(
        session=session, prompt_handler=PromptHandler(), model="claude-haiku-4-5-20251001", max_turns=10
    )

    assert agent.max_turns == 10


def test_repl_agent_clear_history():
    """Test that clear_history delegates to session."""
    session = SessionState()
    session.add_user_message("hello")
    session.add_assistant_message("hi")

    agent = REPLAgent(session=session, prompt_handler=PromptHandler())
    assert len(session.messages) == 2

    agent.clear_history()
    assert len(session.messages) == 0


def test_repl_agent_ask_sync():
    """Test synchronous ask_sync wrapper."""
    session = SessionState()
    agent = REPLAgent(session=session, prompt_handler=PromptHandler())

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


@pytest.mark.asyncio
async def test_ask_accepts_streaming_handler():
    """ask() should accept and pass streaming_handler as callback."""
    from mahtab.ui.streaming import StreamingHandler

    session = SessionState()
    agent = REPLAgent(session=session, prompt_handler=PromptHandler())

    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {
        "current_response": "Hello!",
        "code_blocks": [],
        "turn_count": 1,
        "messages": session.messages,
    }

    from mahtab.ui.console import console as default_console

    handler = StreamingHandler(console=default_console, chars_per_second=200.0)

    with patch.object(agent, "_graph", mock_graph):
        await agent.ask("test prompt", streaming_handler=handler)

    # Verify graph.ainvoke was called with callbacks in config
    call_kwargs = mock_graph.ainvoke.call_args
    assert "config" in call_kwargs.kwargs
    assert handler in call_kwargs.kwargs["config"]["callbacks"]


@pytest.mark.asyncio
async def test_ask_passes_on_execution_callback():
    """ask() should pass on_execution callback in initial state."""
    session = SessionState()
    agent = REPLAgent(session=session, prompt_handler=PromptHandler())

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


class TestContextTagsIntegration:
    """Integration tests to verify all context tags are sent to the model."""

    @pytest.fixture
    def logger_with_handler(self):
        """Create a logger configured with PromptHandler."""
        import logging

        from mahtab.io.handlers import PromptHandler

        handler = PromptHandler()
        logger = logging.getLogger("test_context_tags")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.addHandler(handler)
        return logger, handler

    def test_user_repl_in_appears_in_context(self, logger_with_handler):
        """user-repl-in tag should appear in prompt context."""
        logger, handler = logger_with_handler
        logger.info("x = 42", extra={"tag": "user-repl-in"})
        context = handler.get_context()
        assert "<user-repl-in>" in context
        assert "x = 42" in context
        assert "</user-repl-in>" in context

    def test_user_repl_out_appears_in_context(self, logger_with_handler):
        """user-repl-out tag should appear in prompt context."""
        logger, handler = logger_with_handler
        logger.info("42", extra={"tag": "user-repl-out"})
        context = handler.get_context()
        assert "<user-repl-out>" in context
        assert "42" in context
        assert "</user-repl-out>" in context

    def test_assistant_repl_in_appears_in_context(self, logger_with_handler):
        """assistant-repl-in tag should appear in prompt context."""
        logger, handler = logger_with_handler
        logger.info("print('hello')", extra={"tag": "assistant-repl-in"})
        context = handler.get_context()
        assert "<assistant-repl-in>" in context
        assert "print('hello')" in context
        assert "</assistant-repl-in>" in context

    def test_assistant_repl_out_appears_in_context(self, logger_with_handler):
        """assistant-repl-out tag should appear in prompt context."""
        logger, handler = logger_with_handler
        logger.info("hello", extra={"tag": "assistant-repl-out"})
        context = handler.get_context()
        assert "<assistant-repl-out>" in context
        assert "hello" in context
        assert "</assistant-repl-out>" in context

    def test_user_chat_appears_in_context(self, logger_with_handler):
        """user-chat tag should appear in prompt context."""
        logger, handler = logger_with_handler
        logger.info("what is 2+2?", extra={"tag": "user-chat"})
        context = handler.get_context()
        assert "<user-chat>" in context
        assert "what is 2+2?" in context
        assert "</user-chat>" in context

    def test_assistant_chat_appears_in_context(self, logger_with_handler):
        """assistant-chat tag should appear in prompt context."""
        logger, handler = logger_with_handler
        logger.info("2+2 equals 4", extra={"tag": "assistant-chat"})
        context = handler.get_context()
        assert "<assistant-chat>" in context
        assert "2+2 equals 4" in context
        assert "</assistant-chat>" in context

    def test_all_tags_in_single_context(self, logger_with_handler):
        """All context tags should appear together when logged."""
        logger, handler = logger_with_handler

        # Log messages with all tags
        logger.info("repl input", extra={"tag": "user-repl-in"})
        logger.info("repl output", extra={"tag": "user-repl-out"})
        logger.info("assistant code", extra={"tag": "assistant-repl-in"})
        logger.info("code output", extra={"tag": "assistant-repl-out"})
        logger.info("user question", extra={"tag": "user-chat"})
        logger.info("assistant answer", extra={"tag": "assistant-chat"})

        context = handler.get_context()

        # Verify all tags present
        for tag in [
            "user-repl-in",
            "user-repl-out",
            "assistant-repl-in",
            "assistant-repl-out",
            "user-chat",
            "assistant-chat",
        ]:
            assert f"<{tag}>" in context, f"Missing opening tag: {tag}"
            assert f"</{tag}>" in context, f"Missing closing tag: {tag}"

    def test_context_cleared_after_clear(self, logger_with_handler):
        """Context should be empty after clear() is called."""
        logger, handler = logger_with_handler
        logger.info("test message", extra={"tag": "user-chat"})
        assert handler.get_context() != ""
        handler.clear()
        assert handler.get_context() == ""

    @pytest.mark.asyncio
    async def test_agent_receives_logged_context(self):
        """REPLAgent should receive logged context in system prompt."""
        session = SessionState()
        prompt_handler = PromptHandler()

        # Set up logger to use the handler
        import logging

        logger = logging.getLogger("test_agent_context")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.addHandler(prompt_handler)

        # Log some context
        logger.info("x = 1", extra={"tag": "user-repl-in"})
        logger.info("1", extra={"tag": "user-repl-out"})

        agent = REPLAgent(session=session, prompt_handler=prompt_handler)

        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "current_response": "I see x=1",
            "code_blocks": [],
            "turn_count": 1,
            "messages": session.messages,
        }

        with patch.object(agent, "_graph", mock_graph):
            await agent.ask("what is x?")

        # Verify the logged context was passed to the graph
        call_args = mock_graph.ainvoke.call_args
        initial_state = call_args[0][0]
        system_prompt = initial_state["system_prompt"]

        assert "<user-repl-in>" in system_prompt
        assert "x = 1" in system_prompt
        assert "<user-repl-out>" in system_prompt
