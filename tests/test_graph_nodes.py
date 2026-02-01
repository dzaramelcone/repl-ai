"""Tests for graph node functions."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from mahtab.agent.graph import AgentState


def test_extract_code_node_single_block():
    from mahtab.agent.graph import extract_code_node

    state: AgentState = {
        "current_response": "<assistant-chat>Here's the code:</assistant-chat><assistant-repl-in>print('hello')</assistant-repl-in>",
        "code_blocks": [],
    }
    result = extract_code_node(state)
    assert result["code_blocks"] == ["print('hello')"]


def test_extract_code_node_multiple_blocks():
    from mahtab.agent.graph import extract_code_node

    state: AgentState = {
        "current_response": "<assistant-repl-in>x = 1</assistant-repl-in><assistant-chat>Then:</assistant-chat><assistant-repl-in>y = 2</assistant-repl-in>",
        "code_blocks": [],
    }
    result = extract_code_node(state)
    assert result["code_blocks"] == ["x = 1", "y = 2"]


def test_extract_code_node_no_blocks():
    from mahtab.agent.graph import extract_code_node

    state: AgentState = {
        "current_response": "<assistant-chat>No code here, just text.</assistant-chat>",
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
        "on_execution": None,
    }
    result = execute_node(state)
    assert len(result["execution_results"]) == 2
    assert result["execution_results"][1][0] == "42\n"
    assert result["execution_results"][1][1] is False


def test_execute_node_error():
    from mahtab.agent.graph import execute_node
    from mahtab.core.state import SessionState

    session = SessionState()
    state: AgentState = {
        "code_blocks": ["1/0"],
        "execution_results": [],
        "session": session,
        "on_execution": None,
    }
    result = execute_node(state)
    assert len(result["execution_results"]) == 1
    assert "division by zero" in result["execution_results"][0][0].lower()
    assert result["execution_results"][0][1] is True


def test_execute_node_updates_namespace():
    from mahtab.agent.graph import execute_node
    from mahtab.core.state import SessionState

    session = SessionState()
    state: AgentState = {
        "code_blocks": ["my_var = 'hello'"],
        "execution_results": [],
        "session": session,
        "on_execution": None,
    }
    execute_node(state)
    assert session.locals_ns.get("my_var") == "hello"


def test_reflect_node_parses_complete():
    from mahtab.agent.graph import _parse_reflection_response

    response = "<reflection><is_complete>true</is_complete><reasoning>Task done</reasoning><next_action></next_action></reflection>"
    result = _parse_reflection_response(response)
    assert result.is_complete is True
    assert result.reasoning == "Task done"


def test_reflect_node_parses_incomplete():
    from mahtab.agent.graph import _parse_reflection_response

    response = "<reflection><is_complete>false</is_complete><reasoning>Need more</reasoning><next_action>Add validation</next_action></reflection>"
    result = _parse_reflection_response(response)
    assert result.is_complete is False
    assert result.next_action == "Add validation"


def test_reflect_node_crashes_on_malformed_xml():
    """Malformed XML should crash - fail fast, no defensive handling."""
    from xml.etree.ElementTree import ParseError

    from mahtab.agent.graph import _parse_reflection_response

    response = "This is not XML at all"
    with pytest.raises(ParseError):
        _parse_reflection_response(response)


@pytest.mark.asyncio
async def test_generate_node_calls_llm():
    from langchain_core.messages import AIMessage, HumanMessage

    from mahtab.agent.graph import generate_node

    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = AIMessage(content="Here is code:\n```python\nx=1\n```")

    state: AgentState = {
        "messages": [HumanMessage(content="set x to 1")],
        "system_prompt": "You are helpful.",
        "turn_count": 0,
    }

    result = await generate_node(state, mock_llm, callbacks=[])
    assert "```python" in result["current_response"]
    assert result["turn_count"] == 1
    mock_llm.ainvoke.assert_called_once()


def test_update_messages_after_execution():
    from langchain_core.messages import AIMessage, HumanMessage

    from mahtab.agent.graph import update_messages_node

    state: AgentState = {
        "messages": [HumanMessage(content="do something")],
        "current_response": "Here's code:\n```python\nx=1\n```",
        "execution_results": [("(no output)", False)],
    }

    result = update_messages_node(state)
    new_messages = result["messages"]

    assert len(new_messages) == 3
    assert isinstance(new_messages[1], AIMessage)
    assert isinstance(new_messages[2], HumanMessage)
    assert "<execution>" in new_messages[2].content


@pytest.mark.asyncio
async def test_generate_node_passes_callbacks():
    """generate_node should pass callbacks config to llm.ainvoke."""
    from langchain_core.messages import AIMessage, HumanMessage

    from mahtab.agent.graph import generate_node

    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = AIMessage(content="response")

    mock_callback = MagicMock()

    state = {
        "messages": [HumanMessage(content="test")],
        "system_prompt": "You are helpful.",
        "turn_count": 0,
    }

    await generate_node(state, mock_llm, callbacks=[mock_callback])

    # Verify ainvoke was called with callbacks in config
    call_kwargs = mock_llm.ainvoke.call_args
    assert "config" in call_kwargs.kwargs
    assert call_kwargs.kwargs["config"]["callbacks"] == [mock_callback]


def test_execute_node_calls_on_execution_callback():
    """execute_node should call on_execution callback for each code block."""
    from mahtab.agent.graph import execute_node
    from mahtab.core.state import SessionState

    session = SessionState()
    callback_calls = []

    def on_execution(output, is_error):
        callback_calls.append((output, is_error))

    state: AgentState = {
        "code_blocks": ["x = 42", "print(x)"],
        "execution_results": [],
        "session": session,
        "on_execution": on_execution,
    }
    execute_node(state)

    # Callback should have been called twice (once per block)
    assert len(callback_calls) == 2
    # Second block prints x, so output should be "42\n"
    assert callback_calls[1][0] == "42\n"
    assert callback_calls[1][1] is False


def test_execute_node_callback_receives_errors():
    """on_execution callback should receive is_error=True for failed code."""
    from mahtab.agent.graph import execute_node
    from mahtab.core.state import SessionState

    session = SessionState()
    callback_calls = []

    def on_execution(output, is_error):
        callback_calls.append((output, is_error))

    state: AgentState = {
        "code_blocks": ["1/0"],
        "execution_results": [],
        "session": session,
        "on_execution": on_execution,
    }
    execute_node(state)

    assert len(callback_calls) == 1
    assert "division by zero" in callback_calls[0][0].lower()
    assert callback_calls[0][1] is True


def test_execute_node_works_without_callback():
    """execute_node should work fine when on_execution is None."""
    from mahtab.agent.graph import execute_node
    from mahtab.core.state import SessionState

    session = SessionState()
    state: AgentState = {
        "code_blocks": ["x = 1"],
        "execution_results": [],
        "session": session,
        "on_execution": None,
    }
    # Should not raise
    result = execute_node(state)
    assert len(result["execution_results"]) == 1
