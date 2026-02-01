# LangGraph Reflection Node Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Convert the manual agentic loop to LangGraph StateGraph with a reflection node that evaluates code correctness and completeness.

**Architecture:** StateGraph with nodes for generate, extract_code, execute, and reflect. Conditional edges route based on code presence and reflection judgment. The REPLAgent interface stays identical.

**Tech Stack:** LangGraph, LangChain Core, Pydantic

---

## Task 1: Create Test Infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `pyproject.toml` (modify to add pytest)

**Step 1: Add pytest to dev dependencies**

In `pyproject.toml`, change line 18:
```python
dev = ["ruff", "pre-commit", "pytest>=8.0.0", "pytest-asyncio>=0.23.0"]
```

**Step 2: Create tests directory and conftest**

Create `tests/__init__.py` (empty file).

Create `tests/conftest.py`:
```python
"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import pytest

from mahtab.core.state import SessionState


@pytest.fixture
def session() -> SessionState:
    """Create a fresh session state for testing."""
    return SessionState()


@pytest.fixture
def session_with_namespace() -> SessionState:
    """Create session with some variables in namespace."""
    session = SessionState()
    session.globals_ns = {"x": 42, "name": "test"}
    return session
```

**Step 3: Run pytest to verify setup**

Run: `cd /Users/dzaramelcone/lab/rlm && uv run pytest tests/ -v`
Expected: "no tests ran" or empty collection (no errors)

**Step 4: Commit**

```bash
git add pyproject.toml tests/
git commit -m "chore: add pytest infrastructure"
```

---

## Task 2: Add Reflection Prompt Template

**Files:**
- Modify: `mahtab/llm/prompts.py`
- Create: `tests/test_prompts.py`

**Step 1: Write failing test for reflection prompt**

Create `tests/test_prompts.py`:
```python
"""Tests for prompt templates."""

from mahtab.llm.prompts import build_reflection_prompt


def test_build_reflection_prompt_includes_original_prompt():
    result = build_reflection_prompt(
        original_prompt="calculate 2+2",
        code_blocks=["print(2+2)"],
        execution_results=[("4", False)],
    )
    assert "calculate 2+2" in result
    assert "print(2+2)" in result
    assert "4" in result


def test_build_reflection_prompt_marks_errors():
    result = build_reflection_prompt(
        original_prompt="divide by zero",
        code_blocks=["1/0"],
        execution_results=[("Error: division by zero", True)],
    )
    assert "ERROR" in result or "error" in result.lower()
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/dzaramelcone/lab/rlm && uv run pytest tests/test_prompts.py -v`
Expected: FAIL with "cannot import name 'build_reflection_prompt'"

**Step 3: Implement reflection prompt template**

Add to `mahtab/llm/prompts.py` after line 108:
```python


# Reflection prompt for evaluating code execution
REFLECTION_PROMPT_TEMPLATE = """Evaluate whether the code execution satisfied the user's request.

## Original Request
{original_prompt}

## Code Executed
{code_blocks}

## Execution Output
{execution_results}

## Instructions
Evaluate:
1. CORRECTNESS: Did the code run without errors? If there were errors, are they blocking the task?
2. COMPLETENESS: Does the output satisfy what the user asked for?

Respond with ONLY valid JSON (no markdown, no explanation):
{{"is_complete": true/false, "reasoning": "brief explanation", "next_action": "what to do next" or null}}"""


def build_reflection_prompt(
    original_prompt: str,
    code_blocks: list[str],
    execution_results: list[tuple[str, bool]],
) -> str:
    """Build the prompt for reflection evaluation.

    Args:
        original_prompt: The user's original request.
        code_blocks: List of code blocks that were executed.
        execution_results: List of (output, is_error) tuples.

    Returns:
        Formatted reflection prompt string.
    """
    # Format code blocks
    code_str = "\n\n".join(
        f"```python\n{block}\n```" for block in code_blocks
    )

    # Format execution results with error markers
    results_parts = []
    for i, (output, is_error) in enumerate(execution_results, 1):
        status = "[ERROR]" if is_error else "[OK]"
        results_parts.append(f"Block {i} {status}:\n{output}")
    results_str = "\n\n".join(results_parts)

    return REFLECTION_PROMPT_TEMPLATE.format(
        original_prompt=original_prompt,
        code_blocks=code_str,
        execution_results=results_str,
    )
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/dzaramelcone/lab/rlm && uv run pytest tests/test_prompts.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mahtab/llm/prompts.py tests/test_prompts.py
git commit -m "feat: add reflection prompt template"
```

---

## Task 3: Create AgentState and ReflectionResult Types

**Files:**
- Create: `mahtab/agent/graph.py`
- Create: `tests/test_graph.py`

**Step 1: Write failing test for types**

Create `tests/test_graph.py`:
```python
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
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/dzaramelcone/lab/rlm && uv run pytest tests/test_graph.py::test_reflection_result_complete -v`
Expected: FAIL with "cannot import name 'AgentState'"

**Step 3: Implement types**

Create `mahtab/agent/graph.py`:
```python
"""LangGraph-based agent implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from pydantic import BaseModel

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

    from mahtab.core.state import SessionState


class ReflectionResult(BaseModel):
    """Result of the reflection node's evaluation."""

    is_complete: bool
    reasoning: str
    next_action: str | None = None


class AgentState(TypedDict, total=False):
    """State that flows through the agent graph.

    Attributes:
        messages: Conversation history (LangChain messages).
        system_prompt: The system prompt built at start.
        original_prompt: User's initial request for reflection.
        current_response: Claude's latest response text.
        code_blocks: Extracted Python code blocks.
        execution_results: List of (output, is_error) tuples.
        turn_count: Number of generate cycles completed.
        session: The SessionState for namespace and persistence.
        reflection: Result of the last reflection (if any).
    """

    messages: list[BaseMessage]
    system_prompt: str
    original_prompt: str
    current_response: str
    code_blocks: list[str]
    execution_results: list[tuple[str, bool]]
    turn_count: int
    session: SessionState
    reflection: ReflectionResult | None
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/dzaramelcone/lab/rlm && uv run pytest tests/test_graph.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mahtab/agent/graph.py tests/test_graph.py
git commit -m "feat: add AgentState and ReflectionResult types"
```

---

## Task 4: Implement extract_code_node

**Files:**
- Modify: `mahtab/agent/graph.py`
- Modify: `tests/test_graph.py`

**Step 1: Write failing tests for extract_code_node**

Add to `tests/test_graph.py`:
```python
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
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/dzaramelcone/lab/rlm && uv run pytest tests/test_graph.py::test_extract_code_node_single_block -v`
Expected: FAIL with "cannot import name 'extract_code_node'"

**Step 3: Implement extract_code_node**

Add to `mahtab/agent/graph.py` after the AgentState class:
```python
import re


def extract_code_node(state: AgentState) -> dict:
    """Extract Python code blocks from the current response.

    Args:
        state: Current agent state with current_response.

    Returns:
        Dict with code_blocks list to merge into state.
    """
    response = state.get("current_response", "")
    blocks = re.findall(r"```python\n(.*?)```", response, re.DOTALL)
    return {"code_blocks": [b.strip() for b in blocks]}
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/dzaramelcone/lab/rlm && uv run pytest tests/test_graph.py::test_extract_code_node_single_block tests/test_graph.py::test_extract_code_node_multiple_blocks tests/test_graph.py::test_extract_code_node_no_blocks -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mahtab/agent/graph.py tests/test_graph.py
git commit -m "feat: add extract_code_node"
```

---

## Task 5: Implement execute_node

**Files:**
- Modify: `mahtab/agent/graph.py`
- Modify: `tests/test_graph.py`

**Step 1: Write failing tests for execute_node**

Add to `tests/test_graph.py`:
```python
from mahtab.core.state import SessionState


def test_execute_node_success():
    from mahtab.agent.graph import execute_node

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

    session = SessionState()
    state: AgentState = {
        "code_blocks": ["my_var = 'hello'"],
        "execution_results": [],
        "session": session,
    }
    execute_node(state)
    assert session.globals_ns.get("my_var") == "hello"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/dzaramelcone/lab/rlm && uv run pytest tests/test_graph.py::test_execute_node_success -v`
Expected: FAIL with "cannot import name 'execute_node'"

**Step 3: Implement execute_node**

Add to `mahtab/agent/graph.py`:
```python
from mahtab.core.executor import execute_code


def execute_node(state: AgentState) -> dict:
    """Execute all code blocks and collect results.

    Args:
        state: Current agent state with code_blocks and session.

    Returns:
        Dict with execution_results list to merge into state.
    """
    session = state["session"]
    results = []

    for block in state.get("code_blocks", []):
        output, is_error = execute_code(block, session)
        results.append((output, is_error))

    return {"execution_results": results}
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/dzaramelcone/lab/rlm && uv run pytest tests/test_graph.py::test_execute_node_success tests/test_graph.py::test_execute_node_error tests/test_graph.py::test_execute_node_updates_namespace -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mahtab/agent/graph.py tests/test_graph.py
git commit -m "feat: add execute_node"
```

---

## Task 6: Implement reflect_node

**Files:**
- Modify: `mahtab/agent/graph.py`
- Modify: `tests/test_graph.py`

**Step 1: Write failing tests for reflect_node**

Add to `tests/test_graph.py`:
```python
import json
from unittest.mock import AsyncMock, MagicMock


def test_reflect_node_parses_complete():
    from mahtab.agent.graph import ReflectionResult, _parse_reflection_response

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
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/dzaramelcone/lab/rlm && uv run pytest tests/test_graph.py::test_reflect_node_parses_complete -v`
Expected: FAIL with "cannot import name '_parse_reflection_response'"

**Step 3: Implement _parse_reflection_response and reflect_node**

Add to `mahtab/agent/graph.py`:
```python
import json

from langchain_core.messages import HumanMessage, SystemMessage

from mahtab.llm.prompts import build_reflection_prompt


def _parse_reflection_response(response: str) -> ReflectionResult:
    """Parse LLM response into ReflectionResult.

    Args:
        response: Raw LLM response (should be JSON).

    Returns:
        ReflectionResult, defaulting to incomplete on parse failure.
    """
    try:
        # Try to extract JSON from response (may have surrounding text)
        # Look for JSON object pattern
        json_match = re.search(r'\{[^{}]*"is_complete"[^{}]*\}', response)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(response)

        return ReflectionResult(
            is_complete=data.get("is_complete", False),
            reasoning=data.get("reasoning", ""),
            next_action=data.get("next_action"),
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        return ReflectionResult(
            is_complete=False,
            reasoning="Failed to parse reflection response",
            next_action="Retry with clearer output",
        )


async def reflect_node(state: AgentState, llm) -> dict:
    """Evaluate whether execution satisfied the original request.

    Args:
        state: Current agent state.
        llm: Language model for reflection call.

    Returns:
        Dict with reflection result to merge into state.
    """
    prompt = build_reflection_prompt(
        original_prompt=state["original_prompt"],
        code_blocks=state.get("code_blocks", []),
        execution_results=state.get("execution_results", []),
    )

    messages = [
        SystemMessage(content="You evaluate code execution results. Respond only with JSON."),
        HumanMessage(content=prompt),
    ]

    response = await llm.ainvoke(messages)
    result = _parse_reflection_response(response.content)

    return {"reflection": result}
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/dzaramelcone/lab/rlm && uv run pytest tests/test_graph.py::test_reflect_node_parses_complete tests/test_graph.py::test_reflect_node_parses_incomplete tests/test_graph.py::test_reflect_node_handles_malformed_json -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mahtab/agent/graph.py tests/test_graph.py
git commit -m "feat: add reflect_node with JSON parsing"
```

---

## Task 7: Implement generate_node

**Files:**
- Modify: `mahtab/agent/graph.py`
- Modify: `tests/test_graph.py`

**Step 1: Write failing test for generate_node**

Add to `tests/test_graph.py`:
```python
import pytest


@pytest.mark.asyncio
async def test_generate_node_calls_llm():
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    from mahtab.agent.graph import generate_node

    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = AIMessage(content="Here is code:\n```python\nx=1\n```")

    state: AgentState = {
        "messages": [HumanMessage(content="set x to 1")],
        "system_prompt": "You are helpful.",
        "turn_count": 0,
    }

    result = await generate_node(state, mock_llm)
    assert "```python" in result["current_response"]
    assert result["turn_count"] == 1
    mock_llm.ainvoke.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/dzaramelcone/lab/rlm && uv run pytest tests/test_graph.py::test_generate_node_calls_llm -v`
Expected: FAIL with "cannot import name 'generate_node'"

**Step 3: Implement generate_node**

Add to `mahtab/agent/graph.py`:
```python
async def generate_node(state: AgentState, llm) -> dict:
    """Generate a response from the LLM.

    Args:
        state: Current agent state with messages and system_prompt.
        llm: Language model to call.

    Returns:
        Dict with current_response and incremented turn_count.
    """
    messages = [
        SystemMessage(content=state["system_prompt"]),
        *state["messages"],
    ]

    response = await llm.ainvoke(messages)

    return {
        "current_response": response.content,
        "turn_count": state.get("turn_count", 0) + 1,
    }
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/dzaramelcone/lab/rlm && uv run pytest tests/test_graph.py::test_generate_node_calls_llm -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mahtab/agent/graph.py tests/test_graph.py
git commit -m "feat: add generate_node"
```

---

## Task 8: Implement Edge Conditionals

**Files:**
- Modify: `mahtab/agent/graph.py`
- Modify: `tests/test_graph.py`

**Step 1: Write failing tests for conditionals**

Add to `tests/test_graph.py`:
```python
def test_should_execute_with_code():
    from mahtab.agent.graph import should_execute

    state: AgentState = {"code_blocks": ["x = 1"]}
    assert should_execute(state) == "execute"


def test_should_execute_no_code():
    from mahtab.agent.graph import should_execute

    state: AgentState = {"code_blocks": []}
    assert should_execute(state) == "end"


def test_should_continue_complete():
    from mahtab.agent.graph import ReflectionResult, should_continue

    state: AgentState = {
        "reflection": ReflectionResult(is_complete=True, reasoning="Done"),
        "turn_count": 1,
    }
    assert should_continue(state, max_turns=5) == "end"


def test_should_continue_incomplete_under_limit():
    from mahtab.agent.graph import ReflectionResult, should_continue

    state: AgentState = {
        "reflection": ReflectionResult(is_complete=False, reasoning="Need more", next_action="Fix it"),
        "turn_count": 2,
    }
    assert should_continue(state, max_turns=5) == "generate"


def test_should_continue_incomplete_at_limit():
    from mahtab.agent.graph import ReflectionResult, should_continue

    state: AgentState = {
        "reflection": ReflectionResult(is_complete=False, reasoning="Need more"),
        "turn_count": 5,
    }
    assert should_continue(state, max_turns=5) == "end"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/dzaramelcone/lab/rlm && uv run pytest tests/test_graph.py::test_should_execute_with_code -v`
Expected: FAIL with "cannot import name 'should_execute'"

**Step 3: Implement edge conditionals**

Add to `mahtab/agent/graph.py`:
```python
def should_execute(state: AgentState) -> str:
    """Determine whether to execute code or end.

    Args:
        state: Current agent state.

    Returns:
        "execute" if there are code blocks, "end" otherwise.
    """
    if state.get("code_blocks"):
        return "execute"
    return "end"


def should_continue(state: AgentState, max_turns: int = 5) -> str:
    """Determine whether to continue generating or end.

    Args:
        state: Current agent state with reflection result.
        max_turns: Maximum number of generation turns.

    Returns:
        "generate" to continue, "end" to finish.
    """
    reflection = state.get("reflection")

    # If reflection says complete, we're done
    if reflection and reflection.is_complete:
        return "end"

    # If we've hit max turns, stop
    if state.get("turn_count", 0) >= max_turns:
        return "end"

    # Otherwise, continue
    return "generate"
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/dzaramelcone/lab/rlm && uv run pytest tests/test_graph.py::test_should_execute_with_code tests/test_graph.py::test_should_execute_no_code tests/test_graph.py::test_should_continue_complete tests/test_graph.py::test_should_continue_incomplete_under_limit tests/test_graph.py::test_should_continue_incomplete_at_limit -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mahtab/agent/graph.py tests/test_graph.py
git commit -m "feat: add edge conditionals should_execute and should_continue"
```

---

## Task 9: Build the StateGraph

**Files:**
- Modify: `mahtab/agent/graph.py`
- Modify: `tests/test_graph.py`

**Step 1: Write failing test for graph builder**

Add to `tests/test_graph.py`:
```python
def test_build_agent_graph_creates_graph():
    from mahtab.agent.graph import build_agent_graph

    mock_llm = MagicMock()
    graph = build_agent_graph(llm=mock_llm, max_turns=3)

    # Verify it's a compiled graph
    assert hasattr(graph, "invoke")
    assert hasattr(graph, "ainvoke")
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/dzaramelcone/lab/rlm && uv run pytest tests/test_graph.py::test_build_agent_graph_creates_graph -v`
Expected: FAIL with "cannot import name 'build_agent_graph'"

**Step 3: Implement build_agent_graph**

Add to `mahtab/agent/graph.py`:
```python
from functools import partial

from langgraph.graph import END, StateGraph


def build_agent_graph(llm, max_turns: int = 5):
    """Build the agent StateGraph.

    Args:
        llm: Language model for generate and reflect nodes.
        max_turns: Maximum generation turns before stopping.

    Returns:
        Compiled StateGraph.
    """
    graph = StateGraph(AgentState)

    # Add nodes - wrap async nodes with llm dependency
    async def _generate(state):
        return await generate_node(state, llm)

    async def _reflect(state):
        return await reflect_node(state, llm)

    graph.add_node("generate", _generate)
    graph.add_node("extract_code", extract_code_node)
    graph.add_node("execute", execute_node)
    graph.add_node("reflect", _reflect)

    # Set entry point
    graph.set_entry_point("generate")

    # Add edges
    graph.add_edge("generate", "extract_code")

    # Conditional: has code? -> execute or end
    graph.add_conditional_edges(
        "extract_code",
        should_execute,
        {"execute": "execute", "end": END},
    )

    graph.add_edge("execute", "reflect")

    # Conditional: complete? -> end or generate
    graph.add_conditional_edges(
        "reflect",
        partial(should_continue, max_turns=max_turns),
        {"generate": "generate", "end": END},
    )

    return graph.compile()
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/dzaramelcone/lab/rlm && uv run pytest tests/test_graph.py::test_build_agent_graph_creates_graph -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mahtab/agent/graph.py tests/test_graph.py
git commit -m "feat: add build_agent_graph function"
```

---

## Task 10: Add Message Update Logic

**Files:**
- Modify: `mahtab/agent/graph.py`
- Modify: `tests/test_graph.py`

**Step 1: Write failing test for message updates**

Add to `tests/test_graph.py`:
```python
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

    # Should have original + AI response + execution result
    assert len(new_messages) == 3
    assert isinstance(new_messages[1], AIMessage)
    assert isinstance(new_messages[2], HumanMessage)
    assert "<execution>" in new_messages[2].content
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/dzaramelcone/lab/rlm && uv run pytest tests/test_graph.py::test_update_messages_after_execution -v`
Expected: FAIL with "cannot import name 'update_messages_node'"

**Step 3: Implement update_messages_node**

Add to `mahtab/agent/graph.py`:
```python
from langchain_core.messages import AIMessage


def update_messages_node(state: AgentState) -> dict:
    """Update messages with assistant response and execution results.

    Args:
        state: Current agent state.

    Returns:
        Dict with updated messages list.
    """
    messages = list(state.get("messages", []))

    # Add assistant's response
    messages.append(AIMessage(content=state["current_response"]))

    # Add execution results as user message
    results = state.get("execution_results", [])
    if results:
        exec_report = "\n\n".join(
            f"Code block {i + 1} output:\n{out}"
            for i, (out, _) in enumerate(results)
        )
        messages.append(HumanMessage(content=f"<execution>\n{exec_report}\n</execution>"))

    return {"messages": messages}
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/dzaramelcone/lab/rlm && uv run pytest tests/test_graph.py::test_update_messages_after_execution -v`
Expected: PASS

**Step 5: Update build_agent_graph to include update_messages_node**

Modify `build_agent_graph` in `mahtab/agent/graph.py` - add the node and edge:
```python
def build_agent_graph(llm, max_turns: int = 5):
    """Build the agent StateGraph.

    Args:
        llm: Language model for generate and reflect nodes.
        max_turns: Maximum generation turns before stopping.

    Returns:
        Compiled StateGraph.
    """
    graph = StateGraph(AgentState)

    # Add nodes - wrap async nodes with llm dependency
    async def _generate(state):
        return await generate_node(state, llm)

    async def _reflect(state):
        return await reflect_node(state, llm)

    graph.add_node("generate", _generate)
    graph.add_node("extract_code", extract_code_node)
    graph.add_node("execute", execute_node)
    graph.add_node("update_messages", update_messages_node)
    graph.add_node("reflect", _reflect)

    # Set entry point
    graph.set_entry_point("generate")

    # Add edges
    graph.add_edge("generate", "extract_code")

    # Conditional: has code? -> execute or end
    graph.add_conditional_edges(
        "extract_code",
        should_execute,
        {"execute": "execute", "end": END},
    )

    graph.add_edge("execute", "update_messages")
    graph.add_edge("update_messages", "reflect")

    # Conditional: complete? -> end or generate
    graph.add_conditional_edges(
        "reflect",
        partial(should_continue, max_turns=max_turns),
        {"generate": "generate", "end": END},
    )

    return graph.compile()
```

**Step 6: Run all graph tests**

Run: `cd /Users/dzaramelcone/lab/rlm && uv run pytest tests/test_graph.py -v`
Expected: PASS

**Step 7: Commit**

```bash
git add mahtab/agent/graph.py tests/test_graph.py
git commit -m "feat: add update_messages_node to graph"
```

---

## Task 11: Integration Test - Full Graph Flow

**Files:**
- Create: `tests/test_graph_integration.py`

**Step 1: Write integration test for complete flow**

Create `tests/test_graph_integration.py`:
```python
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

    async def ainvoke(self, messages):
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return AIMessage(content=response)


@pytest.mark.asyncio
async def test_graph_text_only_response():
    """Test that text-only response goes directly to END."""
    llm = MockLLM(["This is just text, no code."])
    graph = build_agent_graph(llm=llm, max_turns=5)

    initial_state: AgentState = {
        "messages": [HumanMessage(content="hello")],
        "system_prompt": "You are helpful.",
        "original_prompt": "hello",
        "turn_count": 0,
        "session": SessionState(),
    }

    result = await graph.ainvoke(initial_state)

    assert result["current_response"] == "This is just text, no code."
    assert result["code_blocks"] == []
    assert result["turn_count"] == 1


@pytest.mark.asyncio
async def test_graph_code_then_complete():
    """Test code execution followed by reflection saying complete."""
    llm = MockLLM([
        "Here's the code:\n```python\nx = 42\nprint(x)\n```",
        '{"is_complete": true, "reasoning": "Variable set and printed", "next_action": null}',
    ])
    graph = build_agent_graph(llm=llm, max_turns=5)

    session = SessionState()
    initial_state: AgentState = {
        "messages": [HumanMessage(content="set x to 42 and print it")],
        "system_prompt": "You are helpful.",
        "original_prompt": "set x to 42 and print it",
        "turn_count": 0,
        "session": session,
    }

    result = await graph.ainvoke(initial_state)

    assert result["reflection"].is_complete is True
    assert session.globals_ns.get("x") == 42
    assert result["turn_count"] == 1


@pytest.mark.asyncio
async def test_graph_multi_turn():
    """Test reflection triggering another generation."""
    llm = MockLLM([
        "First:\n```python\nx = 1\n```",
        '{"is_complete": false, "reasoning": "Need to also print", "next_action": "Print x"}',
        "Now print:\n```python\nprint(x)\n```",
        '{"is_complete": true, "reasoning": "Done", "next_action": null}',
    ])
    graph = build_agent_graph(llm=llm, max_turns=5)

    session = SessionState()
    initial_state: AgentState = {
        "messages": [HumanMessage(content="set x and print it")],
        "system_prompt": "You are helpful.",
        "original_prompt": "set x and print it",
        "turn_count": 0,
        "session": session,
    }

    result = await graph.ainvoke(initial_state)

    # Should have gone through 2 generate cycles
    assert result["turn_count"] == 2
    assert result["reflection"].is_complete is True


@pytest.mark.asyncio
async def test_graph_max_turns_limit():
    """Test that max_turns stops infinite loops."""
    # LLM always says incomplete
    llm = MockLLM([
        "```python\nx = 1\n```",
        '{"is_complete": false, "reasoning": "Never done", "next_action": "Keep going"}',
    ])
    graph = build_agent_graph(llm=llm, max_turns=2)

    initial_state: AgentState = {
        "messages": [HumanMessage(content="infinite task")],
        "system_prompt": "You are helpful.",
        "original_prompt": "infinite task",
        "turn_count": 0,
        "session": SessionState(),
    }

    result = await graph.ainvoke(initial_state)

    # Should stop at max_turns
    assert result["turn_count"] == 2
    assert result["reflection"].is_complete is False
```

**Step 2: Run integration tests**

Run: `cd /Users/dzaramelcone/lab/rlm && uv run pytest tests/test_graph_integration.py -v`
Expected: PASS (or failures that reveal bugs to fix)

**Step 3: Fix any issues found, then commit**

```bash
git add tests/test_graph_integration.py
git commit -m "test: add integration tests for agent graph"
```

---

## Task 12: Refactor REPLAgent to Use Graph

**Files:**
- Modify: `mahtab/agent/repl_agent.py`
- Create: `tests/test_repl_agent.py`

**Step 1: Write failing test for new REPLAgent**

Create `tests/test_repl_agent.py`:
```python
"""Tests for REPLAgent using the graph."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

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
    }

    with patch.object(agent, "_graph", mock_graph):
        result = await agent.ask("hi")

    assert result == "Hello!"
    mock_graph.ainvoke.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/dzaramelcone/lab/rlm && uv run pytest tests/test_repl_agent.py -v`
Expected: FAIL (agent doesn't have _graph attribute yet)

**Step 3: Refactor REPLAgent**

Replace `mahtab/agent/repl_agent.py` contents:
```python
"""REPL Agent implementation using LangGraph."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console

from mahtab.agent.graph import AgentState, build_agent_graph
from mahtab.core.state import SessionState
from mahtab.llm.claude_cli import ChatClaudeCLI
from mahtab.llm.prompts import build_repl_system_prompt
from mahtab.tools.skills import load_skill_descriptions


class REPLAgent(BaseModel):
    """Agent that manages conversation with Claude and code execution.

    Uses a LangGraph StateGraph for the agentic loop:
    1. Generate: Claude responds to the prompt
    2. Extract: Parse code blocks from response
    3. Execute: Run code blocks in session namespace
    4. Reflect: Evaluate if task is complete
    5. Loop back to Generate if incomplete

    Attributes:
        session: The session state containing namespace and history.
        llm: The language model to use (ChatClaudeCLI by default).
        console: Rich console for output (optional, for streaming).
        max_turns: Maximum number of turns in the agentic loop.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session: SessionState
    llm: BaseChatModel = Field(default_factory=ChatClaudeCLI)
    console: Console | None = None
    max_turns: int = 5

    _graph: object = None  # Compiled graph, built lazily

    def model_post_init(self, __context) -> None:
        """Build the graph after model initialization."""
        self._graph = build_agent_graph(llm=self.llm, max_turns=self.max_turns)

    async def ask(
        self,
        prompt: str,
        on_token: callable | None = None,
        on_code_block: callable | None = None,
        on_execution: callable | None = None,
    ) -> str:
        """Send a prompt to Claude and handle the conversation.

        Args:
            prompt: The user's prompt.
            on_token: Callback for each token streamed (NOT SUPPORTED in graph mode yet).
            on_code_block: Callback when a code block is detected (NOT SUPPORTED yet).
            on_execution: Callback with execution results (NOT SUPPORTED yet).

        Returns:
            The final text response from Claude.
        """
        # Build system prompt with current context
        system_prompt = build_repl_system_prompt(
            var_summary=self.session.summarize_namespace(),
            skills_description=load_skill_descriptions(self.session.skills_dir),
            repl_context=self.session.get_activity_context(),
            prior_session=self.session.load_last_session(),
        )

        # Prepare initial state
        initial_state: AgentState = {
            "messages": [*self.session.messages, HumanMessage(content=prompt)],
            "system_prompt": system_prompt,
            "original_prompt": prompt,
            "current_response": "",
            "code_blocks": [],
            "execution_results": [],
            "turn_count": 0,
            "session": self.session,
            "reflection": None,
        }

        # Run the graph
        result = await self._graph.ainvoke(initial_state)

        # Update session with final messages
        final_response = result.get("current_response", "")
        self.session.messages = result.get("messages", self.session.messages)
        self.session.save_last_session(prompt, final_response)

        return final_response

    def ask_sync(
        self,
        prompt: str,
        on_token: callable | None = None,
        on_code_block: callable | None = None,
        on_execution: callable | None = None,
    ) -> str:
        """Synchronous version of ask().

        Args:
            prompt: The user's prompt.
            on_token: Callback for each token streamed.
            on_code_block: Callback when a code block is detected.
            on_execution: Callback with execution results.

        Returns:
            The final text response from Claude.
        """
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.ask(prompt, on_token, on_code_block, on_execution)
            )
        finally:
            loop.close()

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.session.clear_history()


def create_repl_agent(
    session: SessionState | None = None,
    model: str = "claude-opus-4-20250514",
    console: Console | None = None,
    max_turns: int = 5,
) -> REPLAgent:
    """Create a REPL agent with the given configuration.

    Args:
        session: Session state. If None, creates a new one.
        model: Claude model to use.
        console: Rich console for output.
        max_turns: Maximum turns in the agentic loop.

    Returns:
        Configured REPLAgent instance.
    """
    if session is None:
        session = SessionState()

    llm = ChatClaudeCLI(model=model)

    return REPLAgent(
        session=session,
        llm=llm,
        console=console,
        max_turns=max_turns,
    )
```

**Step 4: Run all tests**

Run: `cd /Users/dzaramelcone/lab/rlm && uv run pytest tests/ -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mahtab/agent/repl_agent.py tests/test_repl_agent.py
git commit -m "refactor: REPLAgent now uses LangGraph StateGraph"
```

---

## Task 13: Final Verification and Cleanup

**Files:**
- All modified files

**Step 1: Run full test suite**

Run: `cd /Users/dzaramelcone/lab/rlm && uv run pytest tests/ -v --tb=short`
Expected: All PASS

**Step 2: Run ruff linting**

Run: `cd /Users/dzaramelcone/lab/rlm && uv run ruff check mahtab/ tests/`
Expected: No errors (or fix any that appear)

**Step 3: Run ruff formatting**

Run: `cd /Users/dzaramelcone/lab/rlm && uv run ruff format mahtab/ tests/`
Expected: Files formatted

**Step 4: Manual smoke test**

Run: `cd /Users/dzaramelcone/lab/rlm && uv run mahtab`
Test with a simple prompt like "set x to 42 and print it"
Expected: Code executes, reflection evaluates, response completes

**Step 5: Final commit if any formatting changes**

```bash
git add -A
git commit -m "chore: formatting and cleanup"
```

---

Plan complete and saved to `docs/plans/2026-01-31-langgraph-reflection-impl.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
