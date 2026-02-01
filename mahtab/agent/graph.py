"""LangGraph-based agent implementation."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel

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
        on_execution: Callback for execution output (output, is_error).
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
    on_execution: Callable[[str, bool], None] | None


def extract_code_node(state: AgentState) -> dict:
    """Extract Python code blocks from the current response.

    Args:
        state: Current agent state with current_response.

    Returns:
        Dict with code_blocks list to merge into state.
    """
    print(">>> extract_code_node")
    response = state.get("current_response", "")
    blocks = re.findall(r"```python\n(.*?)```", response, re.DOTALL)
    print("    found %d code blocks", len(blocks))
    return {"code_blocks": [b.strip() for b in blocks]}


def execute_node(state: AgentState) -> dict:
    """Execute all code blocks and collect results.

    Args:
        state: Current agent state with code_blocks and session.

    Returns:
        Dict with execution_results list to merge into state.
    """
    print(">>> execute_node")
    from mahtab.core.executor import execute_code

    session = state["session"]
    results = []
    on_execution = state.get("on_execution")

    for i, block in enumerate(state.get("code_blocks", [])):
        print("    executing block %d", i + 1)
        output, is_error = execute_code(block, session)
        results.append((output, is_error))

        # Call execution callback if provided
        if on_execution:
            on_execution(output, is_error)

        if is_error:
            print("    block %d errored: %s", i + 1, output[:100] if output else "")

    print("    executed %d blocks, %d errors", len(results), sum(1 for _, e in results if e))
    return {"execution_results": results}


def update_messages_node(state: AgentState) -> dict:
    """Update messages with assistant response and execution results.

    Args:
        state: Current agent state.

    Returns:
        Dict with updated messages list.
    """
    print(">>> update_messages_node")
    messages = list(state.get("messages", []))

    # Add assistant's response
    messages.append(AIMessage(content=state["current_response"]))

    # Add execution results as user message
    results = state.get("execution_results", [])
    if results:
        exec_report = "\n\n".join(f"Code block {i + 1} output:\n{out}" for i, (out, _) in enumerate(results))
        messages.append(HumanMessage(content=f"<execution>\n{exec_report}\n</execution>"))

    print("    message count: %d", len(messages))
    return {"messages": messages}


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


async def generate_node(state: AgentState, llm, callbacks=None) -> dict:
    """Generate a response from the LLM.

    Args:
        state: Current agent state with messages and system_prompt.
        llm: Language model to call.
        callbacks: Optional list of callbacks to pass to the LLM.

    Returns:
        Dict with current_response and incremented turn_count.
    """
    turn = state.get("turn_count", 0) + 1
    print(">>> generate_node (turn %d)", turn)
    from langchain_core.messages import SystemMessage

    messages = [
        SystemMessage(content=state["system_prompt"]),
        *state["messages"],
    ]

    print("    invoking LLM with %d messages", len(messages))
    response = await llm.ainvoke(messages, config={"callbacks": callbacks} if callbacks else None)
    print("    got response (%d chars)", len(response.content))

    return {
        "current_response": response.content,
        "turn_count": turn,
    }


async def reflect_node(state: AgentState, llm) -> dict:
    """Evaluate whether execution satisfied the original request.

    Args:
        state: Current agent state.
        llm: Language model for reflection call.

    Returns:
        Dict with reflection result to merge into state.
    """
    print(">>> reflect_node")
    from langchain_core.messages import HumanMessage, SystemMessage

    from mahtab.llm.prompts import build_reflection_prompt

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
    print(
        "    reflection: is_complete=%s, reasoning=%s",
        result.is_complete,
        result.reasoning[:50] if result.reasoning else "",
    )

    return {"reflection": result}


def should_execute(state: AgentState) -> str:
    """Determine whether to execute code or end.

    Args:
        state: Current agent state.

    Returns:
        "execute" if there are code blocks, "end" otherwise.
    """
    has_code = bool(state.get("code_blocks"))
    result = "execute" if has_code else "end"
    print(">>> should_execute -> %s", result)
    return result


def should_continue(state: AgentState, max_turns: int = 5) -> str:
    """Determine whether to continue generating or end.

    Args:
        state: Current agent state with reflection result.
        max_turns: Maximum number of generation turns.

    Returns:
        "generate" to continue, "end" to finish.
    """
    reflection = state.get("reflection")
    turn_count = state.get("turn_count", 0)

    # If reflection says complete, we're done
    if reflection and reflection.is_complete:
        print(">>> should_continue -> end (task complete)")
        return "end"

    # If we've hit max turns, stop
    if turn_count >= max_turns:
        print(">>> should_continue -> end (max turns %d reached)", max_turns)
        return "end"

    # Otherwise, continue
    print(">>> should_continue -> generate (turn %d/%d)", turn_count, max_turns)
    return "generate"


def build_agent_graph(llm, max_turns: int = 5):
    """Build the agent StateGraph.

    Args:
        llm: Language model for generate and reflect nodes.
        max_turns: Maximum generation turns before stopping.

    Returns:
        Compiled StateGraph.
    """
    from functools import partial

    from langgraph.graph import END, StateGraph

    graph = StateGraph(AgentState)

    # Add nodes - wrap async nodes with llm dependency
    async def _generate(state, config=None):
        callbacks = None
        if config:
            cb = config.get("callbacks")
            # LangGraph wraps callbacks in AsyncCallbackManager - extract handlers
            if hasattr(cb, "handlers"):
                callbacks = cb.handlers
            elif cb:
                callbacks = cb
        return await generate_node(state, llm, callbacks=callbacks)

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
