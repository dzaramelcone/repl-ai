"""LangGraph-based agent implementation."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic_xml import BaseXmlModel, element

from mahtab.core.state import SessionState


class ReflectionResult(BaseXmlModel, tag="reflection"):
    """Result of the reflection node's evaluation."""

    is_complete: bool = element()
    reasoning: str = element()
    next_action: str = element(default="")


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
    from mahtab.io.parser import parse_response

    response = state["current_response"]
    parsed = parse_response(response)
    blocks = [content.strip() for tag, content in parsed if tag == "assistant-repl-in"]
    return {"code_blocks": blocks}


def execute_node(state: AgentState) -> dict:
    """Execute all code blocks and collect results.

    Args:
        state: Current agent state with code_blocks and session.

    Returns:
        Dict with execution_results list to merge into state.
    """
    import logging

    from mahtab.core.executor import execute_code

    log = logging.getLogger("mahtab")
    session = state["session"]
    results = []
    on_execution = state["on_execution"]

    for block in state["code_blocks"]:
        # Log the code being executed
        log.info(block, extra={"tag": "assistant-repl-in"})

        output, is_error = execute_code(block, session, output_limit=10000)
        results.append((output, is_error))

        # Log the execution output
        log.info(output, extra={"tag": "assistant-repl-out"})

        if on_execution:
            on_execution(output, is_error)

    return {"execution_results": results}


def update_messages_node(state: AgentState) -> dict:
    """Update messages with assistant response and execution results.

    Args:
        state: Current agent state.

    Returns:
        Dict with updated messages list.
    """
    messages = list(state["messages"])

    # Add assistant's response
    messages.append(AIMessage(content=state["current_response"]))

    # Add execution results as user message
    results = state["execution_results"]
    if results:
        exec_report = "\n\n".join(f"Code block {i + 1} output:\n{out}" for i, (out, _) in enumerate(results))
        messages.append(HumanMessage(content=f"<execution>\n{exec_report}\n</execution>"))

    return {"messages": messages}


def _parse_reflection_response(response: str) -> ReflectionResult:
    """Parse LLM response into ReflectionResult.

    Args:
        response: Raw LLM response (should be XML).

    Returns:
        ReflectionResult parsed from XML.
    """
    # Extract <reflection>...</reflection> block from response
    xml_match = re.search(r"<reflection>.*?</reflection>", response, re.DOTALL)
    xml_str = xml_match.group() if xml_match else response
    return ReflectionResult.from_xml(xml_str)


async def generate_node(state: AgentState, llm, callbacks) -> dict:
    """Generate a response from the LLM.

    Args:
        state: Current agent state with messages and system_prompt.
        llm: Language model to call.
        callbacks: Optional list of callbacks to pass to the LLM.

    Returns:
        Dict with current_response and incremented turn_count.
    """
    from langchain_core.messages import SystemMessage

    turn = state["turn_count"] + 1
    messages = [
        SystemMessage(content=state["system_prompt"]),
        *state["messages"],
    ]

    response = await llm.ainvoke(messages, config={"callbacks": callbacks} if callbacks else None)

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
    from langchain_core.messages import HumanMessage, SystemMessage

    from mahtab.llm.prompts import build_reflection_prompt

    prompt = build_reflection_prompt(
        original_prompt=state["original_prompt"],
        code_blocks=state["code_blocks"],
        execution_results=state["execution_results"],
    )

    messages = [
        SystemMessage(content="You evaluate code execution results. Respond only with XML."),
        HumanMessage(content=prompt),
    ]

    response = await llm.ainvoke(messages)
    result = _parse_reflection_response(response.content)

    return {"reflection": result}


def should_execute(state: AgentState) -> str:
    """Determine whether to execute code or end.

    Args:
        state: Current agent state.

    Returns:
        "execute" if there are code blocks, "end" otherwise.
    """
    return "execute" if state["code_blocks"] else "end"


def should_continue(state: AgentState, max_turns: int) -> str:
    """Determine whether to continue generating or end.

    Args:
        state: Current agent state with reflection result.
        max_turns: Maximum number of generation turns.

    Returns:
        "generate" to continue, "end" to finish.
    """
    reflection = state["reflection"]
    turn_count = state["turn_count"]

    if reflection and reflection.is_complete:
        return "end"

    if turn_count >= max_turns:
        return "end"

    return "generate"


def build_agent_graph(llm, max_turns: int):
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
    async def _generate(state, config):
        # LangGraph wraps callbacks in AsyncCallbackManager
        callbacks = config["callbacks"].handlers
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
