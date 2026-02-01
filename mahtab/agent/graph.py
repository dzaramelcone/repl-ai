"""LangGraph-based REPL agent implementation."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from mahtab.agent.state import AgentStateDict, create_initial_state
from mahtab.core.executor import execute_code
from mahtab.llm.prompts import build_repl_system_prompt
from mahtab.tools.skills import get_skill_tool, load_skill_descriptions

if TYPE_CHECKING:
    from mahtab.core.state import SessionState


def extract_code_blocks(text: str) -> list[str]:
    """Extract Python code blocks from markdown text.

    Args:
        text: Text potentially containing ```python code blocks.

    Returns:
        List of code block contents (without the fence markers).
    """
    return re.findall(r"```python\n(.*?)```", text, re.DOTALL)


def create_repl_graph(
    llm: BaseChatModel,
    session: SessionState,
    tools: list[BaseTool] | None = None,
    max_turns: int = 5,
) -> CompiledStateGraph:
    """Create a LangGraph for the REPL agent.

    The graph implements the following flow:
    1. Model node: Call the LLM with current messages
    2. Route based on response:
       - If tool calls present -> tools node -> back to model
       - If code blocks present -> execute node -> back to model
       - Otherwise -> END

    Args:
        llm: The language model to use.
        session: Session state containing namespace and history.
        tools: Additional tools to bind to the model.
        max_turns: Maximum iterations before stopping.

    Returns:
        Compiled LangGraph ready for invocation.
    """
    # Build tools list - always include the skill tool
    skill_tool = get_skill_tool(session.skills_dir)
    all_tools = [skill_tool]
    if tools:
        all_tools.extend(tools)

    # Check if the LLM supports tool binding
    try:
        model_with_tools = llm.bind_tools(all_tools)
        supports_tools = True
    except (NotImplementedError, AttributeError):
        # LLM doesn't support tool binding (e.g., ChatClaudeCLI)
        model_with_tools = llm
        supports_tools = False

    # Track iterations to prevent infinite loops
    iteration_count = {"count": 0}

    def build_system_prompt() -> str:
        """Build the system prompt with current context."""
        return build_repl_system_prompt(
            var_summary=session.summarize_namespace(),
            skills_description=load_skill_descriptions(session.skills_dir),
            repl_context=session.get_activity_context(),
            prior_session=session.load_last_session(),
        )

    async def model_node(state: AgentStateDict) -> dict[str, Any]:
        """Call the model and return response.

        This node:
        1. Builds messages with system prompt
        2. Calls the LLM
        3. Returns the response message
        """
        iteration_count["count"] += 1

        # Build messages with fresh system prompt
        system_prompt = build_system_prompt()
        messages = [SystemMessage(content=system_prompt)]

        # Add conversation history
        for msg in state["messages"]:
            messages.append(msg)

        # Call the model
        response = await model_with_tools.ainvoke(messages)

        # Track which skills were loaded via tool calls
        loaded_skills = list(state.get("loaded_skills", []))
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                if tc.get("name") == "load_skill":
                    skill_name = tc.get("args", {}).get("name", "")
                    if skill_name and skill_name not in loaded_skills:
                        loaded_skills.append(skill_name)

        return {
            "messages": [response],
            "loaded_skills": loaded_skills,
        }

    def route_after_model(state: AgentStateDict) -> Literal["tools", "execute", "__end__"]:
        """Route based on model response.

        Routes to:
        - "tools" if there are tool calls
        - "execute" if there are Python code blocks
        - END if neither (conversation complete)
        """
        # Check iteration limit
        if iteration_count["count"] >= max_turns:
            return END

        messages = state.get("messages", [])
        if not messages:
            return END

        last_msg = messages[-1]

        # Check for tool calls (if LLM supports them)
        if supports_tools and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"

        # Check for code blocks in response
        content = getattr(last_msg, "content", "")
        if content and "```python" in content:
            return "execute"

        return END

    async def execute_node(state: AgentStateDict) -> dict[str, Any]:
        """Execute Python code blocks from the last message.

        This node:
        1. Extracts code blocks from the last AI message
        2. Executes each in the session's namespace
        3. Returns execution results as a human message
        """
        messages = state.get("messages", [])
        if not messages:
            return {"messages": []}

        last_msg = messages[-1]
        content = getattr(last_msg, "content", "")

        code_blocks = extract_code_blocks(content)
        if not code_blocks:
            return {"messages": []}

        # Execute code blocks
        results = []
        for i, block in enumerate(code_blocks):
            block = block.strip()
            output, is_error = execute_code(block, session)
            prefix = "Error" if is_error else "Output"
            results.append(f"Code block {i + 1} {prefix}:\n{output}")

        # Create execution report message
        exec_report = "\n\n".join(results)
        exec_msg = HumanMessage(content=f"<execution>\n{exec_report}\n</execution>")

        return {"messages": [exec_msg]}

    # Build the graph
    builder = StateGraph(dict)

    # Add nodes
    builder.add_node("model", model_node)

    if supports_tools:
        # Use LangGraph's prebuilt ToolNode for handling tool calls
        tool_node = ToolNode(all_tools)
        builder.add_node("tools", tool_node)

    builder.add_node("execute", execute_node)

    # Add edges
    builder.add_edge(START, "model")

    # Conditional routing after model
    if supports_tools:
        builder.add_conditional_edges(
            "model",
            route_after_model,
            {
                "tools": "tools",
                "execute": "execute",
                END: END,
            },
        )
        builder.add_edge("tools", "model")
    else:
        # Without tool support, only route between execute and end
        builder.add_conditional_edges(
            "model",
            route_after_model,
            {
                "execute": "execute",
                END: END,
            },
        )

    builder.add_edge("execute", "model")

    return builder.compile()


async def run_graph(
    graph: CompiledStateGraph,
    prompt: str,
    session: SessionState,
    on_token: callable | None = None,
    on_tool_call: callable | None = None,
    on_execution: callable | None = None,
) -> str:
    """Run the graph with streaming support.

    Args:
        graph: Compiled LangGraph to run.
        prompt: User's prompt.
        session: Session state.
        on_token: Callback for streamed tokens.
        on_tool_call: Callback when a tool is called.
        on_execution: Callback with code execution results.

    Returns:
        Final response text.
    """
    # Create initial state
    initial_state = create_initial_state(
        messages=[HumanMessage(content=prompt)],
        namespace=session.globals_ns,
    )

    final_response = ""

    # Stream events from the graph
    async for event in graph.astream_events(initial_state, version="v2"):
        event_type = event.get("event", "")

        # Handle streaming tokens from the model
        if event_type == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                if on_token:
                    on_token(chunk.content)
                final_response += chunk.content

        # Handle tool calls
        elif event_type == "on_tool_start":
            tool_name = event.get("name", "")
            tool_input = event.get("data", {}).get("input", {})
            if on_tool_call:
                on_tool_call(tool_name, tool_input)

        # Handle tool results
        elif event_type == "on_tool_end":
            tool_output = event.get("data", {}).get("output", "")
            # Tool output is handled by the graph flow

    # Get final state to extract the last response
    try:
        final_state = await graph.aget_state(config={})
        messages = final_state.values.get("messages", [])
        if messages:
            last_ai_msg = None
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    last_ai_msg = msg
                    break
            if last_ai_msg:
                final_response = last_ai_msg.content
    except Exception:
        pass  # Use accumulated response from streaming

    return final_response


def create_simple_graph(
    llm: BaseChatModel,
    session: SessionState,
) -> CompiledStateGraph:
    """Create a simplified graph without tool support.

    This is useful for LLMs that don't support tool binding,
    like the ChatClaudeCLI.

    Args:
        llm: The language model to use.
        session: Session state.

    Returns:
        Compiled LangGraph.
    """
    return create_repl_graph(llm=llm, session=session, tools=None)
