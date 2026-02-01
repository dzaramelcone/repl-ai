"""REPL Agent implementation using LangGraph."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console

from mahtab.agent.graph import create_repl_graph, extract_code_blocks
from mahtab.agent.state import create_initial_state
from mahtab.core.executor import execute_code
from mahtab.core.state import SessionState
from mahtab.llm.claude_cli import ChatClaudeCLI
from mahtab.llm.prompts import build_repl_system_prompt
from mahtab.tools.skills import load_skill_descriptions

if TYPE_CHECKING:
    from collections.abc import Callable


def get_llm(model: str = "claude-sonnet-4-20250514", use_api: bool | None = None) -> BaseChatModel:
    """Get the appropriate LLM based on configuration.

    If ANTHROPIC_API_KEY is set, uses ChatAnthropic for native tool calling.
    Otherwise falls back to ChatClaudeCLI (subprocess-based).

    Args:
        model: Model name to use.
        use_api: Force API usage (True) or CLI (False). If None, auto-detect.

    Returns:
        Configured LLM instance.
    """
    # Auto-detect based on API key presence
    if use_api is None:
        use_api = bool(os.environ.get("ANTHROPIC_API_KEY"))

    if use_api:
        try:
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(model=model)
        except ImportError:
            # Fall back to CLI if langchain-anthropic not installed
            pass

    return ChatClaudeCLI(model=model)


class REPLAgent(BaseModel):
    """Agent that manages conversation with Claude and code execution.

    This agent uses LangGraph for the agentic loop:
    1. User sends a prompt
    2. Graph routes to model node
    3. Model responds with text, tool calls, or code blocks
    4. Tool calls are executed via ToolNode
    5. Code blocks are executed in the namespace
    6. Results sent back to model
    7. Loop continues until model responds with just text

    Attributes:
        session: The session state containing namespace and history.
        llm: The language model to use.
        console: Rich console for output (optional, for streaming).
        max_turns: Maximum number of turns in the agentic loop.
        graph: The compiled LangGraph (created lazily).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session: SessionState
    llm: BaseChatModel = Field(default_factory=lambda: get_llm())
    console: Console | None = None
    max_turns: int = 5
    graph: CompiledStateGraph | None = Field(default=None, exclude=True)

    def _ensure_graph(self) -> CompiledStateGraph:
        """Ensure the graph is created and return it."""
        if self.graph is None:
            self.graph = create_repl_graph(
                llm=self.llm,
                session=self.session,
                max_turns=self.max_turns,
            )
        return self.graph

    async def ask(
        self,
        prompt: str,
        on_token: Callable[[str], None] | None = None,
        on_code_block: Callable[[str, int], None] | None = None,
        on_execution: Callable[[str, bool, int], None] | None = None,
        on_tool_call: Callable[[str, dict], None] | None = None,
    ) -> str:
        """Send a prompt to Claude and handle the conversation.

        This method uses LangGraph with streaming events for real-time output.

        Args:
            prompt: The user's prompt.
            on_token: Callback for each token streamed (for typewriter effect).
            on_code_block: Callback when a code block is detected.
            on_execution: Callback with execution results.
            on_tool_call: Callback when a tool is called (skill loading, etc.).

        Returns:
            The final text response from Claude.
        """
        graph = self._ensure_graph()

        # Add user message to session history
        self.session.add_user_message(prompt)

        # Create initial state with the user's message
        initial_state = create_initial_state(
            messages=[HumanMessage(content=prompt)],
            namespace=self.session.globals_ns,
        )

        final_response = ""
        current_response = ""

        # Stream events from the graph
        async for event in graph.astream_events(initial_state, version="v2"):
            event_type = event.get("event", "")

            # Handle streaming tokens from the model
            if event_type == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    token = chunk.content
                    current_response += token
                    if on_token:
                        on_token(token)

                    # Track usage from response metadata
                    metadata = getattr(chunk, "response_metadata", {}) or {}
                    usage = metadata.get("usage", {})
                    if usage:
                        self.session.usage.record(
                            cost=metadata.get("total_cost_usd", 0),
                            input_tokens=usage.get("input_tokens", 0),
                            output_tokens=usage.get("output_tokens", 0),
                            cache_read=usage.get("cache_read_input_tokens", 0),
                            cache_create=usage.get("cache_creation_input_tokens", 0),
                        )

            # Handle tool calls (skill loading)
            elif event_type == "on_tool_start":
                tool_name = event.get("name", "")
                tool_input = event.get("data", {}).get("input", {})
                if on_tool_call:
                    on_tool_call(tool_name, tool_input)

            # Handle model completion (to detect code blocks)
            elif event_type == "on_chat_model_end":
                if current_response:
                    # Check for code blocks in the completed response
                    code_blocks = extract_code_blocks(current_response)
                    if code_blocks and on_code_block:
                        for i, block in enumerate(code_blocks):
                            on_code_block(block.strip(), i)

                    final_response = current_response
                    current_response = ""

            # Handle code execution results
            elif event_type == "on_chain_end":
                node_name = event.get("name", "")
                if node_name == "execute":
                    output = event.get("data", {}).get("output", {})
                    messages = output.get("messages", [])
                    if messages and on_execution:
                        # Parse execution results from the message
                        exec_content = messages[0].content if messages else ""
                        if "<execution>" in exec_content:
                            # Extract individual results
                            on_execution(exec_content, False, 0)

        # Save final response to session
        if final_response:
            self.session.add_assistant_message(final_response)
            self.session.save_last_session(prompt, final_response)

        return final_response

    async def ask_legacy(
        self,
        prompt: str,
        on_token: Callable[[str], None] | None = None,
        on_code_block: Callable[[str, int], None] | None = None,
        on_execution: Callable[[str, bool, int], None] | None = None,
    ) -> str:
        """Legacy ask implementation without LangGraph.

        This is the original implementation that uses direct streaming
        and regex-based code extraction. Kept for backwards compatibility.

        Args:
            prompt: The user's prompt.
            on_token: Callback for each token streamed.
            on_code_block: Callback when a code block is detected.
            on_execution: Callback with execution results.

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

        # Add user message to history
        self.session.add_user_message(prompt)

        for _ in range(self.max_turns):
            # Build messages for LLM
            messages = [SystemMessage(content=system_prompt), *self.session.messages]

            # Stream response from Claude
            response_text = ""
            async for chunk in self.llm.astream(messages):
                token = chunk.content
                if token:
                    response_text += token
                    if on_token:
                        on_token(token)

                # Track usage if present
                metadata = getattr(chunk, "response_metadata", {}) or {}
                usage = metadata.get("usage", {})
                if usage:
                    self.session.usage.record(
                        cost=metadata.get("total_cost_usd", 0),
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                        cache_read=usage.get("cache_read_input_tokens", 0),
                        cache_create=usage.get("cache_creation_input_tokens", 0),
                    )

            # Extract code blocks
            code_blocks = re.findall(r"```python\n(.*?)```", response_text, re.DOTALL)

            if not code_blocks:
                # No code, just a text response - we're done
                self.session.add_assistant_message(response_text)
                self.session.save_last_session(prompt, response_text)
                return response_text

            # Execute code blocks and collect output
            outputs = []
            for i, block in enumerate(code_blocks):
                block = block.strip()
                if on_code_block:
                    on_code_block(block, i)

                output, is_error = execute_code(block, self.session)
                outputs.append((output, is_error))

                if on_execution:
                    on_execution(output, is_error, i)

            # Add assistant response and execution results to history
            self.session.add_assistant_message(response_text)

            exec_report = "\n\n".join(
                f"Code block {i + 1} output:\n{out}" for i, (out, _) in enumerate(outputs)
            )
            self.session.messages.append(HumanMessage(content=f"<execution>\n{exec_report}\n</execution>"))

        # Max turns reached
        return "(max turns reached)"

    def ask_sync(
        self,
        prompt: str,
        on_token: Callable[[str], None] | None = None,
        on_code_block: Callable[[str, int], None] | None = None,
        on_execution: Callable[[str, bool, int], None] | None = None,
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
        # Reset graph to pick up fresh history
        self.graph = None


def create_repl_agent(
    session: SessionState | None = None,
    model: str = "claude-sonnet-4-20250514",
    console: Console | None = None,
    max_turns: int = 5,
    use_api: bool | None = None,
) -> REPLAgent:
    """Create a REPL agent with the given configuration.

    Args:
        session: Session state. If None, creates a new one.
        model: Claude model to use.
        console: Rich console for output.
        max_turns: Maximum turns in the agentic loop.
        use_api: Force API usage (True) or CLI (False). If None, auto-detect.

    Returns:
        Configured REPLAgent instance.
    """
    if session is None:
        session = SessionState()

    llm = get_llm(model=model, use_api=use_api)

    return REPLAgent(
        session=session,
        llm=llm,
        console=console,
        max_turns=max_turns,
    )
