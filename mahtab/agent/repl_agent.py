"""REPL Agent implementation using LangGraph."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
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

    _graph: Any = PrivateAttr(default=None)

    def model_post_init(self, _) -> None:
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
            return loop.run_until_complete(self.ask(prompt, on_token, on_code_block, on_execution))
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
