"""REPL Agent implementation using LangGraph."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from mahtab.agent.graph import AgentState, build_agent_graph
from mahtab.llm.claude_cli import ChatClaudeCLI
from mahtab.llm.prompts import build_repl_system_prompt
from mahtab.ui.handlers import SessionStreamingHandler


class REPLAgent(BaseModel):
    """Agent that manages conversation with Claude and code execution.

    Uses a LangGraph StateGraph for the agentic loop:
    1. Generate: Claude responds to the prompt
    2. Extract: Parse code blocks from response
    3. Execute: Run code blocks in session namespace
    4. Reflect: Evaluate if task is complete
    5. Loop back to Generate if incomplete

    Attributes:
        session: The session containing namespace and history.
        llm: The language model to use (ChatClaudeCLI by default).
        max_turns: Maximum number of turns in the agentic loop.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session: Any  # Session, but Any to avoid circular import
    llm: BaseChatModel = Field(default_factory=ChatClaudeCLI)
    max_turns: int = 5

    _graph: Any = PrivateAttr(default=None)

    def model_post_init(self, _) -> None:
        """Build the graph after model initialization."""
        self._graph = build_agent_graph(llm=self.llm, max_turns=self.max_turns)

    async def ask(
        self,
        prompt: str,
        streaming_handler=None,
        on_execution=None,
    ) -> str:
        """Send a prompt to Claude and handle the conversation.

        Args:
            prompt: The user's prompt.
            streaming_handler: Optional callback handler for streaming tokens.
            on_execution: Optional callback for code execution output (output, is_error).

        Returns:
            The final text response from Claude.
        """
        # Build system prompt with current context
        system_prompt = build_repl_system_prompt(
            var_summary=self.session.summarize_namespace(),
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
            "on_execution": on_execution,
        }

        # Run the graph
        callbacks = [streaming_handler or SessionStreamingHandler(self.session)]
        result = await self._graph.ainvoke(initial_state, config={"callbacks": callbacks})

        # Update session with final messages
        final_response = result["current_response"]
        self.session.messages = result["messages"]

        return final_response

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.session.messages.clear()
