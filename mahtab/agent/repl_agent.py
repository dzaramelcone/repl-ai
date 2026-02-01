"""REPL Agent implementation using LangGraph."""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from mahtab.agent.graph import AgentState, build_agent_graph
from mahtab.core.state import SessionState
from mahtab.io.handlers import PromptHandler
from mahtab.llm.claude_cli import ChatClaudeCLI
from mahtab.llm.prompts import build_repl_system_prompt
from mahtab.tools.skills import load_skill_descriptions
from mahtab.ui import StreamingHandler
from mahtab.ui.console import console as default_console


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
        prompt_handler: Handler that accumulates logged messages for context.
        llm: The language model to use (ChatClaudeCLI by default).
        max_turns: Maximum number of turns in the agentic loop.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session: SessionState
    prompt_handler: PromptHandler
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
            var_summary=self.session.summarize_namespace(max_vars=30),
            skills_description=load_skill_descriptions(self.session.skills_dir),
            repl_context=self.prompt_handler.get_context(),
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
            "on_execution": on_execution,
        }

        # Run the graph
        callbacks = [streaming_handler or StreamingHandler(console=default_console, chars_per_second=200.0)]
        result = await self._graph.ainvoke(initial_state, config={"callbacks": callbacks})

        # Update session with final messages
        final_response = result["current_response"]

        # Route structured response through logger
        from mahtab.io import route_response

        route_response(final_response)

        self.session.messages = result["messages"]
        self.session.save_last_session(prompt, final_response)

        return final_response

    def ask_sync(
        self,
        prompt: str,
        streaming_handler=None,
        on_execution=None,
    ) -> str:
        """Synchronous version of ask().

        Args:
            prompt: The user's prompt.
            streaming_handler: Optional callback handler for streaming tokens.
            on_execution: Optional callback for code execution output (output, is_error).

        Returns:
            The final text response from Claude.
        """
        import asyncio

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            self.ask(
                prompt,
                streaming_handler=streaming_handler
                or StreamingHandler(console=default_console, chars_per_second=200.0),
                on_execution=on_execution,
            )
        )
        loop.close()
        return result

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.session.clear_history()

    def set_model(self, model: str) -> None:
        """Switch to a different model.

        Args:
            model: Full model ID (e.g. 'claude-opus-4-20250514').
        """
        self.llm = ChatClaudeCLI(model=model)
        self._graph = build_agent_graph(llm=self.llm, max_turns=self.max_turns)


def create_repl_agent(
    session: SessionState,
    prompt_handler: PromptHandler,
    model: str,
    max_turns: int,
) -> REPLAgent:
    """Create a REPL agent with the given configuration.

    Args:
        session: Session state.
        prompt_handler: Handler for accumulating context.
        model: Claude model to use.
        max_turns: Maximum turns in the agentic loop.

    Returns:
        Configured REPLAgent instance.
    """
    llm = ChatClaudeCLI(model=model)

    return REPLAgent(
        session=session,
        prompt_handler=prompt_handler,
        llm=llm,
        max_turns=max_turns,
    )
