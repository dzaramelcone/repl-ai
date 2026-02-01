"""REPL Agent implementation using LangChain components."""

from __future__ import annotations

import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console

from mahtab.core.executor import execute_code
from mahtab.core.state import SessionState
from mahtab.llm.claude_cli import ChatClaudeCLI
from mahtab.llm.prompts import build_repl_system_prompt
from mahtab.tools.skills import load_skill_descriptions


class REPLAgent(BaseModel):
    """Agent that manages conversation with Claude and code execution.

    This agent implements the agentic loop where:
    1. User sends a prompt
    2. Claude responds with text and optionally code blocks
    3. Code blocks are extracted and executed
    4. Execution results are sent back to Claude
    5. Loop continues until Claude responds without code blocks

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
            on_token: Callback for each token streamed (for typewriter effect).
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
                if chunk.generation_info:
                    usage = chunk.generation_info.get("usage", {})
                    if usage:
                        self.session.usage.record(
                            cost=chunk.generation_info.get("total_cost_usd", 0),
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
