"""Custom LangChain chat model that calls Claude via the CLI subprocess."""

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field


class ChatClaudeCLI(BaseChatModel):
    """LangChain chat model that calls Claude via the CLI subprocess.

    Maintains existing behavior of shelling out to `claude` command
    while providing full LangChain compatibility.

    Example:
        ```python
        from mahtab.llm.claude_cli import ChatClaudeCLI

        model = ChatClaudeCLI(model="claude-opus-4-20250514")
        response = model.invoke("Hello!")
        ```
    """

    model: str = Field(default="claude-opus-4-20250514", description="Claude model identifier")
    max_tokens: int = Field(default=4096, description="Maximum tokens to generate")
    cwd: str = Field(default="/tmp", description="Working directory for subprocess")
    setting_sources: str = Field(default="", description="Setting sources for Claude CLI")

    @property
    def _llm_type(self) -> str:
        return "claude-cli"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "max_tokens": self.max_tokens}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous generation via CLI subprocess."""
        # Run async version in event loop
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(self._agenerate(messages, stop, run_manager=None, **kwargs))
        loop.close()
        return result

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generation via CLI subprocess."""
        prompt = self._messages_to_xml(messages)
        system = self._extract_system_message(messages)
        if "system" in kwargs:
            system = kwargs["system"]

        result, usage = await self._call_claude_async(prompt, system, run_manager)

        generation = ChatGeneration(
            message=AIMessage(content=result),
            generation_info={"usage": usage},
        )
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Synchronous streaming via CLI subprocess with stream-json."""
        # Run async generator in event loop
        loop = asyncio.new_event_loop()
        async_gen = self._astream(messages, stop, run_manager=None, **kwargs)
        while True:
            try:
                chunk = loop.run_until_complete(async_gen.__anext__())
                yield chunk
            except StopAsyncIteration:
                break
        loop.close()

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Async streaming via CLI subprocess with stream-json output."""
        prompt = self._messages_to_xml(messages)
        system = self._extract_system_message(messages)
        if "system" in kwargs:
            system = kwargs["system"]

        cmd = [
            "claude",
            "-p",
            prompt,
            "--model",
            self.model,
            "--output-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
        ]

        if system:
            cmd.extend(["--system-prompt", system])

        cmd.extend(["--setting-sources", self.setting_sources])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.cwd,
        )

        async for line in proc.stdout:
            line_str = line.decode().strip()
            if not line_str:
                continue
            data = json.loads(line_str)

            # Handle streaming deltas
            if data["type"] == "stream_event":
                event = data["event"]
                if event["type"] == "content_block_delta":
                    delta = event["delta"]
                    if delta["type"] == "text_delta":
                        text = delta["text"]
                        if text:
                            yield ChatGenerationChunk(message=AIMessageChunk(content=text))

            # Handle final result for usage stats
            elif data["type"] == "result":
                usage = data["usage"]
                # Put usage in response_metadata so it's accessible via .astream()
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        response_metadata={
                            "usage": usage,
                            "total_cost_usd": data["total_cost_usd"],
                        },
                    ),
                )

        await proc.wait()

        if proc.returncode != 0:
            stderr = await proc.stderr.read()
            raise RuntimeError(f"claude CLI failed: {stderr.decode()}")

    async def _call_claude_async(
        self, prompt: str, system: str, run_manager: AsyncCallbackManagerForLLMRun | None = None
    ) -> tuple[str, dict]:
        """Call Claude CLI and return full response with usage stats."""
        cmd = [
            "claude",
            "-p",
            prompt,
            "--model",
            self.model,
            "--output-format",
            "stream-json",
            "--include-partial-messages",
            "--verbose",
        ]

        if system:
            cmd.extend(["--system-prompt", system])

        cmd.extend(["--setting-sources", self.setting_sources])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.cwd,
        )

        full_response = ""
        usage: dict = {}

        async for line in proc.stdout:
            line_str = line.decode().strip()
            if not line_str:
                continue
            data = json.loads(line_str)

            if data["type"] == "stream_event":
                event = data["event"]
                if event["type"] == "content_block_delta":
                    delta = event["delta"]
                    if delta["type"] == "text_delta":
                        text = delta["text"]
                        full_response += text
                        if text and run_manager:
                            await run_manager.on_llm_new_token(text)

            elif data["type"] == "result":
                usage = {
                    **data["usage"],
                    "total_cost_usd": data["total_cost_usd"],
                }
                if not full_response:
                    full_response = data["result"]

        await proc.wait()

        if proc.returncode != 0:
            stderr = await proc.stderr.read()
            raise RuntimeError(f"claude CLI failed: {stderr.decode()}")

        return full_response.strip(), usage

    def _messages_to_xml(self, messages: list[BaseMessage]) -> str:
        """Convert LangChain messages to XML format for claude CLI."""
        parts = ["<conversation>"]
        for msg in messages:
            if msg.type == "human":
                parts.append(f"<human>{msg.content}</human>")
            elif msg.type == "ai":
                parts.append(f"<assistant>{msg.content}</assistant>")
            # System messages handled separately via --system-prompt
        parts.append("</conversation>")
        return "\n".join(parts)

    def _extract_system_message(self, messages: list[BaseMessage]) -> str:
        """Extract system message content from messages list."""
        for msg in messages:
            if msg.type == "system":
                return str(msg.content)
        return ""
