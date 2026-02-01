"""Session state management using Pydantic models."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, ConfigDict, Field


class UsageStats(BaseModel):
    """Token and cost tracking for the session."""

    total_cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    num_calls: int = 0

    def record(
        self,
        cost: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read: int = 0,
        cache_create: int = 0,
    ) -> None:
        """Record usage from an API call."""
        self.total_cost_usd += cost
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cache_read_input_tokens += cache_read
        self.cache_creation_input_tokens += cache_create
        self.num_calls += 1


class SessionState(BaseModel):
    """Encapsulates all session state with validation.

    Replaces the global variables from the original repl.py:
    - _globals, _locals -> globals_ns, locals_ns
    - _history -> messages
    - _repl_activity -> repl_activity
    - _usage_stats -> usage
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    globals_ns: dict[str, Any] = Field(default_factory=dict)
    locals_ns: dict[str, Any] = Field(default_factory=dict)
    messages: list[BaseMessage] = Field(default_factory=list)
    repl_activity: list[str] = Field(default_factory=list)
    usage: UsageStats = Field(default_factory=UsageStats)

    # Paths for persistence
    skills_dir: Path = Field(default_factory=lambda: Path("~/.mahtab/skills").expanduser())
    last_session_file: Path = Field(default_factory=lambda: Path("~/.mahtab/last_session.json").expanduser())

    def init_namespace(self, globals_dict: dict | None = None, locals_dict: dict | None = None) -> None:
        """Initialize with caller's namespace."""
        self.globals_ns = globals_dict if globals_dict is not None else {}
        self.locals_ns = locals_dict if locals_dict is not None else self.globals_ns

    def add_user_message(self, content: str) -> None:
        """Add a user message to history."""
        self.messages.append(HumanMessage(content=content))

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to history."""
        self.messages.append(AIMessage(content=content))

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.messages.clear()

    def record_activity(self, activity: str) -> None:
        """Record REPL activity for context."""
        self.repl_activity.append(activity)

    def clear_activity(self) -> None:
        """Clear REPL activity buffer."""
        self.repl_activity.clear()

    def get_activity_context(self, max_chars: int = 4000) -> str:
        """Get recent REPL activity, truncated to max_chars."""
        if not self.repl_activity:
            return ""
        text = "\n".join(self.repl_activity)
        if len(text) > max_chars:
            text = "...\n" + text[-max_chars:]
        return text

    def save_last_session(self, user_msg: str, assistant_msg: str) -> None:
        """Save last exchange for next session."""
        self.last_session_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "timestamp": datetime.now().isoformat(),
            "user": user_msg[:2000],  # Truncate to save space
            "assistant": assistant_msg[:2000],
        }
        self.last_session_file.write_text(json.dumps(data, indent=2))

    def load_last_session(self) -> str:
        """Load last session context as XML."""
        if not self.last_session_file.exists():
            return ""
        try:
            data = json.loads(self.last_session_file.read_text())
            return f"""<prior_session timestamp="{data.get("timestamp", "unknown")}">
<human>{data.get("user", "")}</human>
<assistant>{data.get("assistant", "")}</assistant>
</prior_session>"""
        except Exception:
            return ""

    def summarize_namespace(self, max_vars: int = 30) -> str:
        """Summarize variables in the namespace."""
        if not self.globals_ns:
            return "(empty)"

        lines = []
        for name, val in self.globals_ns.items():
            if name.startswith("_"):
                continue
            try:
                typ = type(val).__name__
                if isinstance(val, int | float | str | bool | type(None)):
                    rep = repr(val)[:50]
                elif isinstance(val, list | dict | set | tuple):
                    rep = f"{typ} with {len(val)} items"
                else:
                    rep = typ
                lines.append(f"  {name}: {rep}")
            except Exception:
                lines.append(f"  {name}: <unknown>")

        return "\n".join(lines[:max_vars]) or "(no user variables)"

    def ensure_skills_dir(self) -> None:
        """Ensure the skills directory exists."""
        self.skills_dir.mkdir(parents=True, exist_ok=True)
