---
name: repl-tools
description: Python REPL tool reference
---

<skill>
<assistant-repl-in># Python code executes in the shared namespace</assistant-repl-in>
<assistant-chat>Natural language responses</assistant-chat>

<rule>Limit commands to ONE LINE. If complex, STOP - define a reusable function first, then call it in one line.</rule>
<rule>ALWAYS BE CONCISE. Short responses. 1-2 lines max.</rule>

<read><assistant-repl-in>read(path: str, start: int = 1, end: int | None = None) -> str</assistant-repl-in></read>
<edit><assistant-repl-in>edit(path: str, old: str, new: str) -> None</assistant-repl-in></edit>
<create><assistant-repl-in>create(name: str, content: str) -> None</assistant-repl-in></create>
<peek><assistant-repl-in>peek(text: str, n: int = 2000) -> str</assistant-repl-in></peek>
<grep><assistant-repl-in>grep(text: str, pattern: str) -> list[str]</assistant-repl-in></grep>
<partition><assistant-repl-in>partition(text: str, n: int = 10) -> list[str]</assistant-repl-in></partition>
<rlm><assistant-repl-in>rlm(query: str, context: str) -> str</assistant-repl-in></rlm>
<load_claude_sessions><assistant-repl-in>load_claude_sessions() -> str</assistant-repl-in></load_claude_sessions>
</skill>
