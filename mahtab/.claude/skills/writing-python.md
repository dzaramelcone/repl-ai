---
name: writing-python
description: Python code style guidelines
---

<skill>
- You ALWAYS write fail-fast code.
- You ALWAYS write minimal code that can be understood at a glance by a beginner.
- You ALWAYS write code that is PEP8 friendly.
- You ALWAYS write code that looks like it could be in the official Python documentation.
- You ALWAYS inline aggressively and keep your callstacks and nestings flat.
- You NEVER use None or None checking patterns. When None cannot be avoided, you check it as early as possible in the callstack.
- You NEVER write try excepts, especially not overbroad exception catching - a cardinal sin.
- You NEVER write defensive Python code. Fail fast is your mantra.
- You ALWAYS prefer integration tests.
- ALWAYS BE CONCISE. Short responses. 1-2 lines max.
- Whenever your code accumulates complexity, you STOP, step back and figure out the callstack. Then you slow down, @PLAN a way to clean it up to get it back in compliance, and once complete, continue where you left off.

@https://docs.python.org/3/
@https://peps.python.org/
@~/references/effective-python.pdf
</skill>
