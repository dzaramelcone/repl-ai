# LangGraph Refactor with Reflection Node

## Overview

Convert the manual agentic loop in `REPLAgent.ask()` to a LangGraph `StateGraph` and add a reflection node that evaluates whether code execution satisfied the user's request.

## Requirements

1. **Correctness evaluation**: Did the code execute without errors? If errors, decide whether to fix or surface to user.
2. **Completeness evaluation**: Did the code accomplish what the original prompt asked for?
3. **Auto-continue**: If incomplete, automatically generate more code (up to max_turns limit).

## Graph Structure

```
START → generate → extract_code → [has_code?]
                                      │
                          no ─────────┴─────────── yes
                          │                         │
                          ↓                         ↓
                         END                     execute
                                                    │
                                                    ↓
                                                 reflect
                                                    │
                                        [complete?] ┴ [turn < max?]
                                          │              │
                                   yes ───┤              ├─── no
                                          │              │
                                          ↓              ↓
                                         END            END
                                                  (or generate if yes)
```

## State Schema

```python
class AgentState(TypedDict):
    messages: list[BaseMessage]                    # Conversation history
    system_prompt: str                             # Built once at start
    original_prompt: str                           # User's initial request
    current_response: str                          # Claude's latest response
    code_blocks: list[str]                         # Extracted code
    execution_results: list[tuple[str, bool]]      # (output, is_error) pairs
    turn_count: int                                # For max_turns limit
    session: SessionState                          # Existing session object
```

## Reflection Node

The reflection node makes a focused LLM call to evaluate correctness and completeness.

**Input:** Original prompt, executed code, execution output.

**Output:**
```python
class ReflectionResult(BaseModel):
    is_complete: bool
    reasoning: str           # Brief explanation
    next_action: str | None  # Hint for what's needed if incomplete
```

## File Changes

| File | Change |
|------|--------|
| `mahtab/agent/graph.py` | **NEW** - State, nodes, edges, graph builder |
| `mahtab/llm/prompts.py` | Add `REFLECTION_PROMPT_TEMPLATE`, `build_reflection_prompt()` |
| `mahtab/agent/repl_agent.py` | Replace for-loop with graph invocation, keep interface |
| `pyproject.toml` | Add `langgraph` dependency |

## Node Functions

- `generate_node(state)` - Call LLM, update `current_response`
- `extract_code_node(state)` - Regex extract, update `code_blocks`
- `execute_node(state)` - Run code, update `execution_results` and `session`
- `reflect_node(state)` - Evaluate output, return routing decision

## Edge Conditionals

- `should_execute(state)` - Returns `"execute"` if code_blocks, else `"end"`
- `should_continue(state)` - Returns `"generate"` if incomplete and under max_turns, else `"end"`

## Testing

**Unit tests:**
- `test_generate_node` - mock LLM, verify message construction
- `test_extract_code_node` - single/multiple/none/malformed blocks
- `test_execute_node` - success, errors, namespace mutations
- `test_reflect_node` - complete/incomplete/error judgments

**Integration tests:**
- `test_graph_simple_text_response` - no code path
- `test_graph_code_then_done` - single turn completion
- `test_graph_multi_turn` - reflection triggers continuation
- `test_graph_max_turns_reached` - limit enforcement
- `test_graph_error_recovery` - error handling path

## Dependencies

```
langgraph  # Not currently installed
```
