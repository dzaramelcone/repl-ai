# Mahtab System Architecture

**Version:** 0.2.0  
**Description:** AI-powered shared Python REPL with Claude integration

---

## Table of Contents

1. [Overview](#overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Module Structure](#module-structure)
4. [Core Components](#core-components)
5. [Data Flow](#data-flow)
6. [Key Concepts](#key-concepts)
7. [Component Details](#component-details)

---

## Overview

Mahtab is a collaborative Python REPL environment where Claude (an AI assistant) can:
- Execute code directly in the user's namespace
- Inspect and modify variables
- Read and edit files
- Explore large text contexts using recursive search strategies (RLM)

The system creates a **shared namespace** between the user and Claude, enabling true collaboration where both parties can see and modify the same Python state.

### Key Dependencies

```
langchain-core >= 0.3.0       # LLM abstractions and message types
langchain >= 0.3.0            # Tool decorators and chains
langgraph >= 0.2.0            # Graph-based agent workflows
pydantic >= 2.0.0             # Data validation and state models
rich >= 13.0.0                # Terminal UI (panels, syntax highlighting)
```

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────┐    ┌─────────────────────────────────────────┐ │
│  │   Interactive REPL      │    │         Modal REPL                      │ │
│  │   (interactive.py)      │    │         (modal.py)                      │ │
│  │                         │    │                                         │ │
│  │   • Dynamic prompt      │    │   • Backtick mode switching             │ │
│  │   • ask() function      │    │   • Python ↔ Ask modes                  │ │
│  │   • Inline with Python  │    │   • Tab completion                      │ │
│  └───────────┬─────────────┘    └───────────────┬─────────────────────────┘ │
└──────────────┼──────────────────────────────────┼───────────────────────────┘
               │                                  │
               ▼                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AGENT LAYER                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    REPLAgent (repl_agent.py)                          │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │                    LANGGRAPH AGENT FLOW                        │  │   │
│  │  │                                                                │  │   │
│  │  │   ┌─────────┐                                                  │  │   │
│  │  │   │  START  │                                                  │  │   │
│  │  │   └────┬────┘                                                  │  │   │
│  │  │        │                                                       │  │   │
│  │  │        ▼                                                       │  │   │
│  │  │   ┌─────────────┐                                              │  │   │
│  │  │   │ Model Node  │──────────────────────────────────────────┐   │  │   │
│  │  │   │ (LLM Call)  │                                          │   │  │   │
│  │  │   └──────┬──────┘                                          │   │  │   │
│  │  │          │                                                 │   │  │   │
│  │  │          ▼                                                 │   │  │   │
│  │  │   ┌──────────────┐                                         │   │  │   │
│  │  │   │   Router     │                                         │   │  │   │
│  │  │   └──────┬───────┘                                         │   │  │   │
│  │  │          │                                                 │   │  │   │
│  │  │    ┌─────┴─────┐                                           │   │  │   │
│  │  │    │           │                                           │   │  │   │
│  │  │    ▼           ▼                                           │   │  │   │
│  │  │ ┌───────┐ ┌─────────┐                                      │   │  │   │
│  │  │ │Execute│ │   END   │                                      │   │  │   │
│  │  │ │ Node  │ └─────────┘                                      │   │  │   │
│  │  │ └───┬───┘                                                  │   │  │   │
│  │  │     │                                                      │   │  │   │
│  │  │     └──────────────────────────────────────────────────────┘   │  │   │
│  │  │                 (loop back to Model Node)                      │  │   │
│  │  └────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌────────────────────┐   ┌────────────────────────────────────────────┐    │
│  │  AgentState        │   │  LangGraph (graph.py)                       │    │
│  │  (agent/state.py)  │   │                                             │    │
│  │                    │   │  • create_repl_graph() - builds StateGraph  │    │
│  │  • messages        │   │  • Model node with conditional routing      │    │
│  │  • namespace       │   │  • Execute node for code blocks             │    │
│  │  • pending_code    │   │  • Streaming via astream_events()           │    │
│  └────────────────────┘   └────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CORE LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐   ┌────────────────┐   ┌────────────────────────────┐   │
│  │  SessionState  │   │   Executor     │   │    Namespace Manager       │   │
│  │   (state.py)   │   │ (executor.py)  │   │    (namespace.py)          │   │
│  │                │   │                │   │                            │   │
│  │ • globals_ns   │   │ • execute_code │   │ • init_namespace           │   │
│  │ • locals_ns    │   │ • LimitedOutput│   │ • reload_module_if_imported│   │
│  │ • messages     │   │ • sandboxed    │   │ • ensure_cwd_in_path       │   │
│  │ • usage stats  │   │   execution    │   │                            │   │
│  └────────────────┘   └────────────────┘   └────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LLM LAYER                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────┐   ┌───────────────────────────┐    │
│  │         ChatClaudeCLI               │   │        Prompts            │    │
│  │        (claude_cli.py)              │   │       (prompts.py)        │    │
│  │                                     │   │                           │    │
│  │  • CLI subprocess to `claude`       │   │ • REPL system prompt      │    │
│  │  • No API key needed                │   │ • RLM system prompt       │    │
│  │  • stream-json format               │   │ • Context-aware building  │    │
│  │  • Full LangChain compatibility     │   │                           │    │
│  └─────────────────────────────────────┘   └───────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TOOLS LAYER                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐   ┌───────────────┐   ┌────────────────────────────────┐ │
│  │  File Tools   │   │  Text Tools   │   │       Skills                   │ │
│  │  (files.py)   │   │  (text.py)    │   │      (skills.py)               │ │
│  │               │   │               │   │                                │ │
│  │ • read_file   │   │ • peek        │   │ • load_skill_descriptions      │ │
│  │ • edit_file   │   │ • grep        │   │ • load_skill_content           │ │
│  │ • create_file │   │ • partition   │   │ • load_claude_sessions         │ │
│  │ • open_editor │   │ • *_raw       │   │                                │ │
│  └───────────────┘   └───────────────┘   └────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     RECURSIVE LANGUAGE MODEL (RLM)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        rlm() Function (search.py)                     │   │
│  │                                                                       │   │
│  │   LLM writes code to explore large contexts recursively               │   │
│  │                                                                       │   │
│  │   ┌─────────────────────────────────────────────────────────────┐    │   │
│  │   │                    RLM EXECUTION LOOP                        │    │   │
│  │   │                                                              │    │   │
│  │   │   Query + Context ──▶ LLM generates Python code              │    │   │
│  │   │                              │                               │    │   │
│  │   │                              ▼                               │    │   │
│  │   │                     Execute in sandbox                       │    │   │
│  │   │                     (limited builtins)                       │    │   │
│  │   │                              │                               │    │   │
│  │   │              ┌───────────────┼────────────────┐              │    │   │
│  │   │              │               │                │              │    │   │
│  │   │              ▼               ▼                ▼              │    │   │
│  │   │         peek()          grep()         partition()           │    │   │
│  │   │         rlm()           FINAL()                              │    │   │
│  │   │        (recurse)       (terminate)                           │    │   │
│  │   │                                                              │    │   │
│  │   └─────────────────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              UI LAYER                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐   ┌────────────────┐   ┌──────────────────────────────┐ │
│  │    Console     │   │  Streaming     │   │         Panels               │ │
│  │  (console.py)  │   │ (streaming.py) │   │       (panels.py)            │ │
│  │                │   │                │   │                              │ │
│  │ • Rich Console │   │ • Typewriter   │   │ • print_code_panel           │ │
│  │   singleton    │   │   effect       │   │ • print_output_panel         │ │
│  │                │   │ • Code panel   │   │ • print_banner               │ │
│  │                │   │   detection    │   │ • print_usage_panel          │ │
│  │                │   │ • Spinner      │   │                              │ │
│  └────────────────┘   └────────────────┘   └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Structure

```
mahtab/
├── __init__.py           # Package exports: SessionState, UsageStats
├── __main__.py           # Entry point: python -m mahtab
│
├── agent/                # Agent logic and LangGraph
│   ├── __init__.py
│   ├── graph.py          # LangGraph StateGraph: create_repl_graph(), run_graph()
│   ├── repl_agent.py     # REPLAgent class, get_llm()
│   └── state.py          # AgentState for LangGraph: messages, namespace
│
├── core/                 # Core infrastructure
│   ├── __init__.py
│   ├── executor.py       # Code execution with output limiting
│   ├── namespace.py      # Namespace management and module reloading
│   └── state.py          # SessionState and UsageStats Pydantic models
│
├── llm/                  # LLM integration
│   ├── __init__.py
│   ├── claude_cli.py     # ChatClaudeCLI - LangChain wrapper for claude CLI
│   └── prompts.py        # System prompts for REPL and RLM
│
├── repl/                 # REPL implementations
│   ├── __init__.py
│   ├── interactive.py    # Standard interactive REPL with ask()
│   └── modal.py          # Modal REPL with backtick mode switching
│
├── rlm/                  # Recursive Language Model
│   ├── __init__.py
│   └── search.py         # RLM algorithm implementation
│
├── tools/                # LangChain tools
│   ├── __init__.py
│   ├── files.py          # File operations (read, edit, create)
│   ├── skills.py         # Skill management
│   └── text.py           # Text exploration (peek, grep, partition)
│
└── ui/                   # Terminal UI
    ├── __init__.py
    ├── console.py        # Rich console singleton
    ├── panels.py         # Panel rendering utilities
    └── streaming.py      # Streaming output handler
```

---

## Core Components

### 1. SessionState (state.py)

The central state container that holds everything for a REPL session:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SessionState                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │   globals_ns     │  │    locals_ns     │  │    messages      │   │
│  │   dict[str,Any]  │  │   dict[str,Any]  │  │ list[BaseMessage]│   │
│  │                  │  │                  │  │                  │   │
│  │ User's global    │  │ User's local     │  │ Conversation     │   │
│  │ namespace        │  │ namespace        │  │ history          │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘   │
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │  repl_activity   │  │      usage       │  │    skills_dir    │   │
│  │    list[str]     │  │   UsageStats     │  │      Path        │   │
│  │                  │  │                  │  │                  │   │
│  │ Recent user      │  │ Token/cost       │  │ ~/.mahtab/skills │   │
│  │ commands         │  │ tracking         │  │                  │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘   │
│                                                                      │
│  Methods:                                                            │
│  • init_namespace(globals, locals)  - Initialize with caller's ns   │
│  • add_user_message(content)        - Add human message to history  │
│  • add_assistant_message(content)   - Add AI message to history     │
│  • summarize_namespace(max_vars)    - Describe current variables    │
│  • save_last_session(user, asst)    - Persist last exchange         │
│  • load_last_session()              - Load previous session context │
│  • get_activity_context(max_chars)  - Get recent REPL activity      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2. REPLAgent and LangGraph (repl_agent.py, graph.py)

The agent uses **LangGraph** for the agentic loop with conditional routing:

```
┌─────────────────────────────────────────────────────────────────┐
│                        REPLAgent                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Attributes:                                                     │
│  • session: SessionState      - The shared state                │
│  • llm: BaseChatModel         - ChatClaudeCLI (via subprocess)  │
│  • max_turns: int = 5         - Safety limit for loop           │
│  • graph: CompiledStateGraph  - LangGraph (created lazily)      │
│                                                                  │
│  Methods:                                                        │
│  • ask(prompt, callbacks...) - LangGraph-based with streaming   │
│  • ask_legacy(prompt, ...)   - Original regex-based loop        │
│  • ask_sync(prompt, ...)     - Synchronous wrapper              │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                    LANGGRAPH FLOW (graph.py)                    │
│                                                                  │
│     ┌──────────────────────────────────────────────────────┐    │
│     │  create_repl_graph(llm, session, max_turns)          │    │
│     │                                                       │    │
│     │  Builds a StateGraph with:                            │    │
│     │  • Model node: calls LLM with fresh system prompt     │    │
│     │  • Execute node: runs code blocks in namespace        │    │
│     └──────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│     ┌──────────────────────────────────────────────────────┐    │
│     │              GRAPH ROUTING LOGIC                      │    │
│     │                                                       │    │
│     │  route_after_model(state) decides:                    │    │
│     │    • "execute" → if response has ```python blocks     │    │
│     │    • END       → otherwise (conversation complete)    │    │
│     └──────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│     ┌──────────────────────────────────────────────────────┐    │
│     │  Streaming via graph.astream_events(version="v2")     │    │
│     │                                                       │    │
│     │  Events handled:                                      │    │
│     │    • on_chat_model_stream → tokens for typewriter     │    │
│     │    • on_chat_model_end → code block detection         │    │
│     │    • on_chain_end → execution results                 │    │
│     └──────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3. LLM Layer (claude_cli.py)

The LLM backend communicates with Claude via the CLI subprocess:

```
┌─────────────────────────────────────────────────────────────────┐
│                      ChatClaudeCLI                               │
│                     (claude_cli.py)                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Uses Claude Code CLI for authentication (no API key needed)     │
│                                                                  │
│  Configuration:                                                  │
│  • model: str = "claude-opus-4-20250514"                         │
│  • max_tokens: int = 4096                                        │
│  • cwd: str = "/tmp"      (subprocess working directory)         │
│  • setting_sources: str   (CLI configuration)                    │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Message Flow:                                                   │
│                                                                  │
│   LangChain Messages          Claude CLI                         │
│   ─────────────────          ──────────                          │
│                                                                  │
│   [SystemMessage,     ──▶    claude -p "<conversation>           │
│    HumanMessage,              <human>...</human>                 │
│    AIMessage, ...]            <assistant>...</assistant>         │
│                               </conversation>"                   │
│                               --system-prompt "..."              │
│                               --output-format stream-json        │
│                               --verbose --include-partial-messages
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Streaming Output (stream-json format):                          │
│                                                                  │
│   {"type": "stream_event",                                       │
│    "event": {                                                    │
│      "type": "content_block_delta",                              │
│      "delta": {"type": "text_delta", "text": "..."}              │
│    }}                                                            │
│                                                                  │
│   {"type": "result",                                             │
│    "usage": {"input_tokens": N, "output_tokens": M, ...},        │
│    "total_cost_usd": 0.XX}                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4. Code Executor (executor.py)

Safe code execution with output limiting:

```
┌─────────────────────────────────────────────────────────────────┐
│                        execute_code()                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: code (str), session (SessionState)                       │
│  Output: (output_string, is_error)                               │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    EXECUTION FLOW                          │  │
│  │                                                            │  │
│  │  1. Redirect stdout to LimitedOutput (10KB limit)          │  │
│  │                                                            │  │
│  │  2. Try eval(code) first (for expressions)                 │  │
│  │     • If result is not None, print repr(result)            │  │
│  │                                                            │  │
│  │  3. If SyntaxError, fall back to exec(code)                │  │
│  │     • For statements (assignments, loops, etc.)            │  │
│  │                                                            │  │
│  │  4. Execute in session's namespace:                        │  │
│  │     • globals_ns: session.globals_ns                       │  │
│  │     • locals_ns: session.locals_ns                         │  │
│  │                                                            │  │
│  │  5. Return captured output or error message                │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  LimitedOutput class:                                            │
│  • Raises RuntimeError if output exceeds 10KB                    │
│  • Prevents Claude from generating infinite output               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### User Interaction Flow

```
User types: ask("explain this code")
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ask() in interactive.py                       │
│                                                                  │
│  Calls agent.ask() with callbacks:                               │
│  • on_token → StreamingHandler.process_token()                   │
│  • on_execution → print_output_panel()                           │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│            REPLAgent.ask() - LangGraph Execution                 │
└─────────────────────────────────────────────────────────────────┘
                    │
                    │ 1. Create initial state with user message
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│         create_initial_state(messages=[HumanMessage(prompt)])    │
└─────────────────────────────────────────────────────────────────┘
                    │
                    │ 2. Stream events from LangGraph
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│        graph.astream_events(initial_state, version="v2")         │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              MODEL NODE                                      ││
│  │                                                              ││
│  │  1. Build fresh system prompt:                               ││
│  │     • summarize_namespace() → variable descriptions          ││
│  │     • load_skill_descriptions() → available skills           ││
│  │     • get_activity_context() → recent REPL commands          ││
│  │     • load_last_session() → prior conversation               ││
│  │                                                              ││
│  │  2. Call LLM (ChatClaudeCLI via subprocess)                 ││
│  │                                                              ││
│  │  3. Stream response tokens via on_chat_model_stream          ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                    │
                    │ 3. Route based on response
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│               route_after_model(state) decides:                  │
│                                                                  │
│         ┌─────────────────┐              ┌───────────┐          │
│         │  Code blocks?   │              │  Neither  │          │
│         │  ```python      │              │           │          │
│         └────────┬────────┘              └─────┬─────┘          │
│                  │                             │                │
│                  ▼                             ▼                │
│         ┌─────────────────┐              ┌───────────┐          │
│         │  Execute Node   │              │    END    │          │
│         │                 │              │           │          │
│         └────────┬────────┘              └───────────┘          │
│                  │                                              │
│                  │     Execute code in session.globals_ns       │
│                  │     Shared namespace with user!              │
│                  │                                              │
│                  ▼                                              │
│         Loop back to Model Node                                 │
│         (until END or max_turns)                                │
└─────────────────────────────────────────────────────────────────┘
                    │
                    │ 4. Process streaming events
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│              StreamingHandler.process_token()                    │
│                                                                  │
│  • on_chat_model_stream → tokens for typewriter effect           │
│  • Spinner → first token stops spinner                           │
│  • Text → direct output                                          │
│  • ```python\n → switch to live code panel                       │
│  • ``` → finalize code panel                                     │
└─────────────────────────────────────────────────────────────────┘
                    │
                    │ 5. Save final response
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  session.add_assistant_message(final_response)                   │
│  session.save_last_session(prompt, final_response)               │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
              Final text response
```

### RLM (Recursive Language Model) Flow

```
User: rlm("find the bug", huge_log_file)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                  rlm() in search.py                              │
│                                                                  │
│  Depth: 0, Max Iterations: 10, Max Depth: 3                      │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│              BUILD RLM SYSTEM PROMPT                             │
│                                                                  │
│  "You explore data by writing Python code.                       │
│   You have access to:                                            │
│     context: str (~50,000 chars)                                 │
│   Tools: peek(), grep(), partition(), rlm(), FINAL()"            │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│              ITERATION LOOP (max 10 iterations)                  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ 1. LLM generates Python code                              │  │
│  │                                                            │  │
│  │    # LLM output:                                           │  │
│  │    lines = grep("ERROR")                                   │  │
│  │    if len(lines) > 100:                                    │  │
│  │        chunks = partition(10)                              │  │
│  │        for i, chunk in enumerate(chunks):                  │  │
│  │            result = rlm("find bug", chunk)  # RECURSE      │  │
│  │            if "found" in result:                           │  │
│  │                FINAL(result)                               │  │
│  │    else:                                                   │  │
│  │        FINAL(lines[0])                                     │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ 2. Execute in SANDBOXED environment                       │  │
│  │                                                            │  │
│  │    local_vars = {                                          │  │
│  │      "context": context,                                   │  │
│  │      "peek": peek,                                         │  │
│  │      "grep": grep,                                         │  │
│  │      "partition": partition,                               │  │
│  │      "rlm": recurse,      # Wrapped to track depth         │  │
│  │      "FINAL": FINAL,      # Terminates the loop            │  │
│  │      "print": capture_print,                               │  │
│  │    }                                                       │  │
│  │                                                            │  │
│  │    exec(code, {"__builtins__": {}}, local_vars)            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│              ┌───────────────┴───────────────┐                  │
│              │ FINAL() called                │ No FINAL()       │
│              ▼                               ▼                  │
│     ┌─────────────────┐          ┌─────────────────────────┐   │
│     │  Return result  │          │  Add output to history  │   │
│     │  and stop       │          │  Continue loop          │   │
│     └─────────────────┘          └─────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Concepts

### 1. Shared Namespace

The fundamental design principle is **namespace sharing**:

```python
# User's Python REPL
>>> x = 42
>>> df = pd.read_csv("data.csv")

# Claude can see and use these variables
>>> ask("what's the mean of column A?")
# Claude generates:
#   print(df["A"].mean())
# Executes in same namespace, sees df!

# Claude can also create new variables
# that the user can then use
>>> # Claude created 'result' variable
>>> print(result)
```

### 2. LangGraph Agent Pattern

The agent uses a LangGraph StateGraph with conditional routing:

1. **START** → Model Node (call LLM)
2. **Router** inspects response:
   - Code blocks present → **Execute Node** → back to Model  
   - Neither → **END**
3. Loop continues until END or max_turns

This pattern enables:
- **Code execution**: Python blocks run in shared namespace
- **Error recovery**: Claude sees errors and can fix them
- **Streaming**: Fine-grained token/event callbacks via `astream_events()`

### 3. Context-Aware Prompts

System prompts are dynamically built with:
- **Variable summary**: What's in the namespace
- **Recent activity**: What the user has been typing
- **Prior session**: Last conversation (for continuity)
- **Skills**: Available skill files

### 4. Output Limiting

Safety mechanism to prevent runaway output:
- `LimitedOutput` class caps output at 10KB
- Forces Claude to use `peek()`, `grep()` for large data
- Encourages efficient exploration strategies

### 5. RLM (Recursive Language Model)

A novel approach to exploring large contexts:

```
┌─────────────────────────────────────────────────────────────────┐
│                     RLM EXPLORATION STRATEGY                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Large Context (e.g., 100KB log file)                            │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  peek(2000) → Understand structure                          ││
│  │  "Looks like JSON logs, one per line"                       ││
│  └─────────────────────────────────────────────────────────────┘│
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  grep("ERROR") → Find relevant sections                     ││
│  │  Returns 50 matching lines                                  ││
│  └─────────────────────────────────────────────────────────────┘│
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  If too many results:                                       ││
│  │    partition(10) → Split into chunks                        ││
│  │    rlm(query, chunk) → Recursively search each              ││
│  └─────────────────────────────────────────────────────────────┘│
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  FINAL(answer) → Terminate with result                      ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### Tools Available to Claude

#### File Tools (files.py)
| Tool | Purpose | Example |
|------|---------|---------|
| `read(path, start, end)` | Read file with line numbers | `read("main.py", 1, 50)` |
| `edit(path, old, new)` | Replace text in file | `edit("main.py", "bug", "fix")` |
| `create(name, content)` | Create new Python module | `create("utils")` → utils.py |

#### Text Tools (text.py)
| Tool | Purpose | Example |
|------|---------|---------|
| `peek(text, n)` | First n chars of text | `peek(log, 2000)` |
| `grep(text, pattern)` | Lines matching regex | `grep(log, "ERROR")` |
| `partition(text, n)` | Split into n chunks | `partition(log, 10)` |

#### Special Tools
| Tool | Purpose | Example |
|------|---------|---------|
| `rlm(query, context)` | Recursive LLM search | `rlm("find bug", log)` |
| `skill(name, args)` | Load skill instructions | `skill("debug")` |
| `load_claude_sessions()` | Load ~/.claude/projects/*.jsonl | `grep(sessions, "pattern")` |

#### Skill Loading
Skills are loaded by Claude outputting Python code that calls the `skill()` function:

```python
print(skill("debug"))  # Load the debug skill
```

The skill content is then printed and included in the conversation context.

### REPL Modes

| Mode | Entry | Prompt | Behavior |
|------|-------|--------|----------|
| **Interactive** | `mahtab` or `python -m mahtab` | `◈` (cyan) | Python + `ask()` inline |
| **Modal Python** | Toggle with `` ` `` | `◈` (cyan) | Pure Python execution |
| **Modal Ask** | Toggle with `` ` `` | `◈` (magenta) | Direct Claude conversation |

### File Persistence

```
~/.mahtab/
├── skills/                 # Custom skills (*.md files)
│   ├── debug.md
│   └── refactor.md
└── last_session.json       # Last conversation for continuity
    {
      "timestamp": "2025-01-31T...",
      "user": "explain this code",
      "assistant": "This code does..."
    }
```

---

## Entry Points

### 1. Command Line

```bash
# Using uv (recommended)
uv run mahtab

# As Python module
python -m mahtab

# Interactive mode with existing namespace
uv run python -i -m mahtab
```

### 2. Programmatic

```python
from mahtab.repl.interactive import run_repl
from mahtab.agent.repl_agent import create_repl_agent, get_llm
from mahtab.core.state import SessionState

# Option 1: Run full REPL
run_repl(ns=globals())

# Option 2: Use agent directly
session = SessionState()
session.init_namespace(globals())
agent = create_repl_agent(session=session)
response = agent.ask_sync("analyze this data")

# Option 3: Specify model
agent = create_repl_agent(
    session=session,
    model="claude-sonnet-4-20250514",
)

# Option 4: Async with callbacks
import asyncio

async def main():
    response = await agent.ask(
        "analyze this data",
        on_token=lambda t: print(t, end=""),
        on_code_block=lambda code, idx: print(f"Code {idx}: {code}"),
        on_execution=lambda out, err, idx: print(f"Output: {out}"),
    )

asyncio.run(main())
```

---

## Design Decisions

### Why CLI subprocess instead of API?

The ChatClaudeCLI approach via subprocess has several advantages:
- **No API key management**: Uses Claude Code CLI authentication
- **Simpler setup**: Works out of the box with Claude Code installed
- **Cost tracking**: CLI provides built-in usage and cost tracking
- **Consistent behavior**: Same model behavior as Claude Code

### Why LangGraph?

LangGraph provides structured agent workflows:
- **StateGraph**: Declarative node/edge definitions
- **Conditional routing**: Route to execute or end based on response
- **Streaming events**: Fine-grained control via `astream_events()`
- **Modularity**: Easy to add new nodes (e.g., human-in-the-loop)

### Why LangChain?

- **BaseChatModel**: Standard interface for ChatClaudeCLI
- **Message types**: HumanMessage, AIMessage, SystemMessage
- **Ecosystem**: Integrates with LangGraph, LangSmith, etc.

### Why Pydantic?

- **Validation**: SessionState and AgentState fields are validated
- **Serialization**: Easy JSON export for persistence
- **Type hints**: IDE autocomplete and type checking
- **ConfigDict**: Allows arbitrary types (e.g., BaseChatModel instances)

---

## Future Considerations

1. **Human-in-the-Loop**: LangGraph supports interrupt nodes for user confirmation before code execution
2. **Multi-Agent Patterns**: LangGraph enables supervisor/worker patterns for complex tasks
3. **LangSmith Integration**: Tracing and debugging of agent runs
4. **Additional Tools**: File search, web browsing, shell execution as bound tools
