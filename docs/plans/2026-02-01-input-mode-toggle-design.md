# Input Mode Toggle Design

> **For Claude:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add shift+tab toggle between REPL and CHAT input modes for faster workflow switching.

**Architecture:** Extend DynamicPrompt to track mode state, bind shift+tab to toggle, intercept input in CHAT mode to route to ask() instead of Python interpreter.

---

## Core Behavior

**Mode Toggle:**
- Shift+Tab switches between REPL mode and CHAT mode
- Mode persists until toggled again
- Starts in REPL mode (current default behavior)

**Visual Indicators:**
- REPL mode: `◈` (cyan) - current symbol
- CHAT mode: `◇` (green) - hollow diamond, matches "You" panel color
- The rest of the dynamic prompt info (memory, context, cost) stays the same

**Input Handling:**
- REPL mode: Input goes to Python's exec() as it does now
- CHAT mode: Input goes directly to ask() without needing the function wrapper
- Enter submits in both modes
- Shift+Enter adds a newline for multi-line input in CHAT mode

---

## Implementation

**State Management:**
- Add `input_mode` attribute to `DynamicPrompt` class
- Two values: `"repl"` or `"chat"`
- `__str__` returns appropriate symbol/color based on mode

**Keybinding:**
- Use readline to bind Shift+Tab (`\x1b[Z`) to mode toggle function
- Binding flips the mode and refreshes the prompt

**Input Interception:**
- Use custom `readfunc` with `code.interact()` or similar
- When in CHAT mode, intercept input before Python parses it
- Route to `ask()` instead of interpreter

---

## Edge Cases

- **Ctrl+C:** Cancel current input, stay in current mode
- **Empty input:** No-op in both modes
- **History:** Shared between modes (up arrow shows all previous inputs)
- **Streaming cancel:** Ctrl+C during response stays in CHAT mode

---

## Files to Modify

- `mahtab/repl/interactive.py` - DynamicPrompt class, readline bindings, input interception
