"""
Minimal RLM - Recursive Language Model

The LLM writes code to explore context. That's it.
"""
import re
import anthropic

client = anthropic.Anthropic()

SYSTEM = """You explore data by writing Python code.

You have access to:
- `context`: str - the data to explore (DO NOT print it all - it may be huge)
- `rlm(query, subset)`: recursively explore a subset
- `FINAL(answer)`: return your answer and stop

Write ONLY Python code. No markdown fences. No explanation."""


def rlm(query: str, context: str, depth: int = 0, max_iters: int = 10) -> str:
    """
    Recursive Language Model.

    LLM generates code to explore context.
    Code can call rlm() to recurse on subsets.
    """
    history = ""

    for i in range(max_iters):
        # Build prompt
        prompt = f"""Query: {query}
Context: {len(context)} chars
Depth: {depth}

{f"History:{history}" if history else "(no history yet)"}

Write Python code to answer the query."""

        # Call LLM
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}]
        )

        code = response.content[0].text.strip()
        code = re.sub(r'^```python\n?', '', code)
        code = re.sub(r'\n?```$', '', code)

        print(f"\n[depth={depth} iter={i+1}] Code:\n{code}\n")

        # Execute
        result = {"_final": None}

        def FINAL(answer):
            result["_final"] = str(answer)
            return answer

        def recurse(q, subset):
            return rlm(q, subset, depth=depth+1, max_iters=max_iters)

        local_vars = {
            "context": context,
            "FINAL": FINAL,
            "rlm": recurse,
        }

        try:
            exec(code, {"__builtins__": __builtins__}, local_vars)
        except Exception as e:
            output = f"Error: {e}"
            print(f"[depth={depth} iter={i+1}] {output}")
            history += f"\n---\nCode:\n{code}\nError: {e}\n"
            continue

        # Check for FINAL
        if result["_final"] is not None:
            print(f"[depth={depth}] FINAL: {result['_final']}")
            return result["_final"]

        # Capture any printed output (simplified - just note execution succeeded)
        history += f"\n---\nCode:\n{code}\nExecuted successfully.\n"

    return f"[depth={depth}] Max iterations reached"


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python rlm.py <context_file> <query>")
        sys.exit(1)

    context_file = sys.argv[1]
    query = " ".join(sys.argv[2:])

    with open(context_file) as f:
        context = f.read()

    print(f"Context: {len(context)} chars")
    print(f"Query: {query}")
    print("=" * 60)

    answer = rlm(query, context)

    print("=" * 60)
    print(f"ANSWER: {answer}")
