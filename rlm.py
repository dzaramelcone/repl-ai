"""
Minimal RLM - Recursive Language Model

The LLM writes code to explore context. That's it.
"""
import re
import anthropic
from pathlib import Path

client = anthropic.Anthropic()

SYSTEM = """You explore data by writing Python code.

You have access to:
  context: str           # The full data (~{size} chars). DO NOT print it all.

Tools (these are functions you can call):
  peek(n=2000) -> str              # First n chars of context
  grep(pattern) -> list[str]       # Lines matching regex pattern
  partition(n=10) -> list[str]     # Split context into n chunks
  rlm(query, subset) -> str        # Recursively explore a subset

Termination:
  FINAL(answer)                    # Return answer and stop

Strategy:
1. peek() to understand structure
2. grep() to find relevant sections
3. partition() + rlm() for large sections
4. FINAL() when you have the answer

Write ONLY Python code. No markdown. No explanation. No print() unless debugging."""


def rlm(query: str, context: str, depth: int = 0, max_iters: int = 10) -> str:
    """
    Recursive Language Model.
    LLM generates code to explore context via peek/grep/partition/rlm.
    """
    history = ""
    output_buffer = []

    # Tools available to the LLM's code
    def peek(n: int = 2000) -> str:
        return context[:n]

    def grep(pattern: str) -> list[str]:
        return [line for line in context.split('\n') if re.search(pattern, line, re.IGNORECASE)]

    def partition(n: int = 10) -> list[str]:
        chunk_size = len(context) // n
        return [context[i:i+chunk_size] for i in range(0, len(context), chunk_size)]

    def recurse(q: str, subset: str) -> str:
        if depth >= 3:
            return f"[max depth reached, subset is {len(subset)} chars]"
        return rlm(q, subset, depth=depth+1, max_iters=max_iters)

    result = {"_final": None}

    def FINAL(answer):
        result["_final"] = str(answer)
        return answer

    def capture_print(*args, **kwargs):
        output_buffer.append(" ".join(str(a) for a in args))

    for i in range(max_iters):
        prompt = f"""Query: {query}
Context size: {len(context):,} chars
Depth: {depth}

{f"History:{history}" if history else "(first iteration)"}

Write Python code:"""

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            system=SYSTEM.format(size=f"{len(context):,}"),
            messages=[{"role": "user", "content": prompt}]
        )

        code = response.content[0].text.strip()
        code = re.sub(r'^```python\n?', '', code)
        code = re.sub(r'\n?```$', '', code)

        print(f"\n[depth={depth} iter={i+1}] Code:\n{code}\n")

        output_buffer.clear()

        local_vars = {
            "context": context,
            "peek": peek,
            "grep": grep,
            "partition": partition,
            "rlm": recurse,
            "FINAL": FINAL,
            "print": capture_print,
            "re": re,
            "len": len,
        }

        try:
            exec(code, {"__builtins__": {}}, local_vars)
        except Exception as e:
            output = f"Error: {e}"
            print(f"[depth={depth} iter={i+1}] {output}")
            history += f"\n---\nCode:\n{code}\nError: {e}\n"
            continue

        if result["_final"] is not None:
            print(f"[depth={depth}] FINAL: {result['_final'][:500]}...")
            return result["_final"]

        output = "\n".join(output_buffer) if output_buffer else "(no output)"
        print(f"[depth={depth} iter={i+1}] Output:\n{output[:1000]}")
        history += f"\n---\nCode:\n{code}\nOutput:\n{output}\n"

    return f"[depth={depth}] Max iterations reached"


def load_claude_sessions(projects_path: str = "~/.claude/projects") -> str:
    """Load all JSONL files from Claude projects into one big context."""
    path = Path(projects_path).expanduser()
    chunks = []

    for jsonl_file in sorted(path.rglob("*.jsonl")):
        rel_path = jsonl_file.relative_to(path)
        content = jsonl_file.read_text()
        chunks.append(f"\n\n=== FILE: {rel_path} ===\n{content}")

    return "".join(chunks)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python rlm.py <query>")
        print("       python rlm.py --file <context_file> <query>")
        sys.exit(1)

    if sys.argv[1] == "--file":
        context_file = sys.argv[2]
        query = " ".join(sys.argv[3:])
        with open(context_file) as f:
            context = f.read()
    else:
        query = " ".join(sys.argv[1:])
        print("Loading Claude sessions...")
        context = load_claude_sessions()

    print(f"Context: {len(context):,} chars")
    print(f"Query: {query}")
    print("=" * 60)

    answer = rlm(query, context)

    print("=" * 60)
    print(f"ANSWER: {answer}")
