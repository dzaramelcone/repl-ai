"""
Claude CLI wrapper for rlm.
"""
import subprocess
import json

from rich.console import Console

console = Console()


def messages_create(
    model: str,
    max_tokens: int,
    system: str,
    messages: list[dict],
    on_token: callable = None,
) -> dict:
    """
    Call Claude via CLI with streaming output.
    Returns a dict with content[0].text.

    If on_token is provided, calls it with each token instead of printing.
    """
    # Build the prompt from messages
    prompt = ""
    for msg in messages:
        if msg["role"] == "user":
            prompt += msg["content"]

    proc = subprocess.Popen(
        [
            "claude",
            "-p", prompt,
            "--model", model,
            "--system-prompt", system,
            "--setting-sources", "",
            "--output-format", "stream-json",
            "--include-partial-messages",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd="/tmp",
    )

    full_response = ""

    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)

                # Handle streaming deltas
                if data.get("type") == "stream_event":
                    event = data.get("event", {})
                    if event.get("type") == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if on_token:
                                on_token(text)
                            else:
                                print(f"\033[2m{text}\033[0m", end="", flush=True)
                            full_response += text

                # Handle final result
                elif data.get("type") == "result":
                    if not full_response:
                        full_response = data.get("result", "")

            except json.JSONDecodeError:
                pass

        if not on_token:
            console.print()  # newline after stream
        proc.wait()

    except KeyboardInterrupt:
        proc.terminate()
        console.print("\n[yellow][cancelled][/]")
        raise

    if proc.returncode != 0:
        stderr = proc.stderr.read()
        raise RuntimeError(f"claude CLI failed: {stderr}")

    return {"content": [{"text": full_response}]}
