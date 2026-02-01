"""Main entry point for running mahtab as a module.

Usage:
    python -m mahtab
    uv run python -i -m mahtab
"""

from mahtab.repl.interactive import run_repl

# Run the REPL setup when module is executed
run_repl()
