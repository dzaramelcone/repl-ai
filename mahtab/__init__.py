"""
Mahtab - AI-powered shared Python REPL with Claude integration.

A collaborative environment where Claude can execute code directly in the user's namespace,
inspect variables, modify files, and explore large text contexts using recursive search strategies.
"""

from mahtab.core.state import SessionState, UsageStats

__version__ = "0.2.0"
__all__ = ["SessionState", "UsageStats"]
