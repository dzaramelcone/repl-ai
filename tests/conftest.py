"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import pytest

from mahtab.core.state import SessionState


@pytest.fixture
def session() -> SessionState:
    """Create a fresh session state for testing."""
    return SessionState()


@pytest.fixture
def session_with_namespace() -> SessionState:
    """Create session with some variables in namespace."""
    session = SessionState()
    session.globals_ns = {"x": 42, "name": "test"}
    return session
