"""Tests for interactive REPL module."""

from __future__ import annotations

import pytest

from mahtab.core.state import SessionState


class TestDynamicPromptInputMode:
    """Tests for DynamicPrompt input mode functionality."""

    @pytest.fixture
    def prompt(self):
        """Create a DynamicPrompt for testing."""
        from mahtab.repl.interactive import DynamicPrompt

        session = SessionState()
        ns = {}
        return DynamicPrompt(session, ns)

    def test_input_mode_defaults_to_repl(self, prompt):
        """DynamicPrompt should start in REPL mode."""
        assert prompt.input_mode == "repl"

    def test_input_mode_can_be_chat(self, prompt):
        """DynamicPrompt should be able to switch to chat mode."""
        prompt.input_mode = "chat"
        assert prompt.input_mode == "chat"

    def test_str_shows_filled_diamond_in_repl_mode(self, prompt):
        """REPL mode should show filled cyan diamond."""
        prompt.input_mode = "repl"
        result = str(prompt)
        # Should contain cyan filled diamond
        assert "\033[36m" in result  # cyan
        assert "◈" in result  # filled diamond

    def test_str_shows_hollow_diamond_in_chat_mode(self, prompt):
        """CHAT mode should show green hollow diamond."""
        prompt.input_mode = "chat"
        result = str(prompt)
        # Should contain green hollow diamond
        assert "\033[32m" in result  # green
        assert "◇" in result  # hollow diamond

    def test_str_does_not_show_filled_diamond_in_chat_mode(self, prompt):
        """CHAT mode should NOT show filled diamond."""
        prompt.input_mode = "chat"
        result = str(prompt)
        assert "◈" not in result


class TestToggleMode:
    """Tests for mode toggle functionality."""

    @pytest.fixture
    def prompt(self):
        """Create a DynamicPrompt for testing."""
        from mahtab.repl.interactive import DynamicPrompt

        session = SessionState()
        ns = {}
        return DynamicPrompt(session, ns)

    def test_toggle_mode_from_repl_to_chat(self, prompt):
        """Toggle from REPL should switch to CHAT."""
        assert prompt.input_mode == "repl"
        prompt.toggle_mode()
        assert prompt.input_mode == "chat"

    def test_toggle_mode_from_chat_to_repl(self, prompt):
        """Toggle from CHAT should switch to REPL."""
        prompt.input_mode = "chat"
        prompt.toggle_mode()
        assert prompt.input_mode == "repl"

    def test_toggle_mode_is_idempotent_over_two_calls(self, prompt):
        """Two toggles should return to original mode."""
        original = prompt.input_mode
        prompt.toggle_mode()
        prompt.toggle_mode()
        assert prompt.input_mode == original
