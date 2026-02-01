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


class TestInputRouting:
    """Tests for routing input based on mode."""

    def test_is_toggle_command_recognizes_slash(self):
        """Single '/' should be recognized as toggle command."""
        from mahtab.repl.interactive import is_toggle_command

        assert is_toggle_command("/") is True

    def test_is_toggle_command_rejects_regular_input(self):
        """Regular input should not be recognized as toggle."""
        from mahtab.repl.interactive import is_toggle_command

        assert is_toggle_command("print('hello')") is False
        assert is_toggle_command("ask('question')") is False
        assert is_toggle_command("") is False
        assert is_toggle_command("// comment") is False

    def test_is_toggle_command_strips_whitespace(self):
        """Toggle command should work with leading/trailing whitespace."""
        from mahtab.repl.interactive import is_toggle_command

        assert is_toggle_command(" / ") is True
        assert is_toggle_command("\t/\n") is True

    def test_should_route_to_chat_in_repl_mode(self):
        """In REPL mode, regular input should NOT route to chat."""
        from mahtab.repl.interactive import should_route_to_chat

        assert should_route_to_chat("print('hello')", "repl") is False

    def test_should_route_to_chat_in_chat_mode(self):
        """In CHAT mode, regular input SHOULD route to chat."""
        from mahtab.repl.interactive import should_route_to_chat

        assert should_route_to_chat("hello there", "chat") is True

    def test_should_route_to_chat_empty_input(self):
        """Empty input should not route to chat in any mode."""
        from mahtab.repl.interactive import should_route_to_chat

        assert should_route_to_chat("", "chat") is False
        assert should_route_to_chat("   ", "chat") is False
        assert should_route_to_chat("", "repl") is False


class TestInteractiveREPL:
    """Tests for InteractiveREPL class."""

    @pytest.fixture
    def repl(self):
        """Create an InteractiveREPL for testing."""
        from mahtab.repl.interactive import DynamicPrompt, InteractiveREPL

        session = SessionState()
        ns = {}
        prompt_obj = DynamicPrompt(session, ns)
        calls = []

        def ask_func(text):
            calls.append(text)

        repl = InteractiveREPL(locals=ns, prompt_obj=prompt_obj, ask_func=ask_func)
        repl._ask_calls = calls  # Store for verification
        return repl

    def test_runsource_toggle_command_toggles_mode(self, repl):
        """'/' should toggle mode without executing."""
        assert repl.prompt_obj.input_mode == "repl"
        result = repl.runsource("/")
        assert repl.prompt_obj.input_mode == "chat"
        assert result is False  # No more input needed

    def test_runsource_chat_mode_calls_ask(self, repl):
        """In chat mode, regular input calls ask_func."""
        repl.prompt_obj.input_mode = "chat"
        result = repl.runsource("hello there")
        assert result is False
        assert repl._ask_calls == ["hello there"]

    def test_runsource_repl_mode_executes_python(self, repl):
        """In repl mode, input is executed as Python."""
        repl.prompt_obj.input_mode = "repl"
        # Execute valid Python that sets a variable
        repl.runsource("x = 42")
        assert repl.locals.get("x") == 42
        assert repl._ask_calls == []  # ask was not called

    def test_runsource_empty_input_is_noop(self, repl):
        """Empty input does nothing in any mode."""
        repl.prompt_obj.input_mode = "chat"
        result = repl.runsource("")
        assert result is False
        assert repl._ask_calls == []

    def test_runsource_whitespace_toggle(self, repl):
        """Toggle with whitespace around slash."""
        assert repl.prompt_obj.input_mode == "repl"
        result = repl.runsource("  /  ")
        assert repl.prompt_obj.input_mode == "chat"
        assert result is False
