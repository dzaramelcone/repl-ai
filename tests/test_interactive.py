"""Tests for interactive REPL module."""

import logging

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
        log = logging.getLogger("test")
        return DynamicPrompt(session, ns, log)

    def test_input_mode_defaults_to_repl(self, prompt):
        """DynamicPrompt should start in REPL mode."""
        assert prompt.input_mode == "repl"

    def test_input_mode_can_be_chat(self, prompt):
        """DynamicPrompt should be able to switch to chat mode."""
        prompt.input_mode = "chat"
        assert prompt.input_mode == "chat"

    def test_str_shows_filled_diamond_in_repl_mode(self, prompt):
        """REPL mode should show filled diamond and py label."""
        prompt.input_mode = "repl"
        result = str(prompt)
        assert "◈" in result  # filled diamond
        assert "py" in result

    def test_str_shows_hollow_diamond_in_chat_mode(self, prompt):
        """CHAT mode should show hollow diamond and ai label."""
        prompt.input_mode = "chat"
        result = str(prompt)
        assert "◇" in result  # hollow diamond
        assert "ai" in result

    def test_str_does_not_show_filled_diamond_in_chat_mode(self, prompt):
        """CHAT mode should NOT show filled diamond."""
        prompt.input_mode = "chat"
        result = str(prompt)
        assert "◈" not in result

    def test_get_prompt_parts_has_rich_markup_in_repl_mode(self, prompt):
        """get_prompt_parts should return Rich markup for repl mode."""
        prompt.input_mode = "repl"
        info, mode = prompt.get_prompt_parts()
        assert "[cyan]" in mode
        assert "◈ py" in mode

    def test_get_prompt_parts_has_rich_markup_in_chat_mode(self, prompt):
        """get_prompt_parts should return Rich markup for chat mode."""
        prompt.input_mode = "chat"
        info, mode = prompt.get_prompt_parts()
        assert "[green]" in mode
        assert "◇ ai" in mode


class TestInputLogging:
    """Tests for input logging to correct tags."""

    def test_repl_mode_logs_to_user_repl_in_exactly_once(self):
        """In REPL mode, history should log to user-repl-in exactly once."""
        from unittest.mock import MagicMock, patch

        from mahtab.repl.interactive import DynamicPrompt

        session = SessionState()
        log = MagicMock()
        prompt = DynamicPrompt(session, {}, log)
        prompt.input_mode = "repl"
        prompt._last_history_len = 0

        with patch("readline.get_current_history_length", return_value=1):
            with patch("readline.get_history_item", return_value="x = 1"):
                str(prompt)

        assert log.info.call_count == 1
        log.info.assert_called_with("x = 1", extra={"tag": "user-repl-in"})

    def test_repl_mode_does_not_log_to_user_chat(self):
        """In REPL mode, history should NOT log to user-chat."""
        from unittest.mock import MagicMock, patch

        from mahtab.repl.interactive import DynamicPrompt

        session = SessionState()
        log = MagicMock()
        prompt = DynamicPrompt(session, {}, log)
        prompt.input_mode = "repl"
        prompt._last_history_len = 0

        with patch("readline.get_current_history_length", return_value=1):
            with patch("readline.get_history_item", return_value="x = 1"):
                str(prompt)

        for call in log.info.call_args_list:
            assert call.kwargs.get("extra", {}).get("tag") != "user-chat"

    def test_chat_mode_does_not_log_to_user_repl_in(self):
        """In CHAT mode, DynamicPrompt should NOT log to user-repl-in."""
        from unittest.mock import MagicMock, patch

        from mahtab.repl.interactive import DynamicPrompt

        session = SessionState()
        log = MagicMock()
        prompt = DynamicPrompt(session, {}, log)
        prompt.input_mode = "chat"
        prompt._last_history_len = 0

        with patch("readline.get_current_history_length", return_value=1):
            with patch("readline.get_history_item", return_value="hello"):
                str(prompt)

        for call in log.info.call_args_list:
            assert call.kwargs.get("extra", {}).get("tag") != "user-repl-in"

    def test_chat_mode_does_not_double_log(self):
        """In CHAT mode, DynamicPrompt should not log at all (ask() handles it)."""
        from unittest.mock import MagicMock, patch

        from mahtab.repl.interactive import DynamicPrompt

        session = SessionState()
        log = MagicMock()
        prompt = DynamicPrompt(session, {}, log)
        prompt.input_mode = "chat"
        prompt._last_history_len = 0

        with patch("readline.get_current_history_length", return_value=1):
            with patch("readline.get_history_item", return_value="hello"):
                str(prompt)

        assert log.info.call_count == 0


class TestToggleMode:
    """Tests for mode toggle functionality."""

    @pytest.fixture
    def prompt(self):
        """Create a DynamicPrompt for testing."""
        from mahtab.repl.interactive import DynamicPrompt

        session = SessionState()
        ns = {}
        log = logging.getLogger("test")
        return DynamicPrompt(session, ns, log)

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
        log = logging.getLogger("test")
        prompt_obj = DynamicPrompt(session, ns, log)
        calls = []

        def ask_func(text):
            calls.append(text)

        repl = InteractiveREPL(locals=ns, prompt_obj=prompt_obj, ask_func=ask_func, log=log)
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


class TestDynamicPromptFormatting:
    """Tests for DynamicPrompt string formatting."""

    @pytest.fixture
    def prompt_with_history(self):
        """Create a DynamicPrompt with session history for testing."""
        from mahtab.repl.interactive import DynamicPrompt

        session = SessionState()
        # Add a message to create history using the proper method
        session.add_user_message("test message content that is reasonably long")
        ns = {}
        log = logging.getLogger("test")
        return DynamicPrompt(session, ns, log)

    @pytest.fixture
    def prompt_with_usage(self):
        """Create a DynamicPrompt with usage stats for testing."""
        from mahtab.repl.interactive import DynamicPrompt

        session = SessionState()
        # Record some usage to trigger cost display
        session.usage.record(cost=0.04, input_tokens=100, output_tokens=50, cache_read=0, cache_create=0)
        ns = {}
        log = logging.getLogger("test")
        return DynamicPrompt(session, ns, log)

    def test_prompt_contains_memory_mb(self):
        """Prompt should show memory usage in MB format."""
        from mahtab.repl.interactive import DynamicPrompt

        session = SessionState()
        log = logging.getLogger("test")
        prompt = DynamicPrompt(session, {}, log)
        result = str(prompt)
        # Should contain "MB" for memory
        assert "MB" in result

    def test_prompt_contains_token_count_when_history_exists(self, prompt_with_history):
        """Prompt should show token count with 't' suffix when there is history."""
        result = str(prompt_with_history)
        import re

        # Strip ANSI codes for easier matching
        clean = re.sub(r"\x01?\x1b\[[0-9;]*m\x02?", "", result)
        # Should match patterns like "5t" or "1.2kt" or "3.5Mt"
        assert re.search(r"\d+\.?\d*[kMG]?t", clean), f"Expected Nt pattern in '{clean}'"

    def test_prompt_contains_cost_when_usage_exists(self, prompt_with_usage):
        """Prompt should show cost with $ prefix when there is usage."""
        result = str(prompt_with_usage)
        # Should contain "$" for cost
        assert "$" in result

    def test_prompt_cost_shows_full_value(self, prompt_with_usage):
        """Prompt cost should show the full numeric value, not truncated."""
        result = str(prompt_with_usage)
        # Should show $0.04, not $0. or partial
        assert "$0.04" in result, f"Expected $0.04 in '{result}'"

    def test_prompt_format_order(self, prompt_with_usage):
        """Prompt should show elements in order: memory, cost, mode indicator."""
        result = str(prompt_with_usage)

        # Find positions of key elements
        mb_pos = result.find("MB")
        cost_pos = result.find("$")
        mode_pos = result.find("◈") if "◈" in result else result.find("◇")

        # Memory should come first, then cost, then mode indicator
        assert mb_pos >= 0, f"Missing MB in prompt: {result}"
        assert cost_pos >= 0, f"Missing $ in prompt: {result}"
        assert mode_pos >= 0, f"Missing mode indicator in prompt: {result}"
        assert mb_pos < cost_pos < mode_pos, (
            f"Wrong order in prompt. Expected MB < $ < mode, "
            f"got positions {mb_pos}, {cost_pos}, {mode_pos} in '{result}'"
        )

    def test_get_prompt_parts_returns_rich_markup(self, prompt_with_usage):
        """get_prompt_parts should return Rich markup for colors."""
        info, mode = prompt_with_usage.get_prompt_parts()
        # Info should have bright_blue markup for numbers
        assert "[bright_blue]" in info
        # Mode should have color markup
        assert "[cyan]" in mode or "[green]" in mode


class TestExitIntegration:
    """Integration tests for exit() behavior - shells out to actual process."""

    def test_exit_exits_cleanly_no_traceback(self):
        """exit() should exit with code 0 and no traceback in output."""
        import subprocess
        import sys

        # Run mahtab process, send exit() command
        result = subprocess.run(
            [sys.executable, "-m", "mahtab"],
            input="exit()\n",
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Must exit with code 0
        assert result.returncode == 0, f"Expected exit code 0, got {result.returncode}"

        # No traceback in stdout or stderr
        combined = result.stdout + result.stderr
        assert "Traceback" not in combined, f"Found traceback in output:\n{combined}"
        assert "Error" not in combined, f"Found Error in output:\n{combined}"
        assert "SystemExit" not in combined, f"Found SystemExit in output:\n{combined}"
