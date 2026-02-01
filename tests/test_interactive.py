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
        return DynamicPrompt(session, ns)

    @pytest.fixture
    def prompt_with_usage(self):
        """Create a DynamicPrompt with usage stats for testing."""
        from mahtab.repl.interactive import DynamicPrompt

        session = SessionState()
        # Record some usage to trigger cost display
        session.usage.record(cost=0.04, input_tokens=100, output_tokens=50, cache_read=0, cache_create=0)
        ns = {}
        return DynamicPrompt(session, ns)

    def test_prompt_contains_memory_mb(self):
        """Prompt should show memory usage in MB format."""
        from mahtab.repl.interactive import DynamicPrompt

        session = SessionState()
        prompt = DynamicPrompt(session, {})
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
        import re

        # Strip ANSI codes
        clean = re.sub(r"\x01?\x1b\[[0-9;]*m\x02?", "", result)
        # Should show $0.04, not $0. or partial
        assert re.search(r"\$0\.04", clean), f"Expected $0.04 in '{clean}'"

    def test_prompt_ansi_escapes_properly_wrapped_for_readline(self, prompt_with_history):
        """All ANSI escape sequences must be wrapped in \\x01...\\x02 for readline."""
        result = str(prompt_with_history)

        # Every ANSI escape (\x1b[...m) must be preceded by \x01 and followed by \x02
        # This is required for readline to correctly calculate prompt display width
        import re

        # Find all ANSI escape sequences
        ansi_pattern = r"\x1b\[[0-9;]*m"

        # Check that each ANSI escape is wrapped
        # Pattern: \x01 followed by ANSI escape followed by \x02
        wrapped_pattern = r"\x01\x1b\[[0-9;]*m\x02"

        # Count total ANSI escapes
        total_ansi = len(re.findall(ansi_pattern, result))
        # Count properly wrapped ANSI escapes
        wrapped_ansi = len(re.findall(wrapped_pattern, result))

        assert total_ansi == wrapped_ansi, (
            f"Found {total_ansi} ANSI escapes but only {wrapped_ansi} properly wrapped. "
            f"Unwrapped escapes cause readline to miscalculate prompt length."
        )

    def test_prompt_no_nested_readline_markers(self, prompt_with_history):
        """Readline markers (\\x01...\\x02) must not be nested."""
        result = str(prompt_with_history)

        # Walk through string tracking nesting depth
        depth = 0
        for i, char in enumerate(result):
            if char == "\x01":
                depth += 1
                assert depth == 1, f"Nested \\x01 at position {i}: {repr(result[max(0, i - 5) : i + 10])}"
            elif char == "\x02":
                assert depth == 1, f"Unexpected \\x02 at position {i} with depth {depth}"
                depth = 0

        assert depth == 0, "Unclosed \\x01 marker"

    def test_prompt_format_order(self, prompt_with_usage):
        """Prompt should show elements in order: memory, hist, cost, mode indicator."""
        result = str(prompt_with_usage)
        import re

        # Strip ANSI codes
        clean = re.sub(r"\x01?\x1b\[[0-9;]*m\x02?", "", result)

        # Find positions of key elements
        mb_pos = clean.find("MB")
        cost_pos = clean.find("$")
        mode_pos = clean.find("◈") if "◈" in clean else clean.find("◇")

        # Memory should come first, then cost, then mode indicator
        assert mb_pos >= 0, f"Missing MB in prompt: {clean}"
        assert cost_pos >= 0, f"Missing $ in prompt: {clean}"
        assert mode_pos >= 0, f"Missing mode indicator in prompt: {clean}"
        assert mb_pos < cost_pos < mode_pos, (
            f"Wrong order in prompt. Expected MB < $ < mode, "
            f"got positions {mb_pos}, {cost_pos}, {mode_pos} in '{clean}'"
        )


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
