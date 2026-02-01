"""Tests for prompt templates."""

from mahtab.llm.prompts import build_reflection_prompt, build_repl_system_prompt


def test_build_reflection_prompt_includes_original_prompt():
    result = build_reflection_prompt(
        original_prompt="calculate 2+2",
        code_blocks=["print(2+2)"],
        execution_results=[("4", False)],
    )
    assert "calculate 2+2" in result
    assert "print(2+2)" in result
    assert "4" in result


def test_build_reflection_prompt_marks_errors():
    result = build_reflection_prompt(
        original_prompt="divide by zero",
        code_blocks=["1/0"],
        execution_results=[("Error: division by zero", True)],
    )
    assert "ERROR" in result or "error" in result.lower()


def test_system_prompt_includes_xml_instructions():
    prompt = build_repl_system_prompt()
    assert "<assistant-chat>" in prompt
    assert "<assistant-repl-in>" in prompt
