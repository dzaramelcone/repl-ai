"""Skills management for loading and invoking skill files."""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# Default skills directory
DEFAULT_SKILLS_DIR = Path("~/.mahtab/skills").expanduser()


def get_skill_description(skill_file: Path) -> str:
    """Extract description from a skill file's YAML frontmatter.

    Args:
        skill_file: Path to the skill markdown file.

    Returns:
        The skill description, or the skill name if no description found.
    """
    content = skill_file.read_text()
    name = skill_file.stem

    # Parse YAML frontmatter
    if content.startswith("---"):
        try:
            end = content.index("---", 3)
            frontmatter = content[3:end].strip()
            for line in frontmatter.split("\n"):
                if line.startswith("description:"):
                    return line.split(":", 1)[1].strip().strip("\"'")
        except ValueError:
            pass

    return name


def load_skill_descriptions(skills_dir: Path | None = None) -> str:
    """Load skill descriptions from the skills directory.

    Args:
        skills_dir: Path to skills directory. Default ~/.mahtab/skills.

    Returns:
        Formatted string of skill names and descriptions.
    """
    if skills_dir is None:
        skills_dir = DEFAULT_SKILLS_DIR

    if not skills_dir.exists():
        return ""

    descriptions = []
    for skill_file in sorted(skills_dir.glob("*.md")):
        name = skill_file.stem
        desc = get_skill_description(skill_file)
        descriptions.append(f"    - {name}: {desc}")

    if descriptions:
        return f"""Skills (IMPORTANT: use skill("name") function in Python code to load):
  To load a skill, run: print(skill("skill_name"))
  Available skills:
{chr(10).join(descriptions)}"""
    return ""


def load_skill_content(name: str, args: str = "", skills_dir: Path | None = None) -> str:
    """Load and return a skill's full content.

    Args:
        name: Name of the skill (without .md extension).
        args: Arguments to pass to the skill (replaces $ARGUMENTS).
        skills_dir: Path to skills directory. Default ~/.mahtab/skills.

    Returns:
        The skill content with arguments substituted, or error message.
    """
    if skills_dir is None:
        skills_dir = DEFAULT_SKILLS_DIR

    skill_file = skills_dir / f"{name}.md"
    if not skill_file.exists():
        available = [f.stem for f in skills_dir.glob("*.md")]
        return f"Error: skill '{name}' not found. Available skills: {available}"

    content = skill_file.read_text()

    # Strip frontmatter
    if content.startswith("---"):
        try:
            end = content.index("---", 3)
            content = content[end + 3 :].strip()
        except ValueError:
            pass

    # Replace $ARGUMENTS placeholder
    content = content.replace("$ARGUMENTS", args)

    return content


class LoadSkillInput(BaseModel):
    """Input schema for the load_skill tool."""

    name: str = Field(description="The skill name to load (without .md extension)")
    args: str = Field(default="", description="Optional arguments for the skill (replaces $ARGUMENTS)")


def get_skill_tool(skills_dir: Path | None = None) -> StructuredTool:
    """Create a load_skill tool with dynamically generated description.

    The tool description includes the list of available skills, making it
    easier for the LLM to know what skills are available.

    Args:
        skills_dir: Path to skills directory. Default ~/.mahtab/skills.

    Returns:
        A StructuredTool for loading skills.
    """
    if skills_dir is None:
        skills_dir = DEFAULT_SKILLS_DIR

    # Ensure directory exists
    skills_dir.mkdir(parents=True, exist_ok=True)

    # Build description from available skills
    skill_list = []
    for skill_file in sorted(skills_dir.glob("*.md")):
        name = skill_file.stem
        desc = get_skill_description(skill_file)
        skill_list.append(f"    - {name}: {desc}")

    skill_list_str = "\n".join(skill_list) if skill_list else "    (no skills installed)"

    description = f"""Load a skill to enhance your capabilities for the current task.

IMPORTANT: You SHOULD call this tool when a skill would help you complete
the user's request more effectively. Skills provide specialized knowledge,
step-by-step workflows, and best practices. When in doubt, load the skill -
it will make your response better.

Available skills:
{skill_list_str}

Returns detailed instructions to follow. Apply these to complete the task."""

    # Capture skills_dir in closure
    _skills_dir = skills_dir

    def _load_skill(name: str, args: str = "") -> str:
        """Load a skill by name."""
        return load_skill_content(name, args, _skills_dir)

    return StructuredTool.from_function(
        func=_load_skill,
        name="load_skill",
        description=description,
        args_schema=LoadSkillInput,
    )


def _load_skill_func(name: str, args: str = "") -> str:
    """Load a skill by name (uses default skills directory)."""
    return load_skill_content(name, args, DEFAULT_SKILLS_DIR)


# Create the load_skill tool using StructuredTool for proper schema handling
load_skill = StructuredTool.from_function(
    func=_load_skill_func,
    name="load_skill",
    description="""Load and return a skill's full content.

Skills are markdown files in ~/.mahtab/skills/ with optional YAML frontmatter.
The $ARGUMENTS placeholder in the skill content is replaced with the args parameter.""",
    args_schema=LoadSkillInput,
)


def load_claude_sessions(projects_path: str = "~/.claude/projects") -> str:
    """Load all JSONL files from Claude projects into one big context.

    Args:
        projects_path: Path to Claude projects directory.

    Returns:
        Concatenated content of all JSONL files with file headers.
    """
    path = Path(projects_path).expanduser()
    chunks = []

    for jsonl_file in sorted(path.rglob("*.jsonl")):
        rel_path = jsonl_file.relative_to(path)
        content = jsonl_file.read_text()
        chunks.append(f"\n\n=== FILE: {rel_path} ===\n{content}")

    return "".join(chunks)
