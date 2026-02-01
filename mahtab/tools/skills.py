"""Skills management for loading and invoking skill files."""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

# Default skills directory
DEFAULT_SKILLS_DIR = Path("~/.mahtab/skills").expanduser()


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
        content = skill_file.read_text()
        name = skill_file.stem

        # Parse YAML frontmatter
        desc = name  # Default description
        if content.startswith("---"):
            try:
                end = content.index("---", 3)
                frontmatter = content[3:end].strip()
                for line in frontmatter.split("\n"):
                    if line.startswith("description:"):
                        desc = line.split(":", 1)[1].strip().strip("\"'")
                        break
            except ValueError:
                pass

        descriptions.append(f"  {name}: {desc}")

    if descriptions:
        return "Skills (use skill(name) to invoke):\n" + "\n".join(descriptions)
    return ""


@tool
def load_skill(name: str, args: str = "", skills_dir: Path | None = None) -> str:
    """Load and return a skill's full content.

    Skills are markdown files in ~/.mahtab/skills/ with optional YAML frontmatter.
    The $ARGUMENTS placeholder in the skill content is replaced with the args parameter.

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
        return f"Error: skill '{name}' not found in {skills_dir}"

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
