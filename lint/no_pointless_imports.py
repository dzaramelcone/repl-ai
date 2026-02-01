"""Pylint plugin to ban pointless imports."""

from astroid import nodes
from pylint.checkers import BaseChecker


class NoPointlessImportsChecker(BaseChecker):
    """Check for pointless imports that don't do anything."""

    name = "no-pointless-imports"
    msgs = {
        "E9901": (
            "TYPE_CHECKING import is pointless and doesn't do anything, don't include it",
            "no-type-checking",
            "Importing TYPE_CHECKING from typing is pointless in this codebase.",
        ),
        "E9902": (
            "from __future__ import is pointless and doesn't do anything, don't include it",
            "no-future-imports",
            "Future imports are pointless in Python 3.11+ and don't do anything.",
        ),
    }

    def visit_importfrom(self, node: nodes.ImportFrom) -> None:
        """Check ImportFrom nodes for banned imports."""
        # Check for: from typing import TYPE_CHECKING
        if node.modname == "typing":
            for name, _ in node.names:
                if name == "TYPE_CHECKING":
                    self.add_message("no-type-checking", node=node)

        # Check for: from __future__ import ...
        if node.modname == "__future__":
            self.add_message("no-future-imports", node=node)

    def visit_import(self, node: nodes.Import) -> None:
        """Check Import nodes for banned imports."""
        # Check for: import typing (then using typing.TYPE_CHECKING)
        # We only catch direct TYPE_CHECKING imports, not module-level typing imports
        pass


def register(linter):
    """Register the checker with pylint."""
    linter.register_checker(NoPointlessImportsChecker(linter))
