"""UI components: console utilities, panels, and streaming animations."""

from mahtab.ui.console import console, get_console
from mahtab.ui.panels import (
    print_banner,
    print_code_panel,
    print_final_panel,
    print_modal_banner,
    print_output_panel,
    print_usage_panel,
)
from mahtab.ui.streaming import StreamingHandler

__all__ = [
    "console",
    "get_console",
    "print_code_panel",
    "print_output_panel",
    "print_final_panel",
    "print_banner",
    "print_modal_banner",
    "print_usage_panel",
    "StreamingHandler",
]
