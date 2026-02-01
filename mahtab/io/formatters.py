"""Logging formatters for different output targets."""

import logging


class RichFormatter(logging.Formatter):
    """Formats log messages with Rich markup based on tag."""

    def format(self, record: logging.LogRecord) -> str:
        content = record.getMessage()

        match record.tag:
            case "user-repl-in":
                return f"[bold cyan]>>> [/]{content}"
            case "user-repl-out":
                return content
            case "assistant-repl-in":
                return f"[bold magenta]>>> [/]{content}"
            case "assistant-repl-out":
                return content
            case "user-chat":
                return f"[bold green]You:[/] {content}"
            case "assistant-chat":
                return f"[bold blue]Claude:[/] {content}"
            case _:
                return content


class XMLFormatter(logging.Formatter):
    """Wraps log message in XML tag from record.tag attribute."""

    def format(self, record: logging.LogRecord) -> str:
        return f"<{record.tag}>{record.getMessage()}</{record.tag}>"


class BytesFormatter(logging.Formatter):
    """Formats log messages as UTF-8 encoded XML bytes."""

    def format(self, record: logging.LogRecord) -> bytes:
        return f"<{record.tag}>{record.getMessage()}</{record.tag}>".encode()
