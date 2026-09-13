"""
Auralis Logging System
~~~~~~~~~~~~~~~~~~~~~~

Logging and debug utilities

:copyright: (C) 2024 Auralis Team
:license: GPLv3, see LICENSE for more details.

Refactored from Matchering 2.0 by Sergree and contributors
"""

import logging as _stdlib_logging
from collections.abc import Callable

# Control characters that must not reach a log line verbatim: C0 range
# (0x00-0x1F) except tab, plus DEL (0x7F) and the C1 range (0x80-0x9F).
# Newlines/carriage returns here let attacker-controlled metadata forge log
# lines; ESC (0x1B) and friends can corrupt a terminal log viewer (#4363).
_LOG_UNSAFE_CHARS = frozenset(
    chr(c) for c in range(0x00, 0x20) if c != 0x09
) | {chr(0x7F)} | frozenset(chr(c) for c in range(0x80, 0xA0))


def sanitize_log_value(value: object, max_length: int = 500) -> str:
    """Make an untrusted value safe to interpolate into a single log line.

    Track/tag metadata and client-supplied filenames can contain ``\\r\\n``
    (forging extra log lines) or terminal control sequences (corrupting a log
    viewer). This coerces ``value`` to ``str``, escapes every unsafe control
    character as its ``\\xNN`` form, and truncates to ``max_length`` so a huge
    tag can't flood the log. The result is always a single printable line.

    Args:
        value: Any value (coerced via ``str``); typically metadata/filename.
        max_length: Maximum length before truncation (appends an ellipsis).

    Returns:
        A single-line, control-character-free string safe for logging.
    """
    text = str(value)
    if len(text) > max_length:
        text = text[:max_length] + "…"
    if not any(ch in _LOG_UNSAFE_CHARS for ch in text):
        return text  # fast path: nothing to escape
    return "".join(
        f"\\x{ord(ch):02x}" if ch in _LOG_UNSAFE_CHARS else ch for ch in text
    )


class Code:
    """Log message codes"""
    INFO_LOADING = "Loading audio files..."
    INFO_EXPORTING = "Exporting results..."
    INFO_COMPLETED = "Processing completed successfully"
    ERROR_VALIDATION = "Validation error"
    ERROR_LOADING = "Error loading audio file"
    ERROR_INVALID_AUDIO = "Invalid audio file format"
    ERROR_FILE_NOT_FOUND = "File not found"
    ERROR_EMPTY_AUDIO = "Empty audio file"
    ERROR_INVALID_SAMPLE_RATE = "Invalid sample rate"
    ERROR_FFMPEG_NOT_FOUND = "FFmpeg not found"
    ERROR_FFMPEG_CONVERSION = "FFmpeg conversion error"
    ERROR_FFMPEG_TIMEOUT = "FFmpeg conversion timeout"
    ERROR_TRUNCATED_FILE = "Truncated audio file"
    WARNING_TRUNCATED_FILE = "Audio file may be truncated"
    ERROR_EMPTY_FILE = "Empty file"
    ERROR_UNSUPPORTED_FORMAT = "Unsupported format"
    ERROR_CORRUPTED = "Corrupted or unsupported audio file"
    ERROR_NAN_DETECTED = "NaN or Inf values detected in audio"
    WARNING_NAN_DETECTED = "NaN or Inf values detected and repaired"


class ModuleError(Exception):
    """Custom exception for module errors.

    `path`, when given, carries the filesystem path involved in the failure
    as a structured attribute rather than interpolated into the message
    string. FFmpeg/ffprobe raise sites used to embed the absolute path (and,
    separately, raw subprocess stderr) directly into the exception's string
    form (#4806) — nothing currently forwards `str(exc)` to an HTTP/WS
    client, but a future "surface a more specific ModuleError message"
    change would have started doing exactly that. A caller that needs the
    path for its own DEBUG-level logging should read `.path`, not parse it
    back out of `str(exc)`.
    """
    def __init__(self, code: str, path: str | None = None) -> None:
        self.code = code
        self.path = path
        super().__init__(f"Module error: {code}")


# Global log handler and level
_log_handler: Callable[[str], None] | None = None
_log_level: str = "INFO"


def set_log_handler(handler: Callable[[str], None] | None) -> None:
    """Set the global log handler function"""
    global _log_handler
    _log_handler = handler


def set_log_level(level: str) -> None:
    """Set the global log level"""
    global _log_level
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
    if level.upper() in valid_levels:
        _log_level = level.upper()


def get_log_level() -> str:
    """Get the current log level"""
    return _log_level


def debug(message: str) -> None:
    """Log a debug message"""
    if _log_handler:
        _log_handler(f"DEBUG: {sanitize_log_value(message)}")


def info(message: str) -> None:
    """Log an info message"""
    if _log_handler:
        _log_handler(f"INFO: {sanitize_log_value(message)}")


def warning(message: str) -> None:
    """Log a warning message"""
    if _log_handler:
        _log_handler(f"WARNING: {sanitize_log_value(message)}")


def error(message: str) -> None:
    """Log an error message"""
    if _log_handler:
        _log_handler(f"ERROR: {sanitize_log_value(message)}")


def debug_line() -> None:
    """Log a debug separator line"""
    if _log_handler:
        _log_handler("-" * 50)


# #4828: the explicit sanitize_log_value() call sites from #4363 (4 files) left
# dozens of other sites across auralis/ and auralis-web/backend/ interpolating
# untrusted filenames/tags into log messages unsanitized — sanitize_log_value()
# was never a global hook, just a function every call site had to opt into.
#
# Rather than touching every site individually (an unbounded, easy-to-miss
# list), sanitization is applied centrally at the two places log messages
# actually get emitted:
#   - debug/info/warning/error above, for this module's own callers.
#   - the stdlib `logging` LogRecordFactory below, for the rest of the
#     codebase (mainly auralis-web/backend/, which uses
#     `logging.getLogger(__name__)` throughout rather than this module).
# Both are idempotent (sanitize_log_value's fast path is a no-op on text with
# no unsafe characters), so this is safe even where a call site already
# sanitizes explicitly.
_original_log_record_factory = _stdlib_logging.getLogRecordFactory()


def _sanitizing_log_record_factory(*args: object, **kwargs: object) -> _stdlib_logging.LogRecord:
    """LogRecordFactory wrapper that sanitizes a record's message/args in place.

    Covers %-style logging (``logger.info("...%s...", value)``) as well as
    pre-interpolated f-strings, without touching exception tracebacks (those
    are formatted separately by the handler's Formatter, not via getMessage()).
    """
    record = _original_log_record_factory(*args, **kwargs)
    if isinstance(record.msg, str):
        record.msg = sanitize_log_value(record.msg)
    if isinstance(record.args, tuple):
        record.args = tuple(
            sanitize_log_value(arg) if isinstance(arg, str) else arg
            for arg in record.args
        )
    return record


_stdlib_logging.setLogRecordFactory(_sanitizing_log_record_factory)