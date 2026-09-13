"""
Path Security Utilities
~~~~~~~~~~~~~~~~~~~~~~~

Path validation and sanitization to prevent directory traversal attacks.

Fixes #2069: Path traversal in directory scanning endpoint

Trust model (#4799): Auralis is a single-user desktop app where directories
come from the user's own file picker, not an untrusted network client. Every
real directory entry point (``LibraryScanRequest.validate_directory_paths``
in ``schemas.py``, ``POST /api/settings/scan-folders``) validates through
``validate_user_chosen_directory()`` / ``validate_directory_list()``, which
enforce basic safety (no traversal, no operating-system roots, must exist and
be readable) but
deliberately do NOT restrict to a fixed allowlist — the user's choice is
trusted. Registering a folder via ``register_allowed_directory()`` then
widens the allowlist ``validate_file_path()`` consults for the rest of the
session. There is intentionally no separate allowlist-enforcing directory
validator in this module — an earlier one (``validate_scan_path``) existed
unused alongside this posture and was removed as dead code rather than left
implying a containment guarantee the app doesn't actually enforce.

:copyright: (C) 2024 Auralis Team
:license: GPLv3, see LICENSE for more details.
"""

import functools
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default allowed base directories — intentionally excludes bare Path.home()
# to prevent traversal to sensitive files like ~/.ssh or ~/.gnupg (#2562).
DEFAULT_ALLOWED_DIRS = [
    Path.home() / "Music",      # Standard music directory
    Path.home() / "Documents",  # Documents directory
]

# Extra directories registered at runtime (populated by startup.py after
# reading user-configured scan folders from the database).
_extra_allowed_dirs: list[Path] = []


def _system_directory_roots() -> tuple[Path, ...]:
    """Return OS-owned directory trees that must never become scan roots."""
    if os.name == "nt":
        candidates = {
            Path(os.environ.get("SystemRoot", r"C:\Windows")),
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
            Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
        }
    else:
        candidates = {
            Path("/"), Path("/bin"), Path("/boot"), Path("/dev"),
            Path("/etc"), Path("/proc"), Path("/root"), Path("/sbin"),
            Path("/sys"), Path("/usr"), Path("/var"),
        }
        if sys.platform == "darwin":
            candidates.update({Path("/Library"), Path("/System")})
    return tuple(path.resolve() for path in candidates)


def _is_system_directory(path: Path) -> bool:
    filesystem_root = Path(path.anchor)
    return any(
        path == root or (root != filesystem_root and path.is_relative_to(root))
        for root in _system_directory_roots()
    )


def register_allowed_directory(directory: Path) -> None:
    """Register an additional allowed directory (called during startup or when adding scan folders)."""
    resolved = directory.resolve()
    if resolved not in _extra_allowed_dirs:
        _extra_allowed_dirs.append(resolved)


def unregister_allowed_directory(directory: Path) -> None:
    """Remove a directory registered via `register_allowed_directory` (fixes #3842).

    Called when a user removes a scan folder, so `validate_file_path()` stops
    trusting it for the rest of the session instead of only on next restart.
    """
    resolved = directory.resolve()
    if resolved in _extra_allowed_dirs:
        _extra_allowed_dirs.remove(resolved)


def clear_extra_allowed_directories() -> None:
    """Remove all runtime-registered extra directories (fixes #3842).

    Called on settings reset, since `reset_to_defaults` wipes the configured
    `scan_folders` list — none of the previously-registered extra directories
    should remain implicitly trusted afterward.
    """
    _extra_allowed_dirs.clear()


class PathValidationError(Exception):
    """Raised when path validation fails."""
    pass


def _logs_rejections(fn: Callable[..., Path]) -> Callable[..., Path]:
    """Emit exactly one warning whenever the wrapped validator rejects a path.

    #4925: rejections used to be logged only by whichever caller remembered
    to. Six call sites did, six did not — so a traversal probe against, say,
    ``/api/settings/scan-folder`` left no trace at all, just a 400 to the
    client, and repeated probing could not be noticed after the fact. Only the
    SUCCESS path was logged here, at debug.

    Wrapping rather than editing each ``raise`` site keeps the original
    exception and its traceback untouched (bare ``raise``), and means a
    validator that grows a new rejection branch is covered automatically
    instead of relying on someone remembering this rule.

    The corollary: callers must NOT log their own warning or the line doubles.
    Callers that know something the validator cannot — which track, which
    request parameter — pass ``context`` instead, so the single line still
    carries it.

    Typed as ``Callable[..., Path]`` deliberately: ``context`` is injected by
    the wrapper and is not part of the wrapped function's own signature, so a
    precise ``ParamSpec`` would reject every caller that passes it.
    """
    @functools.wraps(fn)
    def wrapper(*args: Any, context: str | None = None, **kwargs: Any) -> Path:
        try:
            return fn(*args, **kwargs)
        except PathValidationError as e:
            logger.warning(
                "Path validation rejected%s: %s",
                f" ({context})" if context else "",
                e,
            )
            raise
    return wrapper


def get_allowed_directories() -> list[Path]:
    """
    Get list of allowed base directories for scanning.

    Returns:
        List of allowed directory paths

    The allow-list is DEFAULT_ALLOWED_DIRS (home plus the standard music
    folders) widened by two configured sources: XDG_MUSIC_DIR, and the scan
    folders registered at runtime into _extra_allowed_dirs. Startup seeds the
    latter from the settings-backed scan-folder allowlist, so this list is
    already configuration-driven — it is not a hardcoded stand-in awaiting one
    (#5145). Non-existent entries are dropped rather than resolved.
    """
    allowed_dirs = DEFAULT_ALLOWED_DIRS.copy()

    # Add XDG_MUSIC_DIR if available (Linux)
    xdg_music = os.environ.get('XDG_MUSIC_DIR')
    if xdg_music:
        allowed_dirs.append(Path(xdg_music))

    # Include user-configured scan folders registered at runtime
    allowed_dirs.extend(_extra_allowed_dirs)

    # Resolve all paths to absolute and normalize
    return [path.resolve() for path in allowed_dirs if path.exists()]


@_logs_rejections
def validate_file_path(
    filepath: str,
    allowed_base_dirs: list[Path] | None = None
) -> Path:
    """
    Validate and sanitize a file path against allowed directories.

    Security checks:
    - Path must be absolute or relative (will be resolved)
    - No path traversal sequences (../)
    - Must fall within allowed base directories
    - Must exist
    - Must be readable

    Args:
        filepath: File path to validate
        allowed_base_dirs: Allowed base directories (default: user home, Music, Documents)

    Returns:
        Resolved absolute Path if valid

    Raises:
        PathValidationError: If path fails validation
    """
    if not filepath:
        raise PathValidationError("File path cannot be empty")

    try:
        path = Path(filepath)
    except (ValueError, TypeError) as e:
        raise PathValidationError(f"Invalid path format: {e}")

    try:
        resolved_path = path.resolve()
    except (OSError, RuntimeError) as e:
        raise PathValidationError(f"Failed to resolve path: {e}")

    if ".." in Path(filepath).parts:
        raise PathValidationError(
            "Path traversal sequences (..) are not allowed."
        )

    if allowed_base_dirs is None:
        allowed_base_dirs = get_allowed_directories()

    is_allowed = False
    for base_dir in allowed_base_dirs:
        try:
            resolved_path.relative_to(base_dir)
            is_allowed = True
            break
        except ValueError:
            continue

    if not is_allowed:
        # Full detail (resolved path + every allowed directory -- the user's
        # entire configured library layout) at DEBUG only, not in the
        # exception's own string form (#4807). Four HTTP routes used to
        # reflect str(e) verbatim into a 400 body; they now return a fixed
        # generic message regardless of what PathValidationError says, but
        # keeping the enumeration out of the exception itself removes the
        # fragility at its root instead of relying on every future caller
        # remembering not to reflect it. Matches startup.py's existing
        # DEBUG-for-sensitive-paths convention (#3844/#4376).
        allowed_dirs_str = ", ".join(str(d) for d in allowed_base_dirs)
        logger.debug(
            f"Path '{resolved_path}' is outside allowed directories. "
            f"Allowed directories: {allowed_dirs_str}"
        )
        raise PathValidationError("Path is outside allowed directories")

    if not resolved_path.exists():
        raise PathValidationError(f"File does not exist: {resolved_path}")

    if not resolved_path.is_file():
        raise PathValidationError(f"Path is not a file: {resolved_path}")

    if not os.access(resolved_path, os.R_OK):
        raise PathValidationError(f"File is not readable: {resolved_path}")

    # Debug, not info (#3844): validators run very frequently and the
    # resolved path is sensitive (the user's media library layout) — at INFO
    # it floods logs and gets persisted to electron-log on disk. This one
    # runs 5x per /api/metadata/tracks/{id} request.
    logger.debug(f"File path validation successful: {resolved_path}")
    return resolved_path


@_logs_rejections
def validate_user_chosen_directory(directory: str) -> Path:
    """
    Validate a directory path explicitly chosen by the user (e.g., via file picker).

    Auralis is a single-user desktop app. When the user explicitly selects a
    folder to scan, we trust their choice outside operating-system-owned
    directory trees and enforce basic safety checks (no traversal, must exist,
    must be readable) without restricting to predefined media directories.

    Args:
        directory: Directory path chosen by the user

    Returns:
        Resolved absolute Path object if valid

    Raises:
        PathValidationError: If path fails basic safety checks
    """
    if not directory:
        raise PathValidationError("Directory path cannot be empty")

    try:
        path = Path(directory)
    except (ValueError, TypeError) as e:
        raise PathValidationError(f"Invalid path format: {e}")

    try:
        resolved_path = path.resolve()
    except (OSError, RuntimeError) as e:
        raise PathValidationError(f"Failed to resolve path: {e}")

    if ".." in Path(directory).parts:
        raise PathValidationError(
            "Path traversal sequences (..) are not allowed. "
            "Please use absolute paths."
        )

    if _is_system_directory(resolved_path):
        raise PathValidationError(
            "Operating-system directories cannot be used as library scan roots"
        )

    if not resolved_path.exists():
        raise PathValidationError(f"Directory does not exist: {resolved_path}")

    if not resolved_path.is_dir():
        raise PathValidationError(f"Path is not a directory: {resolved_path}")

    if not os.access(resolved_path, os.R_OK):
        raise PathValidationError(f"Directory is not readable: {resolved_path}")

    # Debug, not info (#3844): avoid persisting the user's chosen folder path
    # to logs on every validation.
    logger.debug(f"User-chosen directory validated: {resolved_path}")
    return resolved_path


def validate_directory_list(directories: list[str]) -> list[str]:
    """Validate a batch of user-chosen directory paths with
    ``validate_user_chosen_directory``, returning the resolved, stringified
    list.

    Shared by every write path that persists ``scan_folders`` — the Pydantic
    ``LibraryScanRequest.directories`` field_validator and ``PUT
    /api/settings``'s ``SettingsUpdateRequest.scan_folders`` handling (#4765)
    — so an entry rejected by one is rejected by the other identically,
    instead of one validating and the other writing it through unchecked.
    Raises on the first invalid entry; a mixed batch does not silently drop
    the bad one and proceed with the rest.

    Raises:
        PathValidationError: naming the offending path, on the first invalid entry.
    """
    validated = []
    for directory in directories:
        try:
            validated.append(str(validate_user_chosen_directory(directory)))
        except PathValidationError as e:
            raise PathValidationError(f"Invalid directory path '{directory}': {e}") from e
    return validated


def sanitize_path_for_response(path: Path | str) -> str:
    """
    Sanitize a file path for inclusion in API responses.

    Converts absolute paths to be relative to user's home directory
    to avoid exposing full system paths.

    Args:
        path: File path to sanitize

    Returns:
        Sanitized path string (relative to home if possible)

    Examples:
        >>> sanitize_path_for_response("/home/user/Music/song.mp3")
        "~/Music/song.mp3"
        >>> sanitize_path_for_response("/var/system/file")
        "/var/system/file"  # Not in home, return as-is
    """
    path_obj = Path(path).resolve()
    home = Path.home()

    try:
        # Try to make relative to home directory
        relative = path_obj.relative_to(home)
        return f"~/{relative}"
    except ValueError:
        # Path is not in home directory, return as-is
        # (This shouldn't happen for music files, but handle gracefully)
        return str(path_obj)
