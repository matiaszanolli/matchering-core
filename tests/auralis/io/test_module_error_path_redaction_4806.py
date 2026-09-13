# -*- coding: utf-8 -*-
"""
ModuleError no longer embeds raw stderr or absolute paths (#4806)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ModuleError messages raised from the FFmpeg/ffprobe path used to embed the
whole raw FFmpeg/ffprobe stderr (always echoing the full input path plus the
local build's config banner) and, separately, the absolute file_path,
directly in the exception's string form. Nothing currently forwards that
string to an HTTP/WS client (`_safe_error_message` discards exception text
per BE7-2), but that safety was accidental: the natural fix for BE7-2
("surface a more specific message for ModuleError") is exactly the change
that would have started reflecting this raw text into a client-facing
string.

ModuleError now takes an optional `path` kwarg carrying the path as a
structured attribute instead, and the FFmpeg/ffprobe raw-stderr raise sites
redact stderr to its last line before it reaches the message.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from auralis.io.loaders.ffmpeg_loader import redact_subprocess_output
from auralis.io.unified_loader import _get_info_with_ffprobe, get_audio_info, load_audio
from auralis.utils.logging import Code, ModuleError, set_log_handler


@contextmanager
def _captured_logs():
    """auralis/'s debug()/info()/warning() route through a pluggable
    set_log_handler(...) callback, not the stdlib `logging` module -- caplog
    can never see them. Mirrors the pattern already used elsewhere in this
    suite (e.g. tests/auralis/io_module/test_unified_loader.py), but with
    guaranteed cleanup so a failed assertion here can't leak a handler into
    later tests."""
    messages: list[str] = []
    set_log_handler(messages.append)
    try:
        yield messages
    finally:
        set_log_handler(None)

def _fake_stderr(path: str) -> str:
    """Realistic FFmpeg/ffprobe stderr: a build-config banner, then a last
    line reading '<input path>: <reason>' -- FFmpeg's own error format puts
    the echoed input path on the SAME line as the failure reason, not just
    in the banner above, which is exactly what a naive last-line-only
    truncation would still leak (#4806)."""
    return (
        "ffmpeg version 6.1.1 Copyright (c) 2000-2023 the FFmpeg developers\n"
        "  built with gcc 13.2.1\n"
        "  configuration: --enable-gpl --enable-libx264 --prefix=/usr\n"
        "[mp3 @ 0x5f2a1b2c3d40] Header missing\n"
        f"{path}: Invalid data found when processing input\n"
    )


_SAMPLE_PATH = "/home/someuser/Music/private-folder/track.mp3"
_FAKE_STDERR = _fake_stderr(_SAMPLE_PATH)


class TestModuleErrorStructuredPath:
    def test_path_kwarg_is_stored_as_an_attribute(self):
        exc = ModuleError(Code.ERROR_FILE_NOT_FOUND, path="/home/someuser/secret/track.mp3")
        assert exc.path == "/home/someuser/secret/track.mp3"

    def test_path_defaults_to_none(self):
        exc = ModuleError(Code.ERROR_FILE_NOT_FOUND)
        assert exc.path is None

    def test_path_is_not_interpolated_into_the_string_form(self):
        exc = ModuleError(Code.ERROR_FILE_NOT_FOUND, path="/home/someuser/secret/track.mp3")
        assert "/home/someuser/secret" not in str(exc)


class TestRedactSubprocessOutput:
    def test_keeps_only_the_last_line(self):
        redacted = redact_subprocess_output(_FAKE_STDERR)
        assert "gcc" not in redacted
        assert "--enable-gpl" not in redacted
        assert "Invalid data found when processing input" in redacted

    def test_last_line_still_carries_the_path_without_known_path(self):
        """Documents why callers must pass known_path: the last line alone
        does NOT strip a path FFmpeg's own error format put right there."""
        redacted = redact_subprocess_output(_FAKE_STDERR)
        assert _SAMPLE_PATH in redacted

    def test_drops_the_absolute_path_when_known_path_is_given(self):
        redacted = redact_subprocess_output(_FAKE_STDERR, known_path=_SAMPLE_PATH)
        assert _SAMPLE_PATH not in redacted
        assert "Invalid data found when processing input" in redacted

    def test_empty_or_none_input_is_handled(self):
        assert redact_subprocess_output("") == "(no output)"
        assert redact_subprocess_output(None) == "(no output)"

    def test_bounds_an_extremely_long_last_line(self):
        redacted = redact_subprocess_output("x" * 5000, max_chars=200)
        assert len(redacted) <= 201  # ellipsis + max_chars


class TestFileNotFoundRaiseSites:
    """unified_loader.load_audio / get_audio_info -- FIX site 1 & 2."""

    def test_load_audio_missing_file_carries_path_not_in_string(self, tmp_path):
        missing = tmp_path / "does-not-exist.wav"
        with pytest.raises(ModuleError) as exc_info:
            load_audio(str(missing))
        assert str(missing) not in str(exc_info.value)
        assert exc_info.value.path == str(missing)

    def test_get_audio_info_missing_file_carries_path_not_in_string(self, tmp_path):
        missing = tmp_path / "does-not-exist.wav"
        with pytest.raises(ModuleError) as exc_info:
            get_audio_info(str(missing))
        assert str(missing) not in str(exc_info.value)
        assert exc_info.value.path == str(missing)


class TestFfprobeRawStderrRaiseSite:
    """unified_loader._get_info_with_ffprobe -- FIX site 3."""

    def test_ffprobe_failure_redacts_stderr_and_carries_path(self, tmp_path):
        target = tmp_path / "corrupt.mp3"
        target.write_bytes(b"\x00")

        fake_result = MagicMock(returncode=1, stderr=_fake_stderr(str(target)))
        with (
            patch("auralis.io.unified_loader.check_ffprobe", return_value=True),
            patch("auralis.io.unified_loader.subprocess.run", return_value=fake_result),
            _captured_logs() as logs,
        ):
            with pytest.raises(ModuleError) as exc_info:
                _get_info_with_ffprobe(target)

        msg = str(exc_info.value)
        assert "gcc" not in msg
        assert "--enable-gpl" not in msg
        assert str(target) not in msg
        assert exc_info.value.path == str(target)

        # Full stderr still reaches DEBUG -- no loss of diagnostic detail.
        assert any("--enable-gpl" in m for m in logs)


class TestFfmpegRawStderrRaiseSite:
    """ffmpeg_loader.load_with_ffmpeg's conversion-failure raise -- FIX site 4."""

    def test_conversion_failure_redacts_stderr_and_carries_path(self, tmp_path):
        target = tmp_path / "song.mp3"
        target.write_bytes(b"\x00" * 4096)  # non-empty so the min-duration guard doesn't fire first

        fake_probe = {"duration": 1.0, "sample_rate": 44100, "channels": 2}
        fake_result = MagicMock(returncode=1, stderr=_fake_stderr(str(target)))

        with (
            patch("auralis.io.loaders.ffmpeg_loader.check_ffprobe", return_value=True),
            patch("auralis.io.loaders.ffmpeg_loader._probe_audio", return_value=fake_probe),
            patch("auralis.io.loaders.ffmpeg_loader._run_ffmpeg_cancellable", return_value=fake_result),
            _captured_logs() as logs,
        ):
            from auralis.io.loaders.ffmpeg_loader import load_with_ffmpeg

            with pytest.raises(ModuleError) as exc_info:
                load_with_ffmpeg(target)

        msg = str(exc_info.value)
        assert "gcc" not in msg
        assert "--enable-gpl" not in msg
        assert str(target) not in msg
        assert exc_info.value.path == str(target)

        assert any("--enable-gpl" in m for m in logs)
