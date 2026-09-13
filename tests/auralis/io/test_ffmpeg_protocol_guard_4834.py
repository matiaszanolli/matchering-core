# -*- coding: utf-8 -*-
"""
FFmpeg protocol guard rejects colon-only protocols, not just "://" (#4834)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The guard in `load_with_ffmpeg` only rejected literal `"://"` substrings, but
several FFmpeg/libavformat protocols (`pipe:0`, `concat:a|b`, `data:...`)
never contain `"://"`. A second ffprobe call site
(`unified_loader.get_audio_info` -> `_get_info_with_ffprobe`) had no guard at
all. Both now share `loaders.reject_protocol_path`, a generic
`^[a-zA-Z][a-zA-Z0-9+.-]+:` prefix check.
"""

from pathlib import Path

import pytest

from auralis.io.loaders import reject_protocol_path
from auralis.io.unified_loader import _get_info_with_ffprobe, get_audio_info
from auralis.utils.logging import ModuleError


class TestRejectProtocolPath:
    @pytest.mark.parametrize("path", [
        "pipe:0",
        "concat:/etc/passwd|song.mp3",
        "data:audio/mp3;base64,AAAA",
        "http://example.com/track.mp3",
        "https://example.com/track.mp3",
        "rtmp://example.com/live",
    ])
    def test_rejects_protocol_specifiers(self, path):
        with pytest.raises(ModuleError):
            reject_protocol_path(path)

    @pytest.mark.parametrize("path", [
        "/home/user/Music/Track: Remix.mp3",
        "/home/user/Music/song.mp3",
        "relative/path/song.mp3",
        "song.mp3",
    ])
    def test_allows_plain_paths_including_a_literal_colon(self, path):
        # Must not false-positive on a ':' that isn't a protocol prefix.
        reject_protocol_path(path)  # raises nothing

    def test_allows_windows_drive_letters(self):
        # A single-letter drive prefix ("C:") must not be misclassified as a
        # protocol -- every real FFmpeg protocol name is 2+ characters.
        reject_protocol_path(r"C:\Users\name\Music\song.mp3")


class TestFfprobeCallSiteGuarded:
    def test_ffprobe_call_site_rejects_a_protocol_anchored_path(self):
        # FFmpeg's own protocol parser only treats a "scheme:" prefix as a
        # protocol when it anchors the START of the whole argument string --
        # an absolute path like "/tmp/x/concat:a.mp3" is just an oddly named
        # regular file to FFmpeg, not a concat: invocation. So the realistic
        # exploit path (#4834's confused-deputy scenario) is a bare relative
        # path beginning with the scheme, which this constructs directly
        # against the actual guarded call site rather than routing through
        # get_audio_info()'s exists() check (which would require creating a
        # real "concat:" directory entry on disk).
        with pytest.raises(ModuleError, match="URL/protocol inputs are not allowed"):
            _get_info_with_ffprobe(Path("concat:a|b.mp3"))

    def test_get_audio_info_does_not_false_positive_on_a_colon_in_the_basename(self, tmp_path):
        # Regression (issue Test Plan #2): a literal ':' in an unrelated
        # position (e.g. "Track: Remix.mp3") is not a protocol prefix and
        # must still reach ffprobe rather than being rejected by the guard.
        # ffprobe still fails on this un-decodable 0-byte file, but that
        # failure must not be the guard's.
        benign = tmp_path / "Track: Remix.mp3"
        benign.write_bytes(b"\x00")

        info = get_audio_info(benign)

        assert "error" in info
        assert "URL/protocol" not in info["error"]
