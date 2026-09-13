"""
Audio Loaders
~~~~~~~~~~~~~

Format-specific audio loading implementations

:copyright: (C) 2024 Auralis Team
:license: GPLv3, see LICENSE for more details.
"""

from .ffmpeg_loader import check_ffmpeg, check_ffprobe, load_with_ffmpeg, reject_protocol_path
from .soundfile_loader import load_with_soundfile

__all__ = [
    'load_with_soundfile',
    'load_with_ffmpeg',
    'check_ffmpeg',
    'check_ffprobe',
    'reject_protocol_path',
]
