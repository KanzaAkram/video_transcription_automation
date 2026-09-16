"""Audio extraction via ffmpeg.

16 kHz mono PCM is what Whisper-family models consume internally, so producing
it once up front avoids the model resampling on every run — and avoids redoing
that work on a retry.

The video file is never downloaded where it can be avoided. Zoom exposes an
audio-only M4A alongside the MP4 at roughly a tenth the size, and repositioning
a file in Drive is a metadata operation that needs no content at all. See
ADR-003.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


class FFmpegMissing(RuntimeError):
    """ffmpeg is not on PATH.

    Raised early with an actionable message, because the alternative is a
    confusing FileNotFoundError from deep inside a worker.
    """


def _require(binary: str) -> str:
    path = shutil.which(binary)
    if path is None:
        raise FFmpegMissing(
            f"{binary} not found on PATH. Install ffmpeg and restart the worker "
            "(Windows: winget install Gyan.FFmpeg)."
        )
    return path


def _extract_with_pyav(source: Path, dest: Path, *, sample_rate: int) -> Path:
    """16 kHz mono PCM via PyAV when the ffmpeg binary is not installed."""
    try:
        import av
        from av.audio.resampler import AudioResampler
    except ImportError as exc:
        raise FFmpegMissing(
            "ffmpeg not found on PATH and PyAV is not installed. "
            "Install ffmpeg (Windows: winget install Gyan.FFmpeg) "
            'or pip install -e ".[asr]".'
        ) from exc

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    incoming = av.open(str(source))
    if not incoming.streams.audio:
        incoming.close()
        raise RuntimeError(f"{source.name} has no audio stream")

    def as_frames(value):
        if value is None:
            return []
        return value if isinstance(value, list) else [value]

    outgoing = av.open(str(tmp), mode="w", format="wav")
    ostream = outgoing.add_stream("pcm_s16le", rate=sample_rate)
    resampler = AudioResampler(format="s16", layout="mono", rate=sample_rate)
    try:
        for frame in incoming.decode(audio=0):
            for converted in as_frames(resampler.resample(frame)):
                if converted is None:
                    continue
                for packet in ostream.encode(converted):
                    outgoing.mux(packet)
        for converted in as_frames(resampler.resample(None)):
            if converted is None:
                continue
            for packet in ostream.encode(converted):
                outgoing.mux(packet)
        for packet in ostream.encode(None):
            outgoing.mux(packet)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        outgoing.close()
        incoming.close()
    tmp.rename(dest)
    return dest


def extract_audio(source: Path, dest: Path, *, sample_rate: int = 16000) -> Path:
    """Extract mono PCM audio at `sample_rate`.

    Idempotent: if `dest` already exists and is non-empty, it is reused. That
    is what makes a retry after a transcription failure cheap.
    """
    if dest.exists() and dest.stat().st_size > 0:
        log.debug("audio already extracted: %s", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        ffmpeg = _require("ffmpeg")
    except FFmpegMissing:
        log.info("ffmpeg not on PATH; extracting audio with PyAV")
        return _extract_with_pyav(source, dest, sample_rate=sample_rate)

    cmd = [
        ffmpeg, "-nostdin", "-y",
        "-i", str(source),
        "-vn",                       # drop video
        "-ac", "1",                  # mono
        "-ar", str(sample_rate),     # 16 kHz
        "-c:a", "pcm_s16le",
        str(tmp),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        # ffmpeg's error is on the last few stderr lines; the rest is banner.
        tail = "\n".join(result.stderr.strip().splitlines()[-5:])
        raise RuntimeError(f"ffmpeg failed extracting audio from {source.name}:\n{tail}")

    # Rename only on success, so a crash never leaves a truncated file that a
    # later run would treat as complete.
    tmp.rename(dest)
    return dest


def probe_duration(source: Path) -> float | None:
    """Duration in seconds, or None if it cannot be determined."""
    try:
        ffprobe = _require("ffprobe")
    except FFmpegMissing:
        return None

    result = subprocess.run(
        [ffprobe, "-v", "quiet", "-print_format", "json", "-show_format", str(source)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return None
    try:
        return float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return None
