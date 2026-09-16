"""OpenAI Audio API transcription of an existing .m4a. No local Whisper."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pipeline.models import Transcript, TranscriptSegment

log = logging.getLogger(__name__)

DEFAULT_MODEL = "whisper-1"
#: Rotate before a chunk reaches this size. Stays inside the 20–22 MB target.
TARGET_CHUNK_BYTES = 21 * 1024 * 1024
#: Hard cap per chunk; also the "send the original as one file" threshold.
MAX_CHUNK_BYTES = 22 * 1024 * 1024


def _openai_client(api_key: str | None, client: Any | None) -> Any:
    if client is not None:
        return client
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            'The OpenAI client is not installed. pip install -e ".[classify]"'
        ) from exc
    options: dict[str, Any] = {}
    if api_key:
        options["api_key"] = api_key
    return OpenAI(**options)


def _result_to_transcript(result: Any, *, model: str, offset_s: float = 0.0) -> Transcript:
    text = getattr(result, "text", "") or ""
    duration = float(getattr(result, "duration", 0.0) or 0.0)
    raw_segments = getattr(result, "segments", None) or []
    segments = [
        TranscriptSegment(
            start_s=float(getattr(seg, "start", 0.0) or 0.0) + offset_s,
            end_s=float(getattr(seg, "end", 0.0) or 0.0) + offset_s,
            text=str(getattr(seg, "text", "") or "").strip(),
        )
        for seg in raw_segments
        if str(getattr(seg, "text", "") or "").strip()
    ]
    if not segments and text.strip():
        segments = [
            TranscriptSegment(
                start_s=offset_s,
                end_s=offset_s + duration,
                text=text.strip(),
            )
        ]
    return Transcript(
        segments=segments,
        language=str(getattr(result, "language", "") or "en"),
        duration_s=duration,
        model=model,
        provider="openai_audio",
    )


def _transcribe_file(client: Any, path: Path, *, model: str, offset_s: float = 0.0) -> Transcript:
    with path.open("rb") as fh:
        result = client.audio.transcriptions.create(
            model=model,
            file=fh,
            response_format="verbose_json",
        )
    return _result_to_transcript(result, model=model, offset_s=offset_s)


def merge_transcripts(parts: list[Transcript], *, model: str) -> Transcript:
    segments: list[TranscriptSegment] = []
    duration = 0.0
    language = "en"
    for part in parts:
        segments.extend(part.segments)
        duration += part.duration_s
        language = part.language or language
    return Transcript(
        segments=segments,
        language=language,
        duration_s=duration,
        model=model,
        provider="openai_audio",
    )


def _packet_seconds(packet: Any, time_base: Any) -> float | None:
    if packet.pts is None or time_base is None:
        return None
    return float(packet.pts * time_base)


def split_for_openai(
    path: Path,
    dest_dir: Path,
    *,
    target_bytes: int = TARGET_CHUNK_BYTES,
    max_bytes: int = MAX_CHUNK_BYTES,
) -> list[tuple[Path, float]]:
    """Remux the existing .m4a into the fewest .m4a pieces that stay under `max_bytes`.

    Split points are file size, not a fixed duration. Drive file is untouched.
    """
    try:
        import av
    except ImportError as exc:
        raise RuntimeError(
            "this .m4a is larger than the OpenAI transcription size limit; "
            "install PyAV so it can be split into .m4a chunks"
        ) from exc

    dest_dir.mkdir(parents=True, exist_ok=True)
    incoming = av.open(str(path))
    if not incoming.streams.audio:
        incoming.close()
        raise RuntimeError(f"{path.name} has no audio stream")
    in_stream = incoming.streams.audio[0]
    time_base = in_stream.time_base
    chunks: list[tuple[Path, float]] = []
    outgoing = None
    out_stream = None
    bytes_in_chunk = 0
    index = 0

    def close_chunk() -> None:
        nonlocal outgoing, out_stream, bytes_in_chunk
        if outgoing is not None:
            outgoing.close()
            outgoing = None
            out_stream = None
        bytes_in_chunk = 0

    def open_chunk(start_s: float) -> None:
        nonlocal outgoing, out_stream, index, bytes_in_chunk
        close_chunk()
        dest = dest_dir / f"{path.stem}.part{index:03d}.m4a"
        outgoing = av.open(str(dest), mode="w", format="ipod")
        out_stream = outgoing.add_stream_from_template(in_stream)
        chunks.append((dest, start_s))
        bytes_in_chunk = 0
        index += 1

    try:
        for packet in incoming.demux(in_stream):
            if packet.dts is None or packet.size is None:
                continue
            size = int(packet.size)
            start_s = _packet_seconds(packet, time_base) or 0.0
            would_exceed = bytes_in_chunk > 0 and (
                bytes_in_chunk + size > target_bytes or bytes_in_chunk + size > max_bytes
            )
            if outgoing is None or would_exceed:
                open_chunk(start_s)
            assert outgoing is not None and out_stream is not None
            packet.stream = out_stream
            outgoing.mux(packet)
            bytes_in_chunk += size
    finally:
        close_chunk()
        incoming.close()

    chunks = [(dest, offset) for dest, offset in chunks if dest.exists() and dest.stat().st_size > 0]
    if not chunks:
        raise RuntimeError(f"no audio could be remuxed from {path.name}")
    log.info(
        "split %s (%.1f MB) into %d size-capped .m4a chunks (target %.0f MB)",
        path.name,
        path.stat().st_size / (1024 * 1024),
        len(chunks),
        target_bytes / (1024 * 1024),
    )
    return chunks


def files_to_transcribe(path: Path) -> list[tuple[Path, float]]:
    """One file if it already fits; otherwise the fewest 20–22 MB .m4a chunks."""
    if path.stat().st_size <= MAX_CHUNK_BYTES:
        return [(path, 0.0)]
    return split_for_openai(path, path.parent / f"{path.stem}_chunks")


def format_chunk_plan(
    original: Path,
    pieces: list[tuple[Path, float]],
    *,
    target_bytes: int = TARGET_CHUNK_BYTES,
) -> str:
    lines = [
        f"ORIGINAL AUDIO SIZE: {original.stat().st_size / (1024 * 1024):.1f} MB",
        f"TARGET CHUNK SIZE: {target_bytes / (1024 * 1024):.0f} MB",
        f"NUMBER OF CHUNKS: {len(pieces)}",
        "",
        "CHUNK SIZES:",
    ]
    for i, (piece, _offset) in enumerate(pieces, start=1):
        lines.append(f"{i}: {piece.stat().st_size / (1024 * 1024):.1f} MB")
    return "\n".join(lines)


def transcribe_m4a(
    path: Path,
    *,
    api_key: str | None,
    model: str = DEFAULT_MODEL,
    client: Any | None = None,
    split_files: list[tuple[Path, float]] | None = None,
) -> Transcript:
    """Send the .m4a to OpenAI's audio transcriptions endpoint.

    Files over 22 MB are remuxed into .m4a chunks of about 21 MB.
    """
    client = _openai_client(api_key, client)
    pieces = split_files if split_files is not None else files_to_transcribe(path)
    parts = [
        _transcribe_file(client, piece, model=model, offset_s=offset)
        for piece, offset in pieces
    ]
    if len(parts) == 1:
        return parts[0]
    return merge_transcripts(parts, model=model)
