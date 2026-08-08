"""Transcript output formats.

Four artifacts from one transcription:

  * ``.md``    readable, timestamped — useful to students as a study aid
  * ``.srt``   subtitles, playable directly against the video in Drive
  * ``.vtt``   the web equivalent
  * ``.json``  sidecar with full provenance

The sidecar is what makes the archive auditable a year later, and what allows
reclassification after a taxonomy improvement **without re-transcribing** —
which matters, because transcription is the expensive step and classification
is not.
"""

from __future__ import annotations

import json
from typing import Any

from pipeline.models import Recording, Transcript, utcnow


def _clock(seconds: float) -> str:
    total = int(seconds)
    h, m, s = total // 3600, (total % 3600) // 60, total % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _srt_time(seconds: float) -> str:
    total = int(seconds)
    ms = int((seconds - total) * 1000)
    h, m, s = total // 3600, (total % 3600) // 60, total % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _vtt_time(seconds: float) -> str:
    return _srt_time(seconds).replace(",", ".")


def render_markdown(
    recording: Recording,
    transcript: Transcript,
    *,
    title: str,
    heading_interval_s: int = 300,
    include_timestamps: bool = True,
) -> str:
    """Readable transcript with periodic timestamp headings.

    Headings every few minutes rather than per segment: a heading on every
    line is unreadable, and no headings at all makes the file impossible to
    navigate against the recording.
    """
    lines = [f"# {title}", ""]
    lines.append(f"**Recorded:** {recording.recorded_at:%A, %d %B %Y}")
    if recording.teacher:
        lines.append(f"**Teacher:** {recording.teacher}")
    lines.append(f"**Duration:** {_clock(transcript.duration_s)}")
    lines.append(
        f"**Transcribed by:** {transcript.provider} / {transcript.model} "
        f"({transcript.language})"
    )
    lines.append("")
    lines.append("> Automatically generated. May contain errors, especially in "
                 "technical terms and proper nouns.")
    lines.append("")
    lines.append("---")
    lines.append("")

    next_heading = 0.0
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            lines.append(" ".join(buffer))
            lines.append("")
            buffer.clear()

    for segment in transcript.segments:
        if include_timestamps and segment.start_s >= next_heading:
            flush()
            lines.append(f"## {_clock(segment.start_s)}")
            lines.append("")
            next_heading = segment.start_s + heading_interval_s
        buffer.append(segment.text.strip())
    flush()

    return "\n".join(lines)


def render_srt(transcript: Transcript) -> str:
    blocks = []
    for i, segment in enumerate(transcript.segments, start=1):
        blocks.append(
            f"{i}\n{_srt_time(segment.start_s)} --> {_srt_time(segment.end_s)}\n"
            f"{segment.text.strip()}\n"
        )
    return "\n".join(blocks)


def render_vtt(transcript: Transcript) -> str:
    blocks = ["WEBVTT", ""]
    for segment in transcript.segments:
        blocks.append(f"{_vtt_time(segment.start_s)} --> {_vtt_time(segment.end_s)}")
        blocks.append(segment.text.strip())
        blocks.append("")
    return "\n".join(blocks)


def render_sidecar(recording: Recording, extra: dict[str, Any] | None = None) -> str:
    """Full provenance for one processed recording.

    Records not just the answer but the alternatives considered, so a later
    reviewer can tell a close call from a confident one — and so a taxonomy
    change can be replayed against stored transcripts rather than re-running
    the GPU.
    """
    classification = recording.classification
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": utcnow().isoformat(),
        "recording": {
            "id": recording.id,
            "source": recording.source,
            "source_id": recording.source_id,
            "subject_id": recording.subject_id,
            "original_filename": recording.original_filename,
            "recorded_at": recording.recorded_at.isoformat(),
            "duration_s": recording.duration_s,
            "teacher": recording.teacher,
            "meeting_topic": recording.meeting_topic,
            "drive_file_id": recording.drive_file_id,
        },
        "transcript": None,
        "classification": None,
        "naming": recording.name_plan.model_dump() if recording.name_plan else None,
        "placement": recording.placement_plan.model_dump() if recording.placement_plan else None,
    }

    if recording.transcript:
        payload["transcript"] = {
            "provider": recording.transcript.provider,
            "model": recording.transcript.model,
            "language": recording.transcript.language,
            "duration_s": recording.transcript.duration_s,
            "runtime_s": recording.transcript.runtime_s,
            "segment_count": len(recording.transcript.segments),
            "word_count": recording.transcript.word_count,
        }

    if classification:
        payload["classification"] = {
            "method": classification.method,
            "model": classification.model,
            "abstain": classification.abstain,
            "abstain_reason": classification.abstain_reason,
            "is_continuation": classification.is_continuation,
            "suggested_topic_label": classification.suggested_topic_label,
            "reasoning": classification.reasoning,
            "primary": classification.primary.model_dump() if classification.primary else None,
            "secondary": [s.model_dump() for s in classification.secondary],
        }

    if extra:
        payload["extra"] = extra

    return json.dumps(payload, indent=2, ensure_ascii=False)

