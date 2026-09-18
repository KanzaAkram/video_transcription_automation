"""Preview one Drive video through the production path. Never writes Drive."""

from __future__ import annotations

import socket
import time
from datetime import datetime
from pathlib import Path

socket.setdefaulttimeout(300)

from pipeline.asr import get_provider
from pipeline.classify.openai_classifier import OpenAIClassifier
from pipeline.config import ClassifySettings, NamingSettings, Secrets, load_settings
from pipeline.media import extract_audio
from pipeline.models import Recording, Source, TranscriptionHints
from pipeline.storage.base import DriveFile, DrivePort
from pipeline.syllabus import Taxonomy, load_taxonomy

from pipeline.m4a.discover import find_existing_video
from pipeline.m4a.labels import classification_fields, generated_filename
from pipeline.m4a.routing import SubjectHint, detect_subject

DEFAULT_FOLDER_ID = "1KdYxE8YXDaxifYyoGsTNIqsoLaOmFtzF"
DEFAULT_STEM = "2026-09-14 19.08.53 RA A2 Chemistry M_J 27"
ASR_MODEL = "faster-whisper base (cpu, int8, VAD on)"


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    return f"{seconds / 60:.1f} min"


def _recording(
    video: DriveFile,
    hint: SubjectHint,
    *,
    duration_s: float | None = None,
) -> Recording:
    when = hint.recorded_at or video.created_time or datetime.now()
    naive = when.replace(tzinfo=None) if when.tzinfo is not None else when
    return Recording(
        id=video.id,
        source=Source.DRIVE,
        source_id=video.id,
        subject_id=hint.subject_id,
        original_filename=video.name if video.name.lower().endswith(".mp4") else f"{video.name}.mp4",
        recorded_at=naive,
        duration_s=duration_s,
        drive_file_id=video.id,
        drive_parent_id=video.parents[0] if video.parents else None,
        meeting_topic=f"{hint.class_name} {hint.subject}",
    )


def format_preview(
    *,
    video_name: str,
    audio_extraction_s: float,
    asr_model: str,
    transcription_s: float,
    transcript_status: str,
    fields: dict[str, str],
    taxonomy_path: str,
    filename: str,
) -> str:
    return "\n".join(
        [
            f"VIDEO: {video_name}",
            f"AUDIO EXTRACTION TIME: {_fmt_duration(audio_extraction_s)}",
            "",
            f"ASR MODEL: {asr_model}",
            f"TRANSCRIPTION TIME: {_fmt_duration(transcription_s)}",
            f"TRANSCRIPT STATUS: {transcript_status}",
            "",
            f"CLASS: {fields['class']}",
            f"SUBJECT: {fields['subject']}",
            f"TAXONOMY: {taxonomy_path}",
            "",
            f"CHAPTER: {fields['chapter']}",
            f"TOPIC: {fields['topic']}",
            f"SUBTOPIC: {fields['subtopic']}",
            "",
            f"GENERATED FILENAME: {filename}",
        ]
    )


def preview_one(
    drive: DrivePort,
    folder_id: str,
    stem: str,
    *,
    secrets: Secrets,
    repo_root: Path,
    cache_dir: Path,
    write: bool = False,
) -> str:
    """Transcribe with faster-whisper base and classify with OpenAI.

    `write=False` (preview) never renames. `write=True` renames the original
    .mp4 after a validated classification.
    """
    video = find_existing_video(drive, folder_id, stem)
    hint = detect_subject(video.name, repo_root=repo_root)
    taxonomy: Taxonomy = load_taxonomy(hint.taxonomy_path)
    settings = load_settings(secrets.settings_file if secrets.settings_file.exists() else None)
    settings.asr.provider = "faster_whisper"
    settings.asr.model = "base"
    settings.asr.device = "cpu"
    settings.asr.compute_type = "int8"
    settings.asr.vad_filter = True
    settings.classify.provider = "openai"
    naming: NamingSettings = settings.naming
    classify_settings: ClassifySettings = settings.classify

    api_key = secrets.openai_api_key
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; the production classifier is OpenAI only")

    asr = get_provider(settings.asr)
    ok, reason = asr.is_available()
    if not ok:
        raise RuntimeError(f"faster-whisper is unavailable: {reason}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    for cached in cache_dir.glob("*.transcript.json"):
        cached.unlink()
    local = cache_dir / f"{video.id}.mp4"
    expected = video.size_bytes
    if local.exists() and expected and local.stat().st_size == expected:
        pass
    else:
        last_error: Exception | None = None
        for attempt in range(1, 6):
            try:
                if local.exists():
                    local.unlink()
                drive.download(video.id, local)
                if expected and local.stat().st_size != expected:
                    raise RuntimeError(
                        f"download size mismatch {local.stat().st_size} != {expected}"
                    )
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001 - Drive downloads time out on large files
                last_error = exc
                time.sleep(5 * attempt)
        if last_error is not None:
            raise RuntimeError(f"download failed: {last_error}") from last_error

    audio = cache_dir / f"{video.id}.wav"
    extract_started = time.monotonic()
    audio = extract_audio(local, audio)
    audio_extraction_s = time.monotonic() - extract_started

    language = "en"
    hints = TranscriptionHints(
        language=language,
        initial_prompt=taxonomy.initial_prompt() if settings.asr.prime_from_syllabus else None,
        hotwords=(
            taxonomy.vocabulary(settings.asr.max_hotwords)
            if settings.asr.prime_from_syllabus
            else []
        ),
    )
    transcribe_started = time.monotonic()
    transcript = asr.transcribe(audio, hints)
    transcription_s = transcript.runtime_s if transcript.runtime_s is not None else (
        time.monotonic() - transcribe_started
    )
    if not settings.runtime.keep_audio_after_transcription:
        audio.unlink(missing_ok=True)

    chars = len(transcript.text.strip())
    transcript_status = f"ok ({chars} chars)" if chars else "empty"
    if not chars:
        raise RuntimeError("transcription produced no text")

    recording = _recording(video, hint, duration_s=transcript.duration_s)
    classifier = OpenAIClassifier(
        taxonomy,
        classify_settings,
        naming,
        api_key=api_key,
    )
    classification = classifier.classify(recording, transcript)
    if classification.primary is None:
        raise RuntimeError(
            "OpenAI classifier produced no syllabus node: "
            f"{classification.abstain_reason or 'abstained'}"
        )
    fields = classification_fields(classification, taxonomy, hint)
    sibling_names = {child.name for child in drive.list_children(folder_id) if child.id != video.id}
    filename = generated_filename(
        recording,
        classification,
        taxonomy,
        naming,
        serial=1,
        existing_names=sibling_names,
    )
    if write:
        current = drive.get_file(video.id)
        if current.name != filename:
            drive.rename(video.id, filename)
    try:
        taxonomy_path = str(hint.taxonomy_path.relative_to(repo_root))
    except ValueError:
        taxonomy_path = str(hint.taxonomy_path)
    return format_preview(
        video_name=video.name,
        audio_extraction_s=audio_extraction_s,
        asr_model=ASR_MODEL,
        transcription_s=transcription_s,
        transcript_status=transcript_status,
        fields=fields,
        taxonomy_path=taxonomy_path.replace("\\", "/"),
        filename=filename,
    )
