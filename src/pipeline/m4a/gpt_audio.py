"""One OpenAI audio model call: listen to the lecture and place it in the syllabus.

Isolated from faster-whisper, whisper-1, and the text classifier. Drive is not written here.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from pipeline.config import NamingSettings
from pipeline.models import Classification, Recording, Source, TopicMatch
from pipeline.syllabus import SyllabusNode, Taxonomy

from pipeline.m4a.labels import generated_filename
from pipeline.m4a.routing import SubjectHint

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-audio"

_SPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s+&Δδ]", re.UNICODE)


def _norm(text: str) -> str:
    cleaned = text.replace("Δ", "delta").replace("δ", "delta").replace(",", " ")
    cleaned = _PUNCT.sub(" ", cleaned.lower())
    return _SPACE.sub(" ", cleaned).strip()


def taxonomy_outline(taxonomy: Taxonomy) -> str:
    """Compact syllabus tree the audio model must copy names from."""
    lines = [
        f"Syllabus: {taxonomy.subject_name}"
        + (f" ({taxonomy.syllabus_code})" if taxonomy.syllabus_code else ""),
        "Copy chapter/topic/subtopic names from this list. Do not invent names.",
        "",
    ]
    for chapter in taxonomy.nodes:
        if chapter.exclude_from_classification:
            continue
        ch_name = chapter.drive_folder_name or chapter.title
        lines.append(f"CHAPTER: {ch_name} | {chapter.label}")
        for topic in chapter.children:
            if topic.exclude_from_classification:
                continue
            topic_code = f"{topic.syllabus_topic} " if topic.syllabus_topic else ""
            lines.append(f"  TOPIC: {topic_code}{topic.title} | {topic.label}")
            for leaf in topic.children:
                if leaf.exclude_from_classification:
                    continue
                lines.append(f"    SUBTOPIC: {leaf.label} | {leaf.title}")
    return "\n".join(lines)


def _node_strings(node: SyllabusNode) -> list[str]:
    parts = [node.title, node.label, node.drive_folder_name or ""]
    if node.syllabus_topic:
        parts.append(f"{node.syllabus_topic} {node.title}")
        parts.append(f"{node.syllabus_topic} {node.label}")
    parts.extend(node.aliases)
    return [p for p in parts if p]


def match_node(text: str, nodes: list[SyllabusNode]) -> SyllabusNode | None:
    """Map a model string onto an existing node, or None if it is not in the syllabus."""
    needle = _norm(text)
    if not needle or not nodes:
        return None
    exact: list[tuple[int, SyllabusNode]] = []
    partial: list[tuple[int, SyllabusNode]] = []
    for node in nodes:
        for label in _node_strings(node):
            hay = _norm(label)
            if not hay:
                continue
            if hay == needle:
                exact.append((len(hay), node))
            elif hay in needle or needle in hay:
                partial.append((len(hay), node))
    if exact:
        exact.sort(key=lambda item: -item[0])
        return exact[0][1]
    if partial:
        partial.sort(key=lambda item: -item[0])
        return partial[0][1]
    return None


def validate_placement(
    raw: dict[str, str],
    taxonomy: Taxonomy,
    hint: SubjectHint,
) -> dict[str, str]:
    """Keep class/subject from the filename; keep chapter/topic/subtopic only if they exist."""
    chapters = [n for n in taxonomy.nodes if not n.exclude_from_classification]
    chapter = match_node(raw.get("chapter", ""), chapters)
    topic = None
    subtopic = None
    if chapter is not None:
        pool = list(chapter.children)
        topic = match_node(raw.get("topic", ""), pool)
        if topic is None:
            topic = match_node(raw.get("subtopic", ""), pool)
        if topic is not None and topic.children:
            subtopic = match_node(raw.get("subtopic", ""), topic.children)
        elif topic is not None:
            if match_node(raw.get("subtopic", ""), [topic]) is topic:
                subtopic = topic
    return {
        "class": hint.class_name,
        "subject": hint.subject,
        "chapter": (chapter.drive_folder_name or chapter.label) if chapter else "",
        "topic": topic.label if topic else "",
        "subtopic": subtopic.label if subtopic and subtopic is not topic else (subtopic.label if subtopic else ""),
        "_chapter_id": chapter.id if chapter else "",
        "_topic_id": topic.id if topic else "",
        "_subtopic_id": subtopic.id if subtopic and subtopic is not topic else "",
    }


def classification_from_validated(
    fields: dict[str, str],
    *,
    model: str,
) -> Classification:
    topic_id = fields.get("_topic_id") or fields.get("_chapter_id") or ""
    ids = [node_id for node_id in (fields.get("_topic_id"), fields.get("_subtopic_id")) if node_id]
    if not ids and fields.get("_chapter_id"):
        ids = [fields["_chapter_id"]]
    label = fields.get("subtopic") or fields.get("topic") or "Unclassified"
    return Classification(
        primary=TopicMatch(node_id=topic_id, confidence=0.9, coverage=1.0) if topic_id else None,
        chapter_node_id=fields.get("_chapter_id") or None,
        topic_node_ids=ids[:3],
        suggested_topic_label=label,
        method="gpt-audio",
        model=model,
    )


def encode_mp3(src: Path, dest: Path) -> Path:
    """gpt-audio accepts wav or mp3, not m4a."""
    try:
        import av
        from av.audio.resampler import AudioResampler
    except ImportError as exc:
        raise RuntimeError("PyAV is required to send .m4a audio to gpt-audio") from exc

    dest.parent.mkdir(parents=True, exist_ok=True)
    incoming = av.open(str(src))
    outgoing = av.open(str(dest), mode="w", format="mp3")
    ostream = outgoing.add_stream("libmp3lame", rate=16000)
    ostream.bit_rate = 48000
    resampler = AudioResampler(format="s16", layout="mono", rate=16000)

    def as_frames(value: Any) -> list[Any]:
        if value is None:
            return []
        return value if isinstance(value, list) else [value]

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
    finally:
        outgoing.close()
        incoming.close()
    return dest


def _audio_content_parts(paths: list[Path]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for path in paths:
        payload = base64.standard_b64encode(path.read_bytes()).decode("ascii")
        fmt = "mp3" if path.suffix.lower() == ".mp3" else "wav"
        parts.append(
            {
                "type": "input_audio",
                "input_audio": {"data": payload, "format": fmt},
            }
        )
    return parts


def _build_prompt(hint: SubjectHint, outline: str, n_parts: int) -> str:
    split_note = (
        f"The lecture audio is attached as {n_parts} sequential parts of the same recording. "
        if n_parts > 1
        else "The lecture audio is attached. "
    )
    return (
        f"{split_note}"
        f"Class and subject from the filename: {hint.class_name} {hint.subject}.\n\n"
        "Listen to the lecture and file it in the syllabus below. "
        "Return one chapter, one topic, and one subtopic that this lecture actually taught. "
        "Copy those names from the syllabus. Do not invent a chapter or topic that is not listed.\n\n"
        f"{outline}\n"
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return json.loads(stripped)


def classify_lecture_audio(
    audio_paths: list[Path],
    *,
    taxonomy: Taxonomy,
    hint: SubjectHint,
    api_key: str | None,
    model: str = DEFAULT_MODEL,
    client: Any | None = None,
) -> dict[str, str]:
    """One Chat Completions call with audio in. No transcription API, no second classifier."""
    if client is None:
        from openai import OpenAI

        options: dict[str, Any] = {"timeout": 600.0}
        if api_key:
            options["api_key"] = api_key
        client = OpenAI(**options)

    outline = taxonomy_outline(taxonomy)
    content: list[dict[str, Any]] = [
        {"type": "text", "text": _build_prompt(hint, outline, len(audio_paths))},
        *_audio_content_parts(audio_paths),
    ]
    messages = [
        {
            "role": "system",
            "content": (
                "You file one recorded lecture into a syllabus. "
                "Reply with JSON only. chapter, topic and subtopic must be names from the syllabus."
            ),
        },
        {"role": "user", "content": content},
    ]
    response = client.chat.completions.create(
        model=model,
        modalities=["text"],
        messages=messages,
    )
    message = response.choices[0].message
    raw_text = message.content or "{}"
    raw = _parse_json_object(raw_text)
    fields = validate_placement(
        {
            "class": str(raw.get("class", "")),
            "subject": str(raw.get("subject", "")),
            "chapter": str(raw.get("chapter", "")),
            "topic": str(raw.get("topic", "")),
            "subtopic": str(raw.get("subtopic", "")),
        },
        taxonomy,
        hint,
    )
    return fields


def recording_for_audio(
    audio_id: str,
    hint: SubjectHint,
    display_name: str,
    *,
    parent_id: str | None,
) -> Recording:
    when = hint.recorded_at or datetime.now()
    naive = when.replace(tzinfo=None) if when.tzinfo is not None else when
    return Recording(
        id=audio_id,
        source=Source.DRIVE,
        source_id=audio_id,
        subject_id=hint.subject_id,
        original_filename=display_name,
        recorded_at=naive,
        drive_file_id=audio_id,
        drive_parent_id=parent_id,
        meeting_topic=f"{hint.class_name} {hint.subject}",
    )


def generated_video_name(
    fields: dict[str, str],
    hint: SubjectHint,
    taxonomy: Taxonomy,
    naming: NamingSettings,
    *,
    display_name: str,
    audio_id: str,
    parent_id: str | None,
    existing_names: set[str],
) -> str:
    recording = recording_for_audio(audio_id, hint, display_name, parent_id=parent_id)
    classification = classification_from_validated(fields, model=DEFAULT_MODEL)
    return generated_filename(
        recording,
        classification,
        taxonomy,
        naming,
        serial=1,
        existing_names=existing_names,
    )
