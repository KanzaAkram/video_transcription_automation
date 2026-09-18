"""Unit tests for the m4a preview pipeline. No Drive, no OpenAI, no Whisper."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.config import NamingSettings
from pipeline.m4a.discover import find_existing_m4a, find_existing_video, find_named_item
from pipeline.m4a.labels import classification_fields, generated_filename
from pipeline.m4a.preview import format_preview
from pipeline.m4a.routing import detect_subject
from pipeline.m4a.transcribe import files_to_transcribe, format_chunk_plan, transcribe_m4a
from pipeline.models import (
    Classification,
    Recording,
    Source,
    TopicMatch,
    Transcript,
    TranscriptSegment,
)
from pipeline.storage.local_drive import LocalDriveAdapter
from pipeline.syllabus import load_taxonomy

REPO_ROOT = Path(__file__).resolve().parents[1]
STEM = "2026-09-14 19.08.53 RA A2 Chemistry M_J 27"


def test_a2_chemistry_routes_to_existing_taxonomy():
    hint = detect_subject(STEM, repo_root=REPO_ROOT)
    assert hint.class_name == "A2"
    assert hint.subject == "Chemistry"
    assert hint.subject_id == "a2_chemistry"
    assert hint.taxonomy_path == (REPO_ROOT / "config/syllabi/a2-chemistry-9701.yaml").resolve()
    assert hint.recorded_at == datetime(2026, 9, 14, 19, 8, 53)
    load_taxonomy(hint.taxonomy_path)


def test_as_chemistry_routes_to_existing_taxonomy():
    hint = detect_subject(
        "2026-09-13 20.14.59 RA AS Chemistry M_J 27", repo_root=REPO_ROOT
    )
    assert hint.class_name == "AS"
    assert hint.subject == "Chemistry"
    assert hint.subject_id == "as_chemistry"
    assert hint.taxonomy_path == (REPO_ROOT / "config/syllabi/as-chemistry-9701.yaml").resolve()
    load_taxonomy(hint.taxonomy_path)


def test_ig_chemistry_routes_to_igcse_taxonomy():
    hint = detect_subject(
        "2026-09-13 21.14.07 RA IG Chemistry M_J 27", repo_root=REPO_ROOT
    )
    assert hint.class_name == "IG"
    assert hint.subject == "Chemistry"
    assert hint.subject_id == "igcse_chemistry"
    assert hint.taxonomy_path == (REPO_ROOT / "config/syllabi/igcse-chemistry-0620.yaml").resolve()
    load_taxonomy(hint.taxonomy_path)


def test_ig_mathematics_routes_to_igcse_maths_taxonomy():
    hint = detect_subject("2026-09-14 21.15.13 RA IG Mathematics", repo_root=REPO_ROOT)
    assert hint.class_name == "IG"
    assert hint.subject == "Mathematics"
    assert hint.subject_id == "igcse_maths"
    assert hint.taxonomy_path == (REPO_ROOT / "config/syllabi/igcse-maths-0580.yaml").resolve()
    load_taxonomy(hint.taxonomy_path)


def test_physics_without_taxonomy_file_fails_loudly():
    with pytest.raises(FileNotFoundError, match="no existing taxonomy"):
        detect_subject("2026-09-14 20.06.25 RA A2 Physics M_J 27", repo_root=REPO_ROOT)


def test_finds_m4a_inside_named_folder(tmp_path: Path):
    folder = tmp_path / STEM
    folder.mkdir()
    (folder / f"{STEM}.m4a").write_bytes(b"audio")
    (folder / f"{STEM}.mp4").write_bytes(b"video")
    drive = LocalDriveAdapter(tmp_path)
    named, audio = find_existing_m4a(drive, "root", STEM)
    assert named.is_folder
    assert named.name == STEM
    assert audio.name.endswith(".m4a")
    assert audio.name == f"{STEM}.m4a"


def test_finds_generic_m4a_when_source_is_the_recording_folder(tmp_path: Path):
    root = tmp_path / STEM
    root.mkdir()
    (root / "audio1205897237.m4a").write_bytes(b"audio")
    (root / "chat.txt").write_text("chat", encoding="utf-8")
    (root / "recording.conf").write_text("conf", encoding="utf-8")
    drive = LocalDriveAdapter(root)
    named, audio = find_existing_m4a(drive, "root", STEM)
    assert named.name == STEM
    assert audio.name == "audio1205897237.m4a"


def test_finds_m4a_when_the_item_is_the_audio_file(tmp_path: Path):
    (tmp_path / f"{STEM}.m4a").write_bytes(b"audio")
    drive = LocalDriveAdapter(tmp_path)
    named, audio = find_existing_m4a(drive, "root", STEM)
    assert named.id == audio.id
    assert audio.name.endswith(".m4a")


def test_finds_mp4_and_ignores_chat(tmp_path: Path):
    (tmp_path / f"{STEM}.mp4").write_bytes(b"video")
    (tmp_path / "chat.txt").write_text("chat", encoding="utf-8")
    (tmp_path / "recording.conf").write_text("conf", encoding="utf-8")
    (tmp_path / "notes.pdf").write_bytes(b"pdf")
    drive = LocalDriveAdapter(tmp_path)
    video = find_existing_video(drive, "root", STEM)
    assert video.name == f"{STEM}.mp4"


def test_ignores_unrelated_children(tmp_path: Path):
    (tmp_path / "other lecture.m4a").write_bytes(b"no")
    (tmp_path / f"{STEM}.m4a").write_bytes(b"yes")
    drive = LocalDriveAdapter(tmp_path)
    children = drive.list_children("root")
    found = find_named_item(children, STEM)
    assert found.name == f"{STEM}.m4a"


def test_classification_json_and_serial_date_topic_name():
    taxonomy = load_taxonomy(REPO_ROOT / "config/syllabi/a2-chemistry-9701.yaml")
    hint = detect_subject(STEM, repo_root=REPO_ROOT)
    chapter = next(n for n in taxonomy.nodes if n.children and not n.exclude_from_classification)
    child = chapter.children[0]
    classification = Classification(
        primary=TopicMatch(node_id=child.id, confidence=0.9, coverage=0.8),
        chapter_node_id=chapter.id,
        topic_node_ids=[child.id],
        suggested_topic_label=child.label,
        method="openai",
    )
    fields = classification_fields(classification, taxonomy, hint)
    assert fields["class"] == "A2"
    assert fields["subject"] == "Chemistry"
    assert fields["chapter"]
    assert fields["topic"]
    recording = Recording(
        id="x",
        source=Source.DRIVE,
        source_id="x",
        subject_id="a2_chemistry",
        original_filename=f"{STEM}.m4a",
        recorded_at=datetime(2026, 9, 14, 19, 8, 53),
    )
    naming = NamingSettings(template="{serial} - {month_abbr} {day} - {topic}{ext}")
    name = generated_filename(recording, classification, taxonomy, naming, serial=1)
    assert name.startswith("1 - Sep 14 - ")
    assert name.endswith(".m4a")


def test_transcribe_uses_openai_audio_client(tmp_path: Path):
    audio = tmp_path / "clip.m4a"
    audio.write_bytes(b"fake")
    result = SimpleNamespace(
        text="lattice energy",
        duration=12.5,
        language="en",
        segments=[SimpleNamespace(start=0.0, end=12.5, text=" lattice energy ")],
    )
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        return result

    client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
    transcript = transcribe_m4a(audio, api_key="sk-test", model="whisper-1", client=client)
    assert calls and calls[0]["model"] == "whisper-1"
    assert calls[0]["response_format"] == "verbose_json"
    assert transcript.provider == "openai_audio"
    assert transcript.duration_s == 12.5
    assert "lattice energy" in transcript.text


def test_oversized_audio_is_sent_as_separate_openai_calls(tmp_path: Path):
    big = tmp_path / "lecture.m4a"
    big.write_bytes(b"x")
    first = tmp_path / "part0.wav"
    second = tmp_path / "part1.wav"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    calls: list[str] = []

    def create(**kwargs):
        name = Path(kwargs["file"].name).name
        calls.append(name)
        start = 0.0 if name == "part0.wav" else 0.0
        return SimpleNamespace(
            text="one" if name == "part0.wav" else "two",
            duration=10.0,
            language="en",
            segments=[SimpleNamespace(start=start, end=10.0, text=" one" if name == "part0.wav" else " two")],
        )

    client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
    transcript = transcribe_m4a(
        big,
        api_key="sk-test",
        client=client,
        split_files=[(first, 0.0), (second, 10.0)],
    )
    assert calls == ["part0.wav", "part1.wav"]
    assert transcript.duration_s == 20.0
    assert transcript.segments[1].start_s == 10.0
    assert "one" in transcript.text and "two" in transcript.text


def test_preview_text_includes_required_fields():
    text = format_preview(
        video_name=f"{STEM}.mp4",
        audio_extraction_s=8.4,
        asr_model="faster-whisper base (cpu, int8, VAD on)",
        transcription_s=12.3 * 60,
        transcript_status="ok (1200 chars)",
        fields={
            "class": "A2",
            "subject": "Chemistry",
            "chapter": "15. Analytical Techniques",
            "topic": "Chromatography",
            "subtopic": "Thin-Layer Chromatography",
        },
        taxonomy_path="config/syllabi/a2-chemistry-9701.yaml",
        filename="1 - Sep 14 - Thin-Layer Chromatography.mp4",
    )
    assert "VIDEO:" in text
    assert "AUDIO EXTRACTION TIME:" in text
    assert "ASR MODEL:" in text
    assert "TRANSCRIPTION TIME:" in text
    assert "TRANSCRIPT STATUS:" in text
    assert "CLASS:" in text
    assert "SUBJECT:" in text
    assert "TAXONOMY:" in text
    assert "CHAPTER:" in text
    assert "TOPIC:" in text
    assert "SUBTOPIC:" in text
    assert "GENERATED FILENAME:" in text
    assert "whisper-1" not in text.lower()
    assert "gpt-audio" not in text.lower()


def test_files_under_22mb_are_not_split(tmp_path: Path):
    audio = tmp_path / "lecture.m4a"
    audio.write_bytes(b"x" * (10 * 1024 * 1024))
    pieces = files_to_transcribe(audio)
    assert pieces == [(audio, 0.0)]
    plan = format_chunk_plan(audio, pieces)
    assert "NUMBER OF CHUNKS: 1" in plan
    assert "ORIGINAL AUDIO SIZE:" in plan
    assert "TARGET CHUNK SIZE: 21 MB" in plan
