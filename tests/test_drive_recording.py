"""Drive file -> Recording, the connection PipelineRunner.process() needs."""

from __future__ import annotations

from datetime import datetime

from pipeline.config import IngestSettings
from pipeline.ingest import is_recording_file, recorded_at_from_name, recording_from_drive_file
from pipeline.models import Source
from pipeline.storage import DriveFile


def _file(name: str, **kwargs) -> DriveFile:
    return DriveFile(
        id=kwargs.pop("id", "drive-file-1"),
        name=name,
        mime_type=kwargs.pop("mime_type", "video/mp4"),
        parents=kwargs.pop("parents", ["folder-1"]),
        **kwargs,
    )


def test_recording_from_drive_file_carries_drive_identity():
    created = datetime(2026, 9, 1, 12, 0, 0)
    file = _file(
        "Copy of Sep, 9 - A2 Chemistry M_J 27.mp4",
        id="abc123",
        created_time=created,
        md5_checksum="deadbeef",
    )
    recording = recording_from_drive_file(file, subject_id="a2_chemistry")

    assert recording.source == Source.DRIVE
    assert recording.id == "abc123"
    assert recording.source_id == "abc123"
    assert recording.drive_file_id == "abc123"
    assert recording.drive_parent_id == "folder-1"
    assert recording.original_filename == "Copy of Sep, 9 - A2 Chemistry M_J 27.mp4"
    assert recording.subject_id == "a2_chemistry"
    assert recording.recorded_at == datetime(2026, 9, 9)
    assert recording.content_hash == "deadbeef"
    assert recording.stage.value == "discovered"


def test_sidecar_files_are_not_recordings():
    ingest = IngestSettings()
    assert not is_recording_file(_file("recording.conf", mime_type="text/plain"), ingest)
    assert not is_recording_file(_file("chat.txt", mime_type="text/plain"), ingest)
    assert not is_recording_file(
        _file("meeting_saved_new_chat.txt", mime_type="text/plain"), ingest
    )
    assert not is_recording_file(
        _file("notes", mime_type="application/vnd.google-apps.folder"), ingest
    )
    assert is_recording_file(_file("Copy of Sep, 9 - A2 Chemistry M_J 27.mp4"), ingest)


def test_recorded_at_prefers_the_stem_date():
    fallback = datetime(2026, 1, 1)
    assert recorded_at_from_name(
        "Copy of Sep, 9 - A2 Chemistry M_J 27.mp4", fallback
    ) == datetime(2026, 9, 9)
