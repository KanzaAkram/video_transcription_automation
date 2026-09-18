"""Turn a Google Drive file into a `Recording` the pipeline can process.

This is the missing connection between `GoogleDriveAdapter` and
`PipelineRunner.process()`. Everything downstream already speaks `Recording`;
nothing previously constructed one from a Drive listing.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from pipeline.config import IngestSettings
from pipeline.models import Recording, Source
from pipeline.storage import DriveFile

#: Zoom sidecar files that sit next to a recording. Never selected, never
#: renamed. Matching is by exact name, case-insensitive.
NON_RECORDING_NAMES = {
    "recording.conf",
    "chat.txt",
    "meeting_saved_new_chat.txt",
}

_MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1
    )
}
#: "class 3 (6 aug).mp4", "Lecture 2 - 14 Sept", "Sep, 9 - A2 Chemistry"
_DATE_IN_NAME = re.compile(
    r"(?:(?P<day>\d{1,2})\s*(?P<mon>jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)"
    r"|(?P<mon2>jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\w*\s*,?\s*(?P<day2>\d{1,2}))",
    re.IGNORECASE,
)
_ISO_IN_NAME = re.compile(r"(?P<y>20\d{2})[-_](?P<m>\d{2})[-_](?P<d>\d{2})")


def recorded_at_from_name(name: str, fallback: datetime) -> datetime:
    """Date for the filename, preferring the name over Drive timestamps.

    A download rewrites mtime to now. Drive `createdTime` is better than that,
    but a "Sep, 9" stem is what the naming template should reproduce.
    """
    stem = Path(name).stem
    iso = _ISO_IN_NAME.search(stem)
    if iso:
        return datetime(int(iso["y"]), int(iso["m"]), int(iso["d"]))
    match = _DATE_IN_NAME.search(stem)
    if match:
        day = match["day"] or match["day2"]
        mon = (match["mon"] or match["mon2"]).lower()[:3]
        return datetime(fallback.year, _MONTHS[mon], int(day))
    return fallback


def is_recording_file(file: DriveFile, ingest: IngestSettings | None = None) -> bool:
    """Whether this Drive child is a lecture recording we may process."""
    if file.is_folder:
        return False
    if file.name.lower() in NON_RECORDING_NAMES:
        return False
    ingest = ingest or IngestSettings()
    allowed = {ext.lower() for ext in (*ingest.video_extensions, *ingest.audio_extensions)}
    return file.extension in allowed


def recording_from_drive_file(
    file: DriveFile,
    *,
    subject_id: str,
) -> Recording:
    """Build the `Recording` `PipelineRunner.process()` expects.

    Identity is the Drive file id, so a second run of the same file is the
    same job even if the name has already changed.
    """
    fallback = file.created_time or file.modified_time or datetime.now()
    if fallback.tzinfo is not None:
        fallback = fallback.replace(tzinfo=None)
    recorded_at = recorded_at_from_name(file.name, fallback)
    parent_id = file.parents[0] if file.parents else None
    return Recording(
        id=file.id,
        source=Source.DRIVE,
        source_id=file.id,
        subject_id=subject_id,
        original_filename=file.name,
        recorded_at=recorded_at,
        drive_file_id=file.id,
        drive_parent_id=parent_id,
        content_hash=file.md5_checksum,
    )
