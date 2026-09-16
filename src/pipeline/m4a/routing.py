"""Parse class/subject from a recording name and pick the matching taxonomy.

The existing Chemistry pipeline is not involved. This map points at taxonomy
files that already live under config/syllabi/; it does not create or edit them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

CLASS_SUBJECT = re.compile(
    r"\b(IGCSE|IG|AS|A2)\s+(Chemistry|Physics|Mathematics|Biology|Maths)\b",
    re.IGNORECASE,
)
TIMESTAMP = re.compile(
    r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})"
    r"[ T](?P<hour>\d{2})[.:](?P<minute>\d{2})[.:](?P<second>\d{2})"
)

#: (class, subject) -> taxonomy YAML already in the repo.
TAXONOMY_FILES: dict[tuple[str, str], Path] = {
    ("A2", "Chemistry"): Path("config/syllabi/a2-chemistry-9701.yaml"),
    ("AS", "Chemistry"): Path("config/syllabi/as-chemistry-9701.yaml"),
    ("A2", "Biology"): Path("config/syllabi/a2-biology-9700.yaml"),
    ("IGCSE", "Chemistry"): Path("config/syllabi/igcse-chemistry-0620.yaml"),
    ("IGCSE", "Mathematics"): Path("config/syllabi/igcse-maths-0580.yaml"),
    ("IGCSE", "Maths"): Path("config/syllabi/igcse-maths-0580.yaml"),
}

SUBJECT_IDS: dict[tuple[str, str], str] = {
    ("A2", "Chemistry"): "a2_chemistry",
    ("AS", "Chemistry"): "as_chemistry",
    ("A2", "Biology"): "a2_biology",
    ("IGCSE", "Chemistry"): "igcse_chemistry",
    ("IGCSE", "Mathematics"): "igcse_maths",
    ("IGCSE", "Maths"): "igcse_maths",
}


@dataclass(frozen=True)
class SubjectHint:
    class_name: str
    subject: str
    subject_id: str
    taxonomy_path: Path
    recorded_at: datetime | None


def parse_recorded_at(name: str) -> datetime | None:
    match = TIMESTAMP.search(name)
    if match is None:
        return None
    return datetime(
        int(match["year"]),
        int(match["month"]),
        int(match["day"]),
        int(match["hour"]),
        int(match["minute"]),
        int(match["second"]),
    )


def detect_subject(name: str, *, repo_root: Path | None = None) -> SubjectHint:
    """Read class and subject from a filename like '... RA A2 Chemistry M_J 27'."""
    match = CLASS_SUBJECT.search(name)
    if match is None:
        raise ValueError(f"could not read class/subject from {name!r}")
    token = match.group(1).upper()
    lookup_class = "IGCSE" if token in {"IG", "IGCSE"} else token
    class_name = token
    subject = match.group(2).title()
    if subject == "Maths":
        subject = "Mathematics"
    key = (lookup_class, subject)
    taxonomy_path = TAXONOMY_FILES.get(key)
    subject_id = SUBJECT_IDS.get(key)
    if taxonomy_path is None or subject_id is None:
        raise FileNotFoundError(
            f"no existing taxonomy is mapped for {class_name} {subject}. "
            f"Known keys: {sorted(TAXONOMY_FILES)}"
        )
    root = repo_root or Path.cwd()
    resolved = (root / taxonomy_path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"taxonomy file for {class_name} {subject} is not in the repo: {resolved}"
        )
    return SubjectHint(
        class_name=class_name,
        subject=subject,
        subject_id=subject_id,
        taxonomy_path=resolved,
        recorded_at=parse_recorded_at(name),
    )
