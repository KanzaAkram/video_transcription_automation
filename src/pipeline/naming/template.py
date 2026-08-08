"""Filename construction.

The default template reproduces the convention already in your Drive:

    "Aug, 5 - Atomic Structure.mp4"

That is deliberate. A pipeline that renames everything into some new scheme
would leave the existing archive inconsistent, so the default is your format
and the template is configuration rather than code.
"""

from __future__ import annotations

import re
from pathlib import Path

from pipeline.config import NamingSettings
from pipeline.models import Classification, NamePlan, Recording
from pipeline.syllabus import Taxonomy

MONTH_ABBR = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

_WHITESPACE = re.compile(r"\s+")


def sanitise(text: str, strip_characters: str) -> str:
    """Remove characters that break Drive, Windows sync clients and shells.

    Also collapses whitespace and trims trailing dots — Windows silently drops
    a trailing dot from a filename, which turns a rename into a surprise.
    """
    cleaned = "".join(ch for ch in text if ch not in strip_characters)
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()
    return cleaned.rstrip(". ")


def _topic_label(classification: Classification | None, taxonomy: Taxonomy | None) -> str:
    """What goes in the filename.

    Prefers the classifier's suggested label over the bare node title, because
    it is usually more specific: "Relative Atomic Mass" rather than the
    blunter "Atomic Structure".
    """
    if classification is None:
        return "Unclassified"
    if classification.suggested_topic_label:
        return classification.suggested_topic_label
    if classification.primary and taxonomy is not None:
        node = taxonomy.get(classification.primary.node_id)
        if node:
            return node.title
    return "Unclassified"


def _chapter_label(classification: Classification | None, taxonomy: Taxonomy | None) -> str:
    if not (classification and classification.primary and taxonomy):
        return ""
    node = taxonomy.get(classification.primary.node_id)
    if node is None:
        return ""
    # Walk up to the chapter-level ancestor if this is a subtopic.
    ancestors = taxonomy.ancestors(node.id)
    chapter = ancestors[0] if ancestors else node
    return chapter.title


def build_name(
    recording: Recording,
    classification: Classification | None,
    taxonomy: Taxonomy | None,
    settings: NamingSettings,
    *,
    existing_names: set[str] | None = None,
    is_low_confidence: bool = False,
) -> NamePlan:
    """Render the filename for a classified recording.

    Collision handling is by *meaning*, not by counter: same date plus same
    topic almost always means one lecture split across two recordings, so the
    result is "- Part 1" / "- Part 2" rather than "(1)" / "(2)". That reads
    correctly to a student browsing the folder.
    """
    existing_names = existing_names or set()
    extension = Path(recording.original_filename).suffix or ".mp4"
    when = recording.recorded_at

    topic = sanitise(_topic_label(classification, taxonomy), settings.strip_characters)
    chapter = sanitise(_chapter_label(classification, taxonomy), settings.strip_characters)
    teacher = sanitise(recording.teacher or "", settings.strip_characters)

    fields: dict[str, str] = {
        "date": when.strftime("%Y-%m-%d"),
        "year": str(when.year),
        "month_abbr": MONTH_ABBR[when.month - 1],
        "month": f"{when.month:02d}",
        "day": str(when.day),
        "day_padded": f"{when.day:02d}",
        "subject": sanitise(recording.subject_id, settings.strip_characters),
        "chapter": chapter,
        "subtopic": topic,
        "topic": topic,
        "teacher": teacher,
        "ext": extension,
        "part_suffix": "",
        "confidence_marker": settings.low_confidence_marker if is_low_confidence else "",
    }

    part_number: int | None = None

    def render(overrides: dict[str, str] | None = None) -> str:
        merged = {**fields, **(overrides or {})}
        name = settings.template.format(**merged)
        # The marker is appended before the extension so sorting stays sane.
        if is_low_confidence and settings.low_confidence_marker not in name:
            stem, _, ext = name.rpartition(extension)
            name = f"{stem.rstrip()}{settings.low_confidence_marker}{extension}"
        return _enforce_length(name, extension, settings.max_length)

    candidate = render()

    if candidate in existing_names:
        if settings.collision_strategy == "part_suffix":
            # The first of the pair is retroactively "Part 1" only if it is
            # already on disk under the bare name; we never rename other
            # people's files, so this one becomes Part 2 onward.
            part_number = 2
            while render({"part_suffix": f" - Part {part_number}"}) in existing_names:
                part_number += 1
            candidate = render({"part_suffix": f" - Part {part_number}"})
        elif settings.collision_strategy == "counter":
            n = 2
            while render({"part_suffix": f" ({n})"}) in existing_names:
                n += 1
            candidate = render({"part_suffix": f" ({n})"})
        else:  # timestamp
            candidate = render({"part_suffix": f" [{when.strftime('%H%M')}]"})

    return NamePlan(
        filename=candidate,
        original_filename=recording.original_filename,
        template_used=settings.template,
        is_low_confidence=is_low_confidence,
        part_number=part_number,
    )


def _enforce_length(name: str, extension: str, max_length: int) -> str:
    """Truncate the topic, never the date.

    The date is the part a human scans for; losing it to truncation makes a
    folder listing useless.
    """
    if len(name) <= max_length:
        return name
    stem = name[: -len(extension)] if name.endswith(extension) else name
    keep = max_length - len(extension) - 1
    return stem[:keep].rstrip(" -") + extension
