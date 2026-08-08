"""Filename construction.

The convention under test is the one already in the user's Drive:
``Aug, 5 - Atomic Structure.mp4``. These tests exist mainly to stop a future
change silently altering it, since a drift here makes the archive inconsistent
in a way nobody notices until they browse a folder.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from pipeline.config import NamingSettings
from pipeline.models import Classification, Evidence, TopicMatch
from pipeline.naming import build_name, sanitise


def _classification(node_id: str, label: str, confidence: float = 0.9) -> Classification:
    return Classification(
        primary=TopicMatch(
            node_id=node_id,
            confidence=confidence,
            coverage=0.8,
            evidence=[Evidence(quote="q", start_s=1.0)],
        ),
        suggested_topic_label=label,
    )


def test_reproduces_existing_drive_convention(recording_factory, taxonomy):
    """The exact format observed in the user's Drive."""
    recording = recording_factory("raw.mp4", recorded_at=datetime(2026, 8, 5, 14, 0))
    plan = build_name(
        recording,
        _classification("chem.atomic_structure", "Atomic Structure"),
        taxonomy,
        NamingSettings(),
    )
    assert plan.filename == "Aug, 5 - Atomic Structure.mp4"


def test_day_is_not_zero_padded(recording_factory, taxonomy):
    """"Aug, 5" not "Aug, 05" — matches the observed files."""
    recording = recording_factory("raw.mp4", recorded_at=datetime(2026, 8, 5, 14, 0))
    plan = build_name(
        recording, _classification("chem.hydrocarbons", "Alkanes"), taxonomy, NamingSettings()
    )
    assert plan.filename.startswith("Aug, 5 - ")


def test_specific_label_preferred_over_chapter_title(recording_factory, taxonomy):
    """A lecture on one subtopic is named for it, not for the whole chapter."""
    recording = recording_factory("raw.mp4", recorded_at=datetime(2026, 8, 10, 14, 0))
    plan = build_name(
        recording,
        _classification("chem.atomic_structure.rel_atomic_mass", "Relative Atomic Mass"),
        taxonomy,
        NamingSettings(),
    )
    assert plan.filename == "Aug, 10 - Relative Atomic Mass.mp4"


@pytest.mark.parametrize("bad", ['a/b', "a\\b", "a:b", "a*b", "a?b", 'a"b', "a<b", "a>b", "a|b"])
def test_illegal_characters_removed(bad):
    """These break Drive, Windows sync clients, and shell scripts alike."""
    assert sanitise(bad, NamingSettings().strip_characters) == "ab"


def test_trailing_dot_removed():
    """Windows silently drops a trailing dot, turning a rename into a surprise."""
    assert sanitise("Topic.", NamingSettings().strip_characters) == "Topic"


def test_low_confidence_marker_must_be_filesystem_safe():
    """A "?" marker would make every low-confidence rename fail.

    Regression test: "?" was the original default and is illegal on Windows
    and in Drive, so the guard belongs in config validation rather than in a
    comment.
    """
    with pytest.raises(ValidationError, match="illegal in"):
        NamingSettings(low_confidence_marker=" ?")


def test_low_confidence_marker_is_applied(recording_factory, taxonomy):
    recording = recording_factory("raw.mp4", recorded_at=datetime(2026, 8, 10, 14, 0))
    plan = build_name(
        recording,
        _classification("chem.atomic_structure", "Atomic Structure", confidence=0.4),
        taxonomy,
        NamingSettings(),
        is_low_confidence=True,
    )
    assert plan.filename == "Aug, 10 - Atomic Structure (review).mp4"
    assert plan.is_low_confidence


def test_same_day_same_topic_becomes_part_2(recording_factory, taxonomy):
    """Almost always one lecture split across two recordings, so "Part 2"
    reads correctly to a student; "(2)" does not."""
    recording = recording_factory("raw.mp4", recorded_at=datetime(2026, 8, 10, 14, 0))
    plan = build_name(
        recording,
        _classification("chem.atomic_structure", "Atomic Structure"),
        taxonomy,
        NamingSettings(),
        existing_names={"Aug, 10 - Atomic Structure.mp4"},
    )
    assert plan.filename == "Aug, 10 - Atomic Structure - Part 2.mp4"
    assert plan.part_number == 2


def test_collision_walks_past_existing_parts(recording_factory, taxonomy):
    recording = recording_factory("raw.mp4", recorded_at=datetime(2026, 8, 10, 14, 0))
    plan = build_name(
        recording,
        _classification("chem.atomic_structure", "Atomic Structure"),
        taxonomy,
        NamingSettings(),
        existing_names={
            "Aug, 10 - Atomic Structure.mp4",
            "Aug, 10 - Atomic Structure - Part 2.mp4",
        },
    )
    assert plan.filename == "Aug, 10 - Atomic Structure - Part 3.mp4"


def test_length_cap_truncates_topic_not_date(recording_factory, taxonomy):
    """The date is what a human scans for; losing it makes a listing useless."""
    recording = recording_factory("raw.mp4", recorded_at=datetime(2026, 8, 10, 14, 0))
    plan = build_name(
        recording,
        _classification("chem.atomic_structure", "X" * 300),
        taxonomy,
        NamingSettings(max_length=60),
    )
    assert len(plan.filename) <= 60
    assert plan.filename.startswith("Aug, 10 - ")
    assert plan.filename.endswith(".mp4")


def test_extension_preserved_from_source(recording_factory, taxonomy):
    recording = recording_factory("raw.mkv", recorded_at=datetime(2026, 8, 10, 14, 0))
    plan = build_name(
        recording, _classification("chem.alcohols", "Alcohols"), taxonomy, NamingSettings()
    )
    assert plan.filename.endswith(".mkv")


def test_unclassified_recording_still_gets_a_name(recording_factory, taxonomy):
    recording = recording_factory("raw.mp4", recorded_at=datetime(2026, 8, 10, 14, 0))
    plan = build_name(recording, None, taxonomy, NamingSettings())
    assert plan.filename == "Aug, 10 - Unclassified.mp4"


def test_alternative_template_with_more_identifiers(recording_factory, taxonomy):
    recording = recording_factory(
        "raw.mp4", recorded_at=datetime(2026, 8, 10, 14, 0), teacher="Ms Rahman"
    )
    plan = build_name(
        recording,
        _classification("chem.atomic_structure.rel_atomic_mass", "Relative Atomic Mass"),
        taxonomy,
        NamingSettings(template="{date} - {chapter} - {topic} [{teacher}]{ext}"),
    )
    assert plan.filename == (
        "2026-08-10 - Atomic Structure - Relative Atomic Mass [Ms Rahman].mp4"
    )
