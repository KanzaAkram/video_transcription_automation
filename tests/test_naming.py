"""Filename construction.

The convention under test is the one already in the user's Drive:
``Aug, 5 - Atomic Structure.mp4``. These tests exist mainly to stop a future
change silently altering it, since a drift here makes the archive inconsistent
in a way nobody notices until they browse a folder.

The subtopic rules — one name, or two joined, never three — live in
test_subtopic_naming.py. What is here is the surrounding format and the
compatibility guarantee: a Classification carrying none of the new fields
still names a file exactly as it did before they existed.
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
        _classification("chem.2_1", "Relative Atomic Mass"),
        taxonomy,
        NamingSettings(),
    )
    assert plan.filename == "Aug, 10 - Relative Atomic Mass.mp4"


def test_topic_node_ids_take_precedence_over_the_suggested_label(recording_factory, taxonomy):
    """The node ids are the classifier's committed answer; the label is prose.

    When both are present the ids win, because they are what placement and the
    JSON plan were built from — a filename disagreeing with them would be a
    filename disagreeing with the folder the file went into.
    """
    node = taxonomy.require("chem.2_1")
    recording = recording_factory("raw.mp4", recorded_at=datetime(2026, 8, 5, 14, 0))
    classification = _classification("chem.atomic_structure", "Something Else")
    classification.topic_node_ids = [node.id]
    plan = build_name(recording, classification, taxonomy, NamingSettings())
    assert plan.filename == f"Aug, 5 - {node.label}.mp4"


def test_unknown_topic_node_ids_fall_back_to_the_label(recording_factory, taxonomy):
    """A hallucinated id must not produce an empty name."""
    recording = recording_factory("raw.mp4", recorded_at=datetime(2026, 8, 5, 14, 0))
    classification = _classification("chem.atomic_structure", "Atomic Structure")
    classification.topic_node_ids = ["chem.not_a_real_node"]
    plan = build_name(recording, classification, taxonomy, NamingSettings())
    assert plan.filename == "Aug, 5 - Atomic Structure.mp4"


def test_default_length_cap_rejects_a_syllabus_sentence(recording_factory, taxonomy):
    """80, not 120. A cap generous enough to admit a Cambridge sentence is a
    cap that lets one through."""
    assert NamingSettings().max_length == 80
    recording = recording_factory("raw.mp4", recorded_at=datetime(2026, 8, 5, 14, 0))
    verbose = "Intermolecular forces, electronegativity and bond properties in simple molecules"
    plan = build_name(
        recording, _classification("chem.chemical_bonding", verbose), taxonomy, NamingSettings()
    )
    assert len(plan.filename) <= 80
    assert plan.filename.startswith("Aug, 5 - ")


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


# ── The cap is spent on the topic, never on the protected tail ────────────────


def _long_pair_classification() -> Classification:
    """Two real chemistry subtopics whose joined short titles nearly fill 80.

    "Chlorine & Disproportionation" + "Halogens & Hydrogen Halides" is the pair
    the adversarial review reproduced the hang with, and eight of the fifty-
    seven sibling pairs in this syllabus are this long or longer. A name that
    fits without a suffix but not with one is exactly the case where chopping
    the finished string destroys the suffix.
    """
    return Classification(
        primary=TopicMatch(node_id="chem.11_4", confidence=0.9, coverage=0.7),
        topic_node_ids=["chem.11_4", "chem.11_2"],
    )


def test_three_same_date_recordings_terminate_with_distinct_names(recording_factory, taxonomy):
    """Regression: this used to hang forever, halfway through a folder.

    The part suffix lives in the tail, and the cap used to be enforced by
    chopping the finished string — so once the topic was long enough,
    " - Part 2" and " - Part 3" truncated back to the same name, no part
    number was ever free, and the rename CLI spun at 100% CPU with half the
    folder renamed. Three same-date recordings in one chapter is precisely the
    Zoom-split case the Part-N strategy exists for.
    """
    taken: set[str] = set()
    names: list[str] = []
    for _ in range(3):
        recording = recording_factory("raw.mp4", recorded_at=datetime(2026, 9, 23, 14, 0))
        plan = build_name(
            recording,
            _long_pair_classification(),
            taxonomy,
            NamingSettings(),
            existing_names=taken,
        )
        names.append(plan.filename)
        taken.add(plan.filename)

    assert len(set(names)) == 3
    assert names[1].endswith(" - Part 2.mp4")
    assert names[2].endswith(" - Part 3.mp4")
    assert all(name.startswith("Sep, 23 - ") for name in names)
    assert all(len(name) <= NamingSettings().max_length for name in names)


def test_a_long_low_confidence_name_keeps_its_review_marker(recording_factory, taxonomy):
    """Regression: the cap used to eat the marker off the end of the name.

    A recording that failed the review gate then read exactly like one that
    passed it, which is the single distinction the marker exists to draw and
    the only one visible in a folder listing.
    """
    recording = recording_factory("raw.mp4", recorded_at=datetime(2026, 9, 23, 14, 0))
    confident = build_name(recording, _long_pair_classification(), taxonomy, NamingSettings())
    flagged = build_name(
        recording,
        _long_pair_classification(),
        taxonomy,
        NamingSettings(),
        is_low_confidence=True,
    )
    assert flagged.filename.endswith(" (review).mp4")
    assert flagged.filename != confident.filename
    assert len(flagged.filename) <= NamingSettings().max_length


def test_truncation_never_strands_a_joiner(recording_factory, taxonomy):
    """"Solvents &.mp4" reads as a name with a word missing, not a short one.

    The old cleanup was `rstrip(" -")`, which tidies a cut landing on a dash
    or a space but not one landing on an "&" — and an "&" sits inside half the
    syllabus short titles, so the cut lands there sooner or later.
    """
    recording = recording_factory("raw.mp4", recorded_at=datetime(2026, 8, 10, 14, 0))
    plan = build_name(
        recording,
        _classification("chem.atomic_structure", "Solvents & Induced Dipoles"),
        taxonomy,
        NamingSettings(max_length=25),
    )
    assert plan.filename == "Aug, 10 - Solvents.mp4"


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
        _classification("chem.2_1", "Relative Atomic Mass"),
        taxonomy,
        NamingSettings(template="{date} - {chapter} - {topic} [{teacher}]{ext}"),
    )
    # "Moles and Stoichiometry", not "Atomic Structure": 2.1 lives in the
    # chapter the user teaches fourth, whatever the syllabus numbering says.
    assert plan.filename == (
        "2026-08-10 - Moles and Stoichiometry - Relative Atomic Mass [Ms Rahman].mp4"
    )
