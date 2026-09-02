"""One chapter, one to three subtopics, never four.

The rules being protected here are the user's, stated about their own archive:

  1. Most recordings cover exactly one syllabus subtopic, and the filename is
     that subtopic's name.
  2. A recording covering two or three gets each of their names.
  3. Never four or more. Hard cap.
  4. A recording belongs to exactly one chapter folder, chosen by weighing the
     transcript per chapter rather than by following the single strongest
     subtopic.
  5. Filenames carry keywords, not syllabus sentences: "Shapes of Molecules",
     not "Intermolecular forces, electronegativity and bond properties".

Rules 3 and 4 are the ones with teeth. A fourth subtopic turns a filename into
a table of contents; the wrong chapter makes the recording unfindable, and
silently so.

The keyword classifier exercised here still returns at most two, and that is a
statement about coverage counting rather than about filenames: it can rank
subtopics but cannot tell "taught this" from "said this word", so a third name
chosen that way would be a guess. Three is reachable through
`classify.openai_classifier`, which reads the lecture.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from pipeline.classify.chapters import chapter_weights, select_chapter, select_subtopics
from pipeline.classify.retriever import Candidate, CandidateSet
from pipeline.classify.service import KeywordClassifier
from pipeline.config import ClassifySettings, NamingSettings
from pipeline.models import Classification, Recording, Source, TopicMatch, Transcript
from pipeline.naming import build_name
from pipeline.naming.template import _fit_label
from pipeline.syllabus import load_taxonomy

# A two-chapter slice in the shape the overlay ingester produces: chapters at
# depth 0 carrying Drive folder names, syllabus subtopics at depth 1 carrying
# both the verbatim Cambridge title and the short label a filename wants.
TAXONOMY_YAML = """
version: 1
subject_id: as_chemistry
subject_name: AS Chemistry
syllabus_code: "9701"
nodes:
  - id: chem.atomic_structure
    title: Atomic Structure
    teaching_order: 1
    coursebook_chapter: 1
    drive_folder_name: "1. Atomic Structure (Chapter 1)"
    keywords: [atom, nucleus]
    children:
      - id: chem.1_1
        title: Particles in the atom and atomic radius
        short_title: Subatomic Particles
        keywords: [proton, neutron, electron]
      - id: chem.1_2
        title: The nucleus of the atom
        short_title: Isotopes
        keywords: [isotope, nucleon number]
  - id: chem.chemical_bonding
    title: Chemical Bonding
    teaching_order: 3
    coursebook_chapter: 4
    drive_folder_name: "3. Chemical Bonding (Chapter 4)"
    keywords: [bond, orbital]
    children:
      - id: chem.3_5
        title: Shapes of molecules and bond angles
        short_title: Shapes of Molecules
        keywords: [vsepr, trigonal planar, bond angle, lone pair]
      - id: chem.3_6
        title: Intermolecular forces, electronegativity and bond properties
        short_title: Induced Dipoles
        keywords: [van der waals, induced dipole, electronegativity]
      - id: chem.3_7
        title: Solubility of simple molecules in polar and non-polar solvents
        short_title: Solubility in Solvents
        keywords: [solubility, polar solvent, hydrogen bonding]
  # Two real short titles that carry an "&" of their own. Twenty-five of the
  # fifty-one chemistry short titles look like this, which is the whole reason
  # the subtopic separator cannot also be "&".
  - id: chem.group_17
    title: Group 17
    teaching_order: 9
    coursebook_chapter: 10
    drive_folder_name: "9. Group 17 (Chapter 10)"
    keywords: [halogen, chlorine]
    children:
      - id: chem.11_2
        title: The chemical properties of the halogen elements and the hydrogen halides
        short_title: Halogens & Hydrogen Halides
        keywords: [halogen, hydrogen halide]
      - id: chem.11_4
        title: The reactions of chlorine
        short_title: Chlorine & Disproportionation
        keywords: [chlorine, disproportionation]
"""


@pytest.fixture
def chem(tmp_path):
    path = tmp_path / "chem.yaml"
    path.write_text(TAXONOMY_YAML, encoding="utf-8")
    return load_taxonomy(path)


def _candidates(chem, coverage: dict[str, float]) -> CandidateSet:
    """A retrieval result with the given per-node coverage.

    Share tracks coverage, which is what happens when nothing is unmatched;
    the tests that care about the difference set it explicitly.
    """
    total = sum(coverage.values()) or 1.0
    items = [
        Candidate(
            node=chem.require(node_id),
            score=value,
            coverage=value,
            share=value / total,
            peak_score=value,
            top_windows=[0],
        )
        for node_id, value in coverage.items()
    ]
    items.sort(key=lambda c: (c.coverage, c.score), reverse=True)
    return CandidateSet(candidates=items, windows=[], unmatched_windows=0)


def _recording(recording_factory, day: int = 2, month: int = 9) -> Recording:
    return recording_factory("raw.mp4", recorded_at=datetime(2026, month, day, 14, 0))


def _named(recording: Recording, chem, node_ids: list[str], **settings) -> str:
    classification = Classification(
        primary=TopicMatch(node_id=node_ids[0], confidence=0.9, coverage=0.7),
        topic_node_ids=node_ids,
    )
    return build_name(recording, classification, chem, NamingSettings(**settings)).filename


# ── Rule 1: one subtopic names the file ───────────────────────────────────────


def test_a_single_subtopic_names_the_file(recording_factory, chem):
    """The common case, and the one the user's own archive is full of."""
    assert (
        _named(_recording(recording_factory, day=2), chem, ["chem.3_5"])
        == "Sep, 2 - Shapes of Molecules.mp4"
    )


def test_short_title_beats_the_verbose_syllabus_title(recording_factory, chem):
    """Rule 5. The Cambridge title for 3.6 is a sentence; a filename is not.

    Both fields stay on the node — the title so a human can find the subtopic
    in the PDF, the short title so the folder listing is scannable — and it is
    the short one that reaches the filename.
    """
    node = chem.require("chem.3_6")
    assert node.title == "Intermolecular forces, electronegativity and bond properties"
    assert (
        _named(_recording(recording_factory, day=23), chem, ["chem.3_6"])
        == "Sep, 23 - Induced Dipoles.mp4"
    )


def test_node_label_is_used_when_no_topic_ids_were_set(recording_factory, chem):
    """The fallback path: a classification carrying only `primary` still gets
    the keyword label rather than the syllabus sentence."""
    classification = Classification(
        primary=TopicMatch(node_id="chem.3_6", confidence=0.9, coverage=0.8)
    )
    plan = build_name(_recording(recording_factory), classification, chem, NamingSettings())
    assert plan.filename == "Sep, 2 - Induced Dipoles.mp4"


# ── Rule 2: two subtopics are joined ──────────────────────────────────────────


def test_two_qualifying_subtopics_produce_both_names(recording_factory, chem):
    """Reproduces one of the user's own filenames."""
    assert (
        _named(_recording(recording_factory, day=23), chem, ["chem.3_7", "chem.3_6"])
        == "Sep, 23 - Solubility in Solvents + Induced Dipoles.mp4"
    )


def test_two_names_still_fit_the_default_length_cap(recording_factory, chem):
    name = _named(_recording(recording_factory, day=23), chem, ["chem.3_7", "chem.3_6"])
    assert len(name) <= NamingSettings().max_length


# ── Rule 3: never four ────────────────────────────────────────────────────────


def test_a_fourth_subtopic_is_refused_at_the_model_boundary():
    """Not truncated — refused.

    Silently dropping the fourth would be friendlier and worse: whatever
    supplied four would go on supplying four, unseen.
    """
    with pytest.raises(ValidationError, match="at most 3 subtopics"):
        Classification(topic_node_ids=["chem.3_5", "chem.3_6", "chem.3_7", "chem.1_1"])


def test_three_subtopics_are_accepted_at_the_model_boundary():
    """The cap moved from two to three; it did not stop existing.

    Three is a description a lecture can earn — the AI classifier reads the
    whole transcript and can say "this taught these three" — where the keyword
    path could only ever rank three. So the boundary has to admit it.
    """
    classification = Classification(topic_node_ids=["chem.3_5", "chem.3_6", "chem.3_7"])
    assert classification.topic_node_ids == ["chem.3_5", "chem.3_6", "chem.3_7"]


def test_a_fourth_subtopic_is_refused_on_assignment_too():
    """Construction already refused it. Assignment is the path the code takes.

    `classify/service.py` builds the Classification and then *assigns*
    `topic_node_ids` to it, and pydantic v2 does not validate assignments
    unless the model asks it to — so the rule held on the one path nothing
    used and was skipped on the one path everything uses.
    """
    classification = Classification(topic_node_ids=["chem.3_5"])
    with pytest.raises(ValidationError, match="at most 3 subtopics"):
        classification.topic_node_ids = ["chem.3_5", "chem.3_6", "chem.3_7", "chem.1_1"]
    assert classification.topic_node_ids == ["chem.3_5"]


def test_the_keyword_classifier_never_selects_a_third_however_strong(chem):
    """Three subtopics of one chapter, all comfortably over both thresholds.

    The cap allows three now; coverage counting still must not produce one.
    `select_subtopics` weighs the runner-up and stops, because a third name
    chosen from window counts asserts something the counts cannot support.
    """
    candidates = _candidates(chem, {"chem.3_5": 0.30, "chem.3_6": 0.25, "chem.3_7": 0.22})
    picked = select_subtopics(candidates, chem, "chem.chemical_bonding", ClassifySettings())
    assert [c.node.id for c in picked] == ["chem.3_5", "chem.3_6"]


def test_the_keyword_cap_survives_the_shared_post_processing(chem):
    """Regression: `select_subtopics` stopping at two was not enough.

    `assign_chapter_and_topics` refills the subtopic list from `secondary`,
    which on the keyword path carries reporting-only candidates that were
    never picked. Raising `naming.max_subtopics` to 3 therefore produced
    three-part keyword filenames through the back door, while the function
    everyone reads to check the rule still returned two.

    Three same-chapter subtopics within a few points of each other is the
    shape that exposes it, and it is not exotic: a lecture working through one
    chapter looks exactly like this to a word count.
    """
    result = _classify(chem, {"chem.3_5": 0.30, "chem.3_6": 0.28, "chem.3_7": 0.26})
    assert result.topic_node_ids == ["chem.3_5", "chem.3_6"]


def test_the_cap_cannot_be_raised_in_configuration():
    """Rule 3 expressed where it cannot be worked around."""
    assert NamingSettings(max_subtopics=3).max_subtopics == 3
    with pytest.raises(ValidationError):
        NamingSettings(max_subtopics=4)


def test_the_cap_is_honoured_when_lowered(recording_factory, chem):
    name = _named(
        _recording(recording_factory, day=23), chem, ["chem.3_7", "chem.3_6"], max_subtopics=1
    )
    assert name == "Sep, 23 - Solubility in Solvents.mp4"


def test_one_subtopics_own_ampersand_is_never_rewritten(recording_factory, chem):
    """Regression: a single title's " & " came back as " + ".

    `_fit_label` rebuilt every label from its parts and re-joined with
    `subtopic_separator`. Half the syllabus short titles carry an "&" of their
    own, so a one-subtopic label was split on the fallback separator and
    reassembled as a two-subtopic one: the 19 August maths lecture was named
    "Estimation + Rounding", a pair of topics that does not exist, from a
    classification that had chosen the single subtopic "Estimation & Rounding".

    83 of the labels in the shipped syllabi are exposed to this, so it is
    checked on the real thing rather than on a fixture.
    """
    name = _named(_recording(recording_factory, day=23), chem, ["chem.11_4"])
    assert name == "Sep, 23 - Chlorine & Disproportionation.mp4"
    assert " + " not in name


def test_a_label_that_fits_is_returned_byte_for_byte(chem):
    """The correctness condition behind the test above, stated directly."""
    settings = NamingSettings()
    for label in (
        "Estimation & Rounding",
        "Acids, Bases & Buffers",
        "Halogens & Hydrogen Halides",
        "Indices I + Surds",
    ):
        assert _fit_label(label, settings.max_length, settings) == label


def test_trimming_puts_back_the_separator_it_took_out(chem):
    """A label that genuinely has to lose a subtopic keeps the joiner it came
    with — dropping a part must not also rewrite the survivors."""
    settings = NamingSettings()
    joined = "Chlorine & Disproportionation + Halogens & Hydrogen Halides"
    assert _fit_label(joined, 35, settings) == "Chlorine & Disproportionation"


def test_two_subtopics_never_read_as_four(recording_factory, chem):
    """Rule 3 as the user actually checks it: by looking at the folder.

    Half the syllabus short titles contain an "&" of their own, so joining two
    of them with " & " produced a name that reads as four subtopics even
    though the classification behind it named two. A name that looks like it
    broke the rule has broken the rule, because the listing is the only place
    the rule is ever inspected.
    """
    name = _named(_recording(recording_factory, day=23), chem, ["chem.11_2", "chem.11_4"])
    assert name == "Sep, 23 - Halogens & Hydrogen Halides + Chlorine & Disproportionation.mp4"
    # One join, so the name reads as two subtopics however many "&" are inside
    # the titles themselves.
    assert name.count(NamingSettings().subtopic_separator) == 1


def test_a_free_text_label_naming_four_subtopics_is_capped(recording_factory, chem):
    """The label is governed wherever it came from, not just where it is set.

    `suggested_topic_label` reaches the filename with no taxonomy behind it on
    two live paths: an abstaining reply still carries one, because the reply
    schema marks every property required, and the unknown-node-id path clears
    the ids and the primary but not the label. Capping at those two call sites
    would leave the next one free to name a fourth subtopic, so the cap lives
    at the point every caller passes through.
    """
    classification = Classification(
        suggested_topic_label="Enthalpy Change & Hess's Law & Bond Energies & Entropy",
        abstain=True,
        abstain_reason="no candidate cleared the floor",
    )
    plan = build_name(
        _recording(recording_factory, day=23), classification, chem, NamingSettings()
    )
    assert plan.filename == "Sep, 23 - Enthalpy Change + Hess's Law + Bond Energies.mp4"


def test_a_verbose_free_text_label_is_held_to_the_length_rule(recording_factory, chem):
    """Rule 5 on the same path. A label is a filename field, not prose."""
    classification = Classification(
        suggested_topic_label=(
            "Intermolecular forces, electronegativity and bond properties in simple molecules"
        ),
        abstain=True,
    )
    plan = build_name(
        _recording(recording_factory, day=23), classification, chem, NamingSettings()
    )
    assert len(plan.filename) <= NamingSettings().max_length
    assert plan.filename.startswith("Sep, 23 - ")
    assert plan.filename.endswith(".mp4")


# ── The second subtopic has to earn its place ─────────────────────────────────


def test_a_second_subtopic_below_the_coverage_floor_is_dropped(chem):
    """Four minutes of ninety is a mention, not a lesson."""
    candidates = _candidates(chem, {"chem.3_5": 0.60, "chem.3_6": 0.15})
    picked = select_subtopics(candidates, chem, "chem.chemical_bonding", ClassifySettings())
    assert [c.node.id for c in picked] == ["chem.3_5"]


def test_a_second_subtopic_that_trails_too_far_is_dropped(chem):
    """Over the absolute floor, under the relative one.

    0.22 clears the 20% floor but is well under 45% of the winner's 0.60. Both
    bars exist because they catch different failures.
    """
    candidates = _candidates(chem, {"chem.3_5": 0.60, "chem.3_6": 0.22})
    picked = select_subtopics(candidates, chem, "chem.chemical_bonding", ClassifySettings())
    assert [c.node.id for c in picked] == ["chem.3_5"]


def test_a_genuinely_shared_lecture_keeps_both(chem):
    candidates = _candidates(chem, {"chem.3_5": 0.35, "chem.3_6": 0.30})
    picked = select_subtopics(candidates, chem, "chem.chemical_bonding", ClassifySettings())
    assert [c.node.id for c in picked] == ["chem.3_5", "chem.3_6"]


# ── Rule 4: one chapter, weighed ──────────────────────────────────────────────


def test_chapter_weight_is_the_sum_of_its_subtopics(chem):
    weights = chapter_weights(
        _candidates(chem, {"chem.1_1": 0.30, "chem.3_5": 0.25, "chem.3_6": 0.20}), chem
    )
    assert weights["chem.chemical_bonding"] == pytest.approx(0.45 / 0.75)
    assert weights["chem.atomic_structure"] == pytest.approx(0.30 / 0.75)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_the_heavier_chapter_wins_even_without_the_top_subtopic(chem):
    """The whole point of weighing chapters rather than following the winner.

    Atomic structure holds the single largest subtopic at 30%. Chemical
    bonding holds two at 25% and 20%, so it holds more of the lecture, and the
    lecture goes in its folder.
    """
    candidates = _candidates(chem, {"chem.1_1": 0.30, "chem.3_5": 0.25, "chem.3_6": 0.20})
    choice = select_chapter(candidates, chem)
    assert choice.chapter_id == "chem.chemical_bonding"
    assert choice.runner_up_id == "chem.atomic_structure"
    assert choice.margin == pytest.approx(0.20, abs=1e-6)


def test_the_chosen_chapter_is_the_one_the_filename_comes_from(chem):
    candidates = _candidates(chem, {"chem.1_1": 0.30, "chem.3_5": 0.25, "chem.3_6": 0.20})
    choice = select_chapter(candidates, chem)
    picked = select_subtopics(candidates, chem, choice.chapter_id, ClassifySettings())
    assert [c.node.id for c in picked] == ["chem.3_5", "chem.3_6"]


def test_an_uncontested_chapter_reports_no_runner_up(chem):
    choice = select_chapter(_candidates(chem, {"chem.3_5": 0.6}), chem)
    assert choice.chapter_id == "chem.chemical_bonding"
    assert choice.runner_up_id is None
    assert choice.margin == pytest.approx(1.0)


def test_nothing_matched_means_no_chapter(chem):
    assert select_chapter(CandidateSet(candidates=[], windows=[]), chem) is None


def test_ties_break_deterministically(chem):
    """A coin flip here misfiles a lecture, so the order is fully specified:
    weight, then the best single subtopic, then the node id."""
    candidates = _candidates(chem, {"chem.1_1": 0.20, "chem.1_2": 0.20, "chem.3_5": 0.40})
    first = select_chapter(candidates, chem)
    reversed_input = CandidateSet(
        candidates=list(reversed(candidates.candidates)), windows=[], unmatched_windows=0
    )
    assert select_chapter(reversed_input, chem).chapter_id == first.chapter_id
    # Equal weight, so the chapter carrying the stronger single subtopic wins.
    assert first.chapter_id == "chem.chemical_bonding"


# ── End to end through the keyword classifier ─────────────────────────────────


def _transcript() -> Transcript:
    return Transcript(
        segments=[], language="en", duration_s=3600.0, model="fixture", provider="test"
    )


def _classify(chem, coverage: dict[str, float]) -> Classification:
    recording = Recording(
        id="t",
        source=Source.MANUAL,
        source_id="t",
        subject_id="as_chemistry",
        original_filename="raw.mp4",
        recorded_at=datetime(2026, 9, 4, 14, 0),
    )
    classifier = KeywordClassifier(chem, ClassifySettings(), NamingSettings())
    return classifier.classify(recording, _transcript(), _candidates(chem, coverage))


def test_classifier_commits_to_one_chapter_and_names_from_inside_it(chem):
    result = _classify(chem, {"chem.1_1": 0.30, "chem.3_5": 0.25, "chem.3_6": 0.20})
    assert result.chapter_node_id == "chem.chemical_bonding"
    assert result.topic_node_ids == ["chem.3_5", "chem.3_6"]
    assert result.suggested_topic_label == "Shapes of Molecules + Induced Dipoles"


def test_a_subtopic_from_another_chapter_never_reaches_the_filename(chem):
    """Even at high coverage. Naming a subtopic from a folder the recording is
    not in contradicts where it was filed."""
    result = _classify(chem, {"chem.3_5": 0.45, "chem.1_1": 0.40})
    assert result.chapter_node_id == "chem.chemical_bonding"
    assert result.topic_node_ids == ["chem.3_5"]
    assert any(m.node_id == "chem.1_1" for m in result.secondary)


def test_a_near_tied_chapter_costs_confidence(chem):
    """Two plausible chapters is genuine uncertainty even when the subtopic
    picture is identical.

    Both cases put chem.3_5 on 0.40 against a rival on 0.35, so the winning
    subtopic's share and its margin are the same in each. The only difference
    is that the rival is inside the chapter in one case and inside a different
    chapter in the other — and that is the difference between misnaming a
    recording and misfiling it.
    """
    within_one_chapter = _classify(chem, {"chem.3_5": 0.40, "chem.3_6": 0.35})
    across_two_chapters = _classify(chem, {"chem.3_5": 0.40, "chem.1_1": 0.35})
    assert across_two_chapters.primary.confidence < within_one_chapter.primary.confidence


def test_the_reasoning_names_the_chapter_and_its_weight(chem):
    result = _classify(chem, {"chem.3_5": 0.45, "chem.1_1": 0.40})
    assert "Chemical Bonding" in result.reasoning
    assert "%" in result.reasoning


def test_nothing_matched_abstains(chem):
    result = _classify(chem, {})
    assert result.abstain
    assert result.primary is None
