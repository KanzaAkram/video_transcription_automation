"""The ChatGPT classifier, exercised without an API key.

Every test here drives `OpenAIClassifier` through a fake client, so the whole
path — prompt construction, the reply schema, and the conversion of a reply
into a filename — is checked on a machine that has no key and makes no network
call. That is deliberate rather than merely convenient: the guarantees worth
protecting are all on this side of the wire.

The one being protected hardest is that **nothing the model writes reaches a
filename**. The reply carries syllabus node ids and nothing else that names
anything; the labels are rendered from the taxonomy. So a model that invents a
subtopic cannot produce a file named after it — the id is not in the syllabus,
and an id that is not in the syllabus is dropped.
"""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from pipeline.classify.openai_classifier import (
    NO_CHAPTER,
    OpenAIClassifier,
    default_allowed_ids,
    response_schema,
    syllabus_block,
    transcript_block,
)
from pipeline.config import ClassifySettings, NamingSettings
from pipeline.models import Recording, Source, Transcript, TranscriptSegment
from pipeline.naming import build_name
from pipeline.syllabus import load_taxonomy

# One chapter with a lesson layer, mirroring A2 Biology 9700 — the syllabus
# whose subtopics are too coarse to name a single recording, which is why the
# lesson layer exists and why `--level` defaults to it.
TAXONOMY_YAML = """
version: 1
subject_id: a2_biology
subject_name: A2 Biology
syllabus_code: "9700"
nodes:
  - id: bio.topic_12
    title: Energy and respiration
    drive_folder_name: "chap 12-Respiration"
    keywords: [respiration, ATP]
    children:
      - id: bio.12_1
        title: Energy
        short_title: Energy
        keywords: [ATP, energy currency]
        children:
          - id: bio.12_1.atp
            title: Structure and Properties of ATP
            keywords: [adenosine, ribose, hydrolysis]
          - id: bio.12_1.rq
            title: The Respiratory Quotient
            keywords: [RQ, respirometer]
      - id: bio.12_2
        title: Respiration
        short_title: Respiration
        keywords: [glycolysis, Krebs]
        children:
          - id: bio.12_2.glycolysis
            title: Glycolysis
            keywords: [pyruvate, phosphorylation]
          - id: bio.12_2.link_and_krebs
            title: The Link Reaction and the Krebs Cycle
            keywords: [oxaloacetate, decarboxylation]
  - id: bio.topic_13
    title: Photosynthesis
    drive_folder_name: "chap 13-Photosynthesis"
    keywords: [chloroplast, Calvin]
    children:
      - id: bio.13_1
        title: Photosynthesis as an energy transfer process
        short_title: Light Dependent Stage
        keywords: [thylakoid, photophosphorylation]
        children:
          - id: bio.13_1.light_dependent
            title: The Light Dependent Stage
            keywords: [photolysis, photosystem]
"""


@pytest.fixture
def bio(tmp_path):
    path = tmp_path / "bio.yaml"
    path.write_text(TAXONOMY_YAML, encoding="utf-8")
    return load_taxonomy(path)


@pytest.fixture
def transcript() -> Transcript:
    """Twelve minutes in three-minute segments, so timestamps are checkable."""
    return Transcript(
        segments=[
            TranscriptSegment(
                start_s=0, end_s=180, text="Assalam o alaikum, attendance le lete hain."
            ),
            TranscriptSegment(
                start_s=180, end_s=360, text="Glycolysis mein glucose phosphorylate hota hai."
            ),
            TranscriptSegment(start_s=360, end_s=540, text="Pyruvate banta hai, two ATP net gain."),
            TranscriptSegment(
                start_s=540, end_s=720, text="Link reaction mein decarboxylation hoti hai."
            ),
        ],
        language="en",
        duration_s=720,
        model="small",
        provider="faster-whisper",
    )


@pytest.fixture
def recording(tmp_path) -> Recording:
    return Recording(
        id="class-4",
        source=Source.MANUAL,
        source_id="class 4 (9 aug).mp4",
        subject_id="a2_biology",
        original_filename="class 4 (9 aug).mp4",
        recorded_at=datetime(2026, 8, 9, 14, 0),
        duration_s=720,
    )


def _reply(**overrides) -> dict:
    payload = {
        "chapter_id": "bio.topic_12",
        "subtopics": [{"node_id": "bio.12_2.glycolysis", "coverage": 0.7}],
        "confidence": 0.9,
        "evidence": [
            {"quote": "Glycolysis mein glucose phosphorylate hota hai.", "start_s": 180.0},
            {"quote": "Pyruvate banta hai, two ATP net gain.", "start_s": 360.0},
        ],
        "reasoning": "The lecture works through glycolysis end to end.",
        "abstain": False,
        "abstain_reason": "",
    }
    payload.update(overrides)
    return payload


class _FakeCompletions:
    """Records the request and returns a canned reply."""

    def __init__(self, payload: dict, *, refusal: str | None = None, finish: str = "stop"):
        self.payload = payload
        self.refusal = refusal
        self.finish = finish
        self.request: dict | None = None

    def create(self, **kwargs):
        self.request = kwargs
        message = SimpleNamespace(content=json.dumps(self.payload), refusal=self.refusal)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason=self.finish)]
        )


def _classifier(bio, payload: dict, **fake_kwargs) -> tuple[OpenAIClassifier, _FakeCompletions]:
    """A classifier wired to a fake client. No key, no network."""
    classifier = OpenAIClassifier(bio, ClassifySettings(), NamingSettings())
    completions = _FakeCompletions(payload, **fake_kwargs)
    classifier._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return classifier, completions


# ── What the model is shown ───────────────────────────────────────────────────


def test_the_whole_syllabus_is_shown_not_a_retrieved_shortlist(bio):
    """The point of the rewrite, stated as a test.

    Narrowing the options by keyword score before the model reads the lecture
    would reintroduce exactly the failure being replaced: the right answer
    could be missing from the list before anything had read a word of the
    transcript.
    """
    block = syllabus_block(bio, default_allowed_ids(bio))
    for node in bio.all_nodes():
        assert node.id in block, f"{node.id} was hidden from the model"


def test_grouping_nodes_are_shown_but_marked_unchoosable(bio):
    """A chapter has to be visible to be a chapter and must not be a filename.

    The model needs the tree to know which subtopics share a folder — that is
    what makes "one chapter" a decision it can take — but a lecture named after
    the chapter it is already filed in says nothing twice.
    """
    block = syllabus_block(bio, default_allowed_ids(bio))
    chapter_line = next(line for line in block.splitlines() if "bio.topic_12 " in line)
    assert chapter_line.startswith("## CHAPTER")
    after = block.split("bio.topic_12", 1)[1]
    assert "do not return this id" in after.split("bio.12_1")[0]
    # ... and the lesson under it carries no such marker.
    lesson = block.split("bio.12_2.glycolysis", 1)[1].split("####")[0]
    assert "do not return this id" not in lesson


def test_a_chapter_with_no_drive_folder_is_still_shown(tmp_path):
    """Regression: `pending_nodes` is where whole syllabi live.

    A topic the syllabus has and the folder tree does not is parked in
    `Taxonomy.pending_nodes`, and that is not a corner case — all nine IGCSE
    Maths chapters are there, and one of Chemistry's. Walking `nodes` alone
    showed the model an empty syllabus for Maths, so every lecture abstained:
    correct behaviour according to the prompt, for a reason nothing printed.

    Naming a recording is not placing it. A pending chapter names a file
    perfectly well; what it cannot do is receive one, and that is the folder
    resolver's call to make later.
    """
    path = tmp_path / "pending.yaml"
    path.write_text(
        """
version: 1
subject_id: igcse_maths
subject_name: IGCSE Maths
nodes: []
pending_nodes:
  - id: math.number
    title: Number
    children:
      - id: math.1_3
        title: Powers and roots
        short_title: Powers & Roots
        keywords: [surd, index]
""",
        encoding="utf-8",
    )
    maths = load_taxonomy(path)
    allowed = default_allowed_ids(maths)
    assert allowed == {"math.1_3"}
    block = syllabus_block(maths, allowed)
    assert "math.number" in block and "math.1_3" in block


def test_an_empty_syllabus_says_so_instead_of_abstaining_quietly(tmp_path, transcript, recording):
    """"Nothing to choose from" and "this lecture is off-syllabus" are very
    different problems, and only one of them is the operator's to fix."""
    path = tmp_path / "empty.yaml"
    path.write_text(
        "version: 1\nsubject_id: x\nsubject_name: X\nnodes: []\n", encoding="utf-8"
    )
    empty = load_taxonomy(path)
    classifier = OpenAIClassifier(empty, ClassifySettings(), NamingSettings())
    result = classifier.classify(recording, transcript)  # no client needed: never called
    assert result.abstain
    assert "offered nothing to choose from" in result.abstain_reason


def test_the_whole_transcript_goes_in_with_timestamps(bio, transcript):
    """Not the first five minutes, and not a window digest.

    What a lecture is about is a fact about all of it, and the timestamps are
    what let the model tell an opening recap from the subject.
    """
    block = transcript_block(transcript)
    assert "[00:00]" in block and "[09:00]" in block
    assert "decarboxylation" in block  # the last segment, not just the first


def test_an_overlong_transcript_is_trimmed_evenly_not_truncated(bio, transcript):
    """Dropping the tail would delete the second half of a two-topic lecture."""
    block = transcript_block(transcript, char_budget=120)
    assert len(block) <= 200  # trimmed, with a floor per line
    assert block.startswith("[00:00]")
    assert "[09:00]" in block  # the end of the lecture survives


def test_the_lesson_layer_is_what_names_a_file_by_default(bio):
    """A 9700 subtopic is six recordings; a lesson is one."""
    assert default_allowed_ids(bio) == {
        "bio.12_1.atp",
        "bio.12_1.rq",
        "bio.12_2.glycolysis",
        "bio.12_2.link_and_krebs",
        "bio.13_1.light_dependent",
    }


# ── The reply cannot name something the syllabus does not have ────────────────


def test_the_schema_enumerates_the_syllabus(bio):
    """A conforming reply cannot contain an invented id, because the legal
    values *are* the syllabus."""
    allowed = sorted(default_allowed_ids(bio))
    schema = response_schema(["bio.topic_12", "bio.topic_13"], allowed)
    assert schema["properties"]["chapter_id"]["enum"] == [
        "bio.topic_12",
        "bio.topic_13",
        NO_CHAPTER,
    ]
    assert schema["properties"]["subtopics"]["items"]["properties"]["node_id"]["enum"] == allowed


def test_the_schema_is_shaped_for_strict_structured_output(bio):
    """Strict mode requires every property required and no extras, everywhere."""

    def check(node: dict) -> None:
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert sorted(node["required"]) == sorted(node["properties"])
            for value in node["properties"].values():
                check(value)
        if isinstance(node.get("items"), dict):
            check(node["items"])

    check(response_schema(["bio.topic_12"], ["bio.12_2.glycolysis"]))


def test_an_id_the_syllabus_does_not_have_is_dropped(bio, transcript, recording):
    """The enum makes this impossible in a conforming reply. This is the check
    that holds when the enum was too large to send, or the schema is stale."""
    classifier, _ = _classifier(
        bio,
        _reply(
            subtopics=[
                {"node_id": "bio.12_2.oxidative_phosphorylation", "coverage": 0.5},
                {"node_id": "bio.12_2.glycolysis", "coverage": 0.4},
            ]
        ),
    )
    result = classifier.classify(recording, transcript)
    assert result.topic_node_ids == ["bio.12_2.glycolysis"]


def test_a_subtopic_from_another_chapter_is_dropped(bio, transcript, recording):
    """A name that disagrees with the folder is a name that misleads, and only
    one of the two is visible in a listing."""
    classifier, _ = _classifier(
        bio,
        _reply(
            subtopics=[
                {"node_id": "bio.12_2.glycolysis", "coverage": 0.6},
                {"node_id": "bio.13_1.light_dependent", "coverage": 0.3},
            ]
        ),
    )
    result = classifier.classify(recording, transcript)
    assert result.topic_node_ids == ["bio.12_2.glycolysis"]
    assert result.chapter_node_id == "bio.topic_12"


def test_a_reply_naming_nothing_real_abstains(bio, transcript, recording):
    """Never a fallback guess. Misfiling is worse than not filing."""
    classifier, _ = _classifier(
        bio, _reply(subtopics=[{"node_id": "bio.99_9.invented", "coverage": 0.9}])
    )
    result = classifier.classify(recording, transcript)
    assert result.abstain
    assert result.primary is None
    assert "bio.99_9.invented" in result.abstain_reason


# ── Turning a reply into a filename ───────────────────────────────────────────


def test_three_subtopics_all_reach_the_filename(bio, transcript, recording):
    """The user's rule: every subtopic the teacher actually discussed."""
    classifier, _ = _classifier(
        bio,
        _reply(
            subtopics=[
                {"node_id": "bio.12_2.glycolysis", "coverage": 0.4},
                {"node_id": "bio.12_2.link_and_krebs", "coverage": 0.3},
                {"node_id": "bio.12_1.atp", "coverage": 0.2},
            ]
        ),
    )
    result = classifier.classify(recording, transcript)
    assert result.topic_node_ids == [
        "bio.12_2.glycolysis",
        "bio.12_2.link_and_krebs",
        "bio.12_1.atp",
    ]
    name = build_name(recording, result, bio, NamingSettings()).filename
    assert name == (
        "Aug, 9 - Glycolysis + The Link Reaction and the Krebs Cycle + "
        "Structure and Properties of ATP.mp4"
    )


def test_a_fourth_subtopic_is_cut_rather_than_crashing_the_run(bio, transcript, recording):
    """A reply naming four has misbehaved; it has not earned a failed batch.

    `Classification` refuses a fourth outright, so the trimming has to happen
    before the object is built or one bad reply aborts the folder.
    """
    classifier, _ = _classifier(
        bio,
        _reply(
            subtopics=[
                {"node_id": "bio.12_2.glycolysis", "coverage": 0.3},
                {"node_id": "bio.12_2.link_and_krebs", "coverage": 0.3},
                {"node_id": "bio.12_1.atp", "coverage": 0.2},
                {"node_id": "bio.12_1.rq", "coverage": 0.1},
            ]
        ),
    )
    result = classifier.classify(recording, transcript)
    assert len(result.topic_node_ids) == 3
    assert "bio.12_1.rq" not in result.topic_node_ids


def test_the_filename_comes_from_the_syllabus_not_from_the_model(bio, transcript, recording):
    """The reply has no field a name could come from, and this is why.

    Every other guarantee here is a check that can be got round; this one is
    the absence of a path. `suggested_topic_label` is filled from the taxonomy
    afterwards so a reviewer reading rename_plan.json sees the same words the
    file was given.
    """
    classifier, _ = _classifier(bio, _reply())
    result = classifier.classify(recording, transcript)
    assert result.suggested_topic_label == "Glycolysis"
    assert result.primary is not None
    assert result.primary.node_id == "bio.12_2.glycolysis"


def test_evidence_and_reasoning_survive_into_the_sidecar(bio, transcript, recording):
    """An unauditable classifier degrades a folder tree silently."""
    classifier, _ = _classifier(bio, _reply())
    result = classifier.classify(recording, transcript)
    assert len(result.primary.evidence) == 2
    assert result.primary.evidence[0].start_s == 180.0
    assert "glycolysis" in result.reasoning.lower()


def test_the_chapter_weight_is_the_models_own_coverage(bio, transcript, recording):
    """Not a keyword count. The number a reviewer reads should be the number
    the decision was actually made on."""
    classifier, _ = _classifier(
        bio,
        _reply(
            subtopics=[
                {"node_id": "bio.12_2.glycolysis", "coverage": 0.5},
                {"node_id": "bio.12_1.atp", "coverage": 0.3},
            ]
        ),
    )
    result = classifier.classify(recording, transcript)
    assert result.chapter_weights == pytest.approx({"bio.topic_12": 0.8})


# ── Abstaining ────────────────────────────────────────────────────────────────


def test_an_explicit_abstain_is_honoured(bio, transcript, recording):
    classifier, _ = _classifier(
        bio,
        _reply(
            chapter_id=NO_CHAPTER,
            subtopics=[],
            abstain=True,
            abstain_reason="a past-paper session, not a lesson",
        ),
    )
    result = classifier.classify(recording, transcript)
    assert result.abstain
    assert result.abstain_reason == "a past-paper session, not a lesson"
    assert result.topic_node_ids == []


def test_an_abstain_never_leaves_a_label_behind(bio, transcript, recording):
    """A label with no placement behind it is a proposed filename with none of
    the rules that make a filename true — and naming reads it as soon as the
    node ids are empty."""
    classifier, _ = _classifier(bio, _reply(chapter_id=NO_CHAPTER, subtopics=[], abstain=True))
    result = classifier.classify(recording, transcript)
    assert result.suggested_topic_label is None
    name = build_name(recording, result, bio, NamingSettings()).filename
    assert name == "Aug, 9 - Unclassified.mp4"


# ── Which part of the subtopic (the duplicate-name fix) ───────────────────────

# Subtopic 12.1 with two outcomes that are plainly two different lessons — the
# shape of AS Chemistry 1.1, which is seven outcomes over at least two lessons
# and produced two recordings both called "Subatomic Particles".
FOCUS_YAML = """
version: 1
subject_id: a2_biology
subject_name: A2 Biology
nodes:
  - id: bio.topic_12
    title: Energy and respiration
    drive_folder_name: "chap 12-Respiration"
    children:
      - id: bio.12_1
        title: Energy
        short_title: Energy
        keywords: [ATP]
        learning_outcomes:
          - code: "12.1.2"
            text: >-
              describe the features of ATP that make it suitable as the
              universal energy currency
            verified: true
          - code: "12.1.5"
            text: >-
              state that the respiratory quotient is the ratio of carbon
              dioxide produced to oxygen taken in
            verified: true
"""


@pytest.fixture
def focus_bio(tmp_path):
    path = tmp_path / "focus.yaml"
    path.write_text(FOCUS_YAML, encoding="utf-8")
    return load_taxonomy(path)


def _focus_classifier(tax, **reply_kw):
    classifier = OpenAIClassifier(tax, ClassifySettings(), NamingSettings())
    payload = _reply(
        chapter_id="bio.topic_12",
        subtopics=[{"node_id": "bio.12_1", "coverage": 0.8}],
        **reply_kw,
    )
    completions = _FakeCompletions(payload)
    classifier._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return classifier


def test_two_lessons_on_one_subtopic_no_longer_share_a_name(
    focus_bio, transcript, recording
):
    """The whole point of reading the learning outcomes.

    Both lectures are subtopic 12.1, so both used to be called "Energy". The
    syllabus distinguishes them one level down and now so does the filename.
    """
    first = _focus_classifier(
        focus_bio, outcome_codes=["12.1.2"], topic_keywords=["Universal Energy Currency"]
    ).classify(recording, transcript)
    second = _focus_classifier(
        focus_bio, outcome_codes=["12.1.5"], topic_keywords=["Respiratory Quotient"]
    ).classify(recording, transcript)

    naming = NamingSettings()
    assert build_name(recording, first, focus_bio, naming).filename == (
        "Aug, 9 - Universal Energy Currency.mp4"
    )
    assert build_name(recording, second, focus_bio, naming).filename == (
        "Aug, 9 - Respiratory Quotient.mp4"
    )


def test_a_focus_word_the_syllabus_never_used_is_refused(focus_bio, transcript, recording):
    """The one place the model chooses words, so the one place to check them.

    "Mitochondria" may well be true of the lecture; it is not in outcome
    12.1.2, so it cannot go in the filename. Dropped whole rather than
    repaired, because a repaired phrase reads as though the syllabus said
    something it did not.
    """
    result = _focus_classifier(
        focus_bio, outcome_codes=["12.1.2"], topic_keywords=["Energy Currency in Mitochondria"]
    ).classify(recording, transcript)

    assert result.focus is None
    assert build_name(recording, result, focus_bio, NamingSettings()).filename == (
        "Aug, 9 - Energy.mp4"
    )


def test_a_focus_is_grounded_only_in_the_outcomes_it_claimed(
    focus_bio, transcript, recording
):
    """Grounding in an outcome the lecture did not teach is grounding in the
    wrong thing. "Respiratory Quotient" is real syllabus wording — from the
    other outcome."""
    result = _focus_classifier(
        focus_bio, outcome_codes=["12.1.2"], topic_keywords=["Respiratory Quotient"]
    ).classify(recording, transcript)
    assert result.focus is None


def test_connecting_words_do_not_have_to_be_in_the_syllabus(
    focus_bio, transcript, recording
):
    """Otherwise "Behaviour in an Electric Field" fails on the word "an"."""
    result = _focus_classifier(
        focus_bio, outcome_codes=["12.1.5"], topic_keywords=["Ratio of Carbon Dioxide to Oxygen"]
    ).classify(recording, transcript)
    assert result.focus == "Ratio of Carbon Dioxide to Oxygen"


def test_a_grammatical_variant_of_syllabus_wording_is_allowed(
    focus_bio, transcript, recording
):
    """Grounding is a quoting rule, not a spelling rule.

    Cambridge writes outcomes as instructions - "describe the features of ATP"
    - and a filename wants the noun phrase. Demanding the exact surface form
    banned "Describing" and left names that are not English, so a word is
    accepted when its stem matches one the syllabus used. The word still has to
    be the syllabus', only conjugated.
    """
    result = _focus_classifier(
        focus_bio, outcome_codes=["12.1.5"], topic_keywords=["Ratios of Produced Carbon Dioxide"]
    ).classify(recording, transcript)
    # "ratios" from "ratio", "produced" from "produced" - all syllabus stems.
    assert result.focus == "Ratios of Produced Carbon Dioxide"


def test_no_focus_leaves_the_plain_subtopic_name(focus_bio, transcript, recording):
    """A lecture that worked evenly through the subtopic has nothing to narrow,
    and a focus restating the heading would add length and no information."""
    result = _focus_classifier(focus_bio, outcome_codes=[], topic_keywords=[]).classify(
        recording, transcript
    )
    assert result.focus is None
    assert build_name(recording, result, focus_bio, NamingSettings()).filename == (
        "Aug, 9 - Energy.mp4"
    )


def test_the_outcome_codes_are_kept_for_audit(focus_bio, transcript, recording):
    """What the focus was drawn from, so a wrong one is diagnosable."""
    result = _focus_classifier(
        focus_bio, outcome_codes=["12.1.5"], topic_keywords=["Respiratory Quotient"]
    ).classify(recording, transcript)
    assert result.outcome_codes == ["12.1.5"]


def test_several_keywords_are_joined_with_the_subtopic_separator(
    focus_bio, transcript, recording
):
    """Two keywords render as "A + B", which is the shape the user asked for
    and the shape the archive already uses for two subtopics."""
    result = _focus_classifier(
        focus_bio,
        outcome_codes=["12.1.5"],
        topic_keywords=["Respiratory Quotient", "Carbon Dioxide"],
    ).classify(recording, transcript)
    assert result.focus == "Respiratory Quotient + Carbon Dioxide"
    name = build_name(recording, result, focus_bio, NamingSettings()).filename
    assert name == "Aug, 9 - Respiratory Quotient + Carbon Dioxide.mp4"
    # The subtopic heading is not in it.
    assert "Energy" not in name


def test_one_bad_keyword_does_not_cost_the_others(focus_bio, transcript, recording):
    """Per keyword, not all-or-nothing: three phrases of which one is invented
    should cost that one phrase, not the filename."""
    result = _focus_classifier(
        focus_bio,
        outcome_codes=["12.1.5"],
        topic_keywords=["Respiratory Quotient", "Cathode Ray Tubes"],
    ).classify(recording, transcript)
    assert result.focus == "Respiratory Quotient"


def test_a_keyword_repeated_in_other_words_is_dropped(focus_bio, transcript, recording):
    """"Respiratory Quotient + Quotient of Respiration" claims the lecture
    covered two things when it covered one."""
    result = _focus_classifier(
        focus_bio,
        outcome_codes=["12.1.5"],
        topic_keywords=["Respiratory Quotient", "Quotient Respiratory"],
    ).classify(recording, transcript)
    assert result.focus == "Respiratory Quotient"


# ── The class about the course, not about the subject ─────────────────────────

ORIENTATION_YAML = TAXONOMY_YAML.replace(
    "nodes:\n",
    """nodes:
  - id: bio.introduction
    title: Introduction
    drive_folder_name: "0. Introduction"
    exclude_from_classification: true
    aliases: [orientation, course intro]
""",
    1,
)


@pytest.fixture
def bio_with_intro(tmp_path):
    path = tmp_path / "bio_intro.yaml"
    path.write_text(ORIENTATION_YAML, encoding="utf-8")
    return load_taxonomy(path)


def _orientation_reply(**kw) -> dict:
    return _reply(
        chapter_id=NO_CHAPTER,
        subtopics=[],
        session_type="orientation",
        confidence=0.97,
        abstain=True,
        abstain_reason="course orientation: year plan, paper codes, calculator policy",
        reasoning="The teacher outlines the year and says they are not explaining concepts.",
        **kw,
    )


def test_an_orientation_class_is_named_and_filed_when_the_syllabus_has_a_folder(
    bio_with_intro, transcript, recording
):
    """The one recording whose nature is least in doubt used to be the one
    named "Unclassified".

    It is the first session of the year — year plan, paper codes, calculator
    policy — and it has a folder. Naming it for what it is beats naming it
    after the classifier's own uncertainty.
    """
    classifier = OpenAIClassifier(bio_with_intro, ClassifySettings(), NamingSettings())
    completions = _FakeCompletions(_orientation_reply())
    classifier._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    result = classifier.classify(recording, transcript)

    assert not result.abstain
    assert result.chapter_node_id == "bio.introduction"
    assert result.topic_node_ids == ["bio.introduction"]
    assert build_name(recording, result, bio_with_intro, NamingSettings()).filename == (
        "Aug, 9 - Introduction.mp4"
    )


def test_an_orientation_class_clears_the_review_gate(bio_with_intro, transcript, recording):
    """Coverage is stated as 1.0 rather than read off the reply.

    The reply reports coverage per subtopic and an orientation session has
    none, so a coverage of 0 would fail `coverage_floor` and send to review a
    decision that nothing about it is uncertain.
    """
    settings = ClassifySettings()
    classifier = OpenAIClassifier(bio_with_intro, settings, NamingSettings())
    completions = _FakeCompletions(_orientation_reply())
    classifier._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    result = classifier.classify(recording, transcript)
    assert not result.needs_review(settings.review_threshold, settings.coverage_floor)


def test_an_orientation_class_is_still_named_without_a_folder(bio, transcript, recording):
    """No Introduction chapter: keep the name, keep the review.

    The name is what a reader wants and costs nothing to be right about. The
    placement genuinely has nowhere to go, and inventing one would file the
    recording into a folder picked at random.
    """
    settings = ClassifySettings()
    classifier = OpenAIClassifier(bio, settings, NamingSettings())
    completions = _FakeCompletions(_orientation_reply())
    classifier._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    result = classifier.classify(recording, transcript)

    assert result.abstain
    low = result.needs_review(settings.review_threshold, settings.coverage_floor)
    assert low
    assert "no 'Introduction' chapter" in result.abstain_reason
    # `is_low_confidence` comes from the caller, exactly as rename_folder.py
    # computes it — the marker is the review decision, not part of the label.
    name = build_name(recording, result, bio, NamingSettings(), is_low_confidence=low)
    assert name.filename == "Aug, 9 - Introduction (review).mp4"


def test_the_orientation_label_and_the_folder_are_one_string(bio_with_intro):
    """Rename the folder in config and the lookup follows, or neither moves."""
    from pipeline.classify.openai_classifier import orientation_node

    assert orientation_node(bio_with_intro, "Introduction").id == "bio.introduction"
    assert orientation_node(bio_with_intro, "introduction").id == "bio.introduction"
    assert orientation_node(bio_with_intro, "Course Admin") is None


def test_orientation_is_offered_to_the_model_as_a_session_type(bio, transcript, recording):
    classifier, completions = _classifier(bio, _reply())
    classifier.classify(recording, transcript)
    schema = completions.request["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["session_type"]["enum"] == ["lesson", "orientation"]
    assert "session_type" in schema["required"]
    system = completions.request["messages"][0]["content"]
    assert "orientation" in system and "first session of the year" in system


def test_the_intro_chapter_is_never_a_candidate_for_a_real_lesson(bio_with_intro):
    """`exclude_from_classification` still holds where it matters.

    An intro session name-checks half the syllabus in forty minutes; letting it
    compete on retrieval would have it outscore the lecture that actually
    taught the topic. It is reachable only once the model has said what kind of
    session this is.
    """
    allowed = default_allowed_ids(bio_with_intro)
    assert "bio.introduction" not in allowed
    block = syllabus_block(bio_with_intro, allowed)
    assert "bio.introduction" not in block


def test_a_refusal_routes_to_review_rather_than_raising(bio, transcript, recording):
    classifier, _ = _classifier(bio, _reply(), refusal="I can't help with that")
    result = classifier.classify(recording, transcript)
    assert result.abstain
    assert "declined" in result.abstain_reason


def test_a_truncated_reply_is_an_error_not_a_guess(bio, transcript, recording):
    """The JSON is emitted last, so a reply cut short is not partial — it is
    absent. Failing loudly is the only honest option."""
    classifier, _ = _classifier(bio, _reply(), finish="length")
    with pytest.raises(RuntimeError, match="output"):
        classifier.classify(recording, transcript)


# ── The request itself ────────────────────────────────────────────────────────


def test_the_request_pins_the_schema_and_the_model(bio, transcript, recording):
    classifier, completions = _classifier(bio, _reply())
    classifier.classify(recording, transcript)
    request = completions.request
    assert request["model"] == ClassifySettings().openai_model
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert "reasoning_effort" not in request  # unset means the model's default


def test_reasoning_effort_is_sent_only_when_configured(bio, transcript, recording):
    """So the same call reaches reasoning and non-reasoning models alike."""
    classifier = OpenAIClassifier(
        bio, ClassifySettings(openai_reasoning_effort="high"), NamingSettings()
    )
    completions = _FakeCompletions(_reply())
    classifier._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    classifier.classify(recording, transcript)
    assert completions.request["reasoning_effort"] == "high"


def test_the_secondary_bars_reach_the_model_from_settings(bio, transcript, recording):
    """A prompt stating a bar the configuration does not hold is worse than one
    stating no bar, because it reads as though the two agree.

    Without these numbers the first live run named a third subtopic at 10%
    coverage in three lectures out of five, and cited no quote for any of them:
    "taught, not mentioned" is a rule the model cannot apply without knowing
    where the house draws the line.
    """
    settings = ClassifySettings(
        secondary_subtopic_min_coverage=0.25, secondary_subtopic_ratio=0.5
    )
    classifier = OpenAIClassifier(bio, settings, NamingSettings())
    completions = _FakeCompletions(_reply())
    classifier._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    classifier.classify(recording, transcript)
    system = completions.request["messages"][0]["content"]
    assert "at least 25% of the lecture" in system
    assert "at least 50% of what the first subtopic took" in system
    assert "if you cannot quote the teacher teaching it" in system.lower()


def test_the_prompt_tells_the_model_not_to_trust_the_filename(bio, transcript, recording):
    """ "class 4 (9 aug).mp4" is the input, and it is evidence of nothing."""
    classifier, completions = _classifier(bio, _reply())
    classifier.classify(recording, transcript)
    prompt = completions.request["messages"][1]["content"]
    assert "class 4 (9 aug).mp4" in prompt
    assert "Do not classify from them" in prompt
