"""Retrieval and the confidence gate.

The behaviour worth protecting here is that the system classifies on
*content* and refuses to guess. Both fixtures carry the meeting topic
"Chemistry Class", which is exactly as informative as the real ones.
"""

from __future__ import annotations

from pipeline.classify import ClassificationService
from pipeline.classify.retriever import HybridRetriever
from pipeline.config import ClassifySettings, RetrievalSettings
from pipeline.models import Classification, TopicMatch


def _retriever(taxonomy) -> HybridRetriever:
    return HybridRetriever(taxonomy, ClassifySettings(), RetrievalSettings())


def _transcript(stub_asr, name):
    from pipeline.models import TranscriptionHints

    return stub_asr(name).transcribe(__import__("pathlib").Path("unused"), TranscriptionHints())


# ── Retrieval ─────────────────────────────────────────────────────────────────


def test_finds_the_right_topic_from_content_alone(taxonomy, stub_asr):
    transcript = _transcript(stub_asr, "lecture_halogenoalkanes")
    top = _retriever(taxonomy).retrieve(transcript).top(1)[0]
    assert top.node.id == "chem.15_1"


def test_finds_a_subtopic_rather_than_its_parent(taxonomy, stub_asr):
    """The lecture is full of chapter-level vocabulary, and the answer is
    still the subtopic — that is what makes rule 1 possible.

    "9. Halogenoalkanes" the chapter and 15.1 "Halogenoalkanes" the subtopic
    carry near-identical vocabulary, so this is the hardest version of the
    case: nothing but the depth distinguishes them.
    """
    transcript = _transcript(stub_asr, "lecture_halogenoalkanes")
    result = _retriever(taxonomy).retrieve(transcript)
    top = result.top(1)[0]
    assert top.node.id == "chem.15_1"
    assert top.node.depth == 1
    chapter = next(c for c in result.candidates if c.node.id == "chem.halogenoalkanes")
    assert top.coverage > chapter.coverage


def test_recap_section_surfaces_as_a_secondary_topic(taxonomy, stub_asr):
    """The lecture opens with several minutes on the previous chapter, and
    the coverage profile should show that rather than hide it.

    Asserted at chapter level because that is the level the recap is *from*:
    the recap covers 1.1 and 1.2, the new material 2.1, and they sit in
    different Drive folders. A profile that showed only the winner would make
    the multi-topic case invisible, which is the one case a reviewer needs.
    """
    from pipeline.classify.chapters import chapter_weights

    transcript = _transcript(stub_asr, "lecture_relative_atomic_mass")
    result = _retriever(taxonomy).retrieve(transcript)
    profile = result.coverage_profile()
    # The recap and the new material both show up, as separate subtopics.
    assert profile["chem.1_2"] > 0  # isotopes — the recap
    assert profile["chem.2_1"] > 0  # relative atomic mass — the new material
    weights = chapter_weights(result, taxonomy)
    assert len(weights) > 1, "a lecture that spans chapters must look like one"
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_unrelated_topics_get_no_coverage(taxonomy, stub_asr):
    """Regression: per-window normalisation puts some node at 1.0 in every
    window, so a score-based gate let admin chatter ("can everybody hear me
    at the back") attach coverage to whichever node scored least badly.
    Coverage is gated on actual vocabulary presence instead.
    """
    transcript = _transcript(stub_asr, "lecture_relative_atomic_mass")
    profile = _retriever(taxonomy).retrieve(transcript).coverage_profile()
    assert "chem.states_of_matter" not in profile
    assert "chem.hydrocarbons" not in profile


def test_coverage_and_share_are_equal_when_every_window_matched(taxonomy, stub_asr):
    """The identity the two measures are built on.

    `share` divides by the windows that matched anything, `coverage` by all of
    them, so they coincide exactly when nothing was unmatched. Both fixtures
    are wall-to-wall chemistry against the full 51-subtopic syllabus, so this
    is the case they exercise.
    """
    transcript = _transcript(stub_asr, "lecture_halogenoalkanes")
    result = _retriever(taxonomy).retrieve(transcript)
    assert result.unmatched_windows == 0
    for candidate in result.candidates:
        assert candidate.share == candidate.coverage


def test_coverage_and_share_answer_different_questions(taxonomy, stub_asr):
    """`coverage` is share of the whole lecture; `share` is share of the part
    that could be classified at all. A lecture padded with admin should not
    look uncertain when its subject matter is unambiguous.

    The padding is added here rather than baked into a fixture so the test
    states its own premise: fifteen minutes of genuinely contentless talk on
    the end of an otherwise unambiguous lecture.
    """
    transcript = _transcript(stub_asr, "lecture_halogenoalkanes")
    last = transcript.segments[-1].end_s
    padding = [
        "Right, that is us for today. Before you go, a few notices from the office.",
        "The hall is being used on Thursday so we are in the room across the corridor.",
        "Anyone who has not handed in the permission slip, please see me at the front.",
        "Can whoever borrowed the spare whiteboard pens bring them back. That is all.",
    ]
    for i, text in enumerate(padding):
        transcript.segments.append(
            type(transcript.segments[0])(
                start_s=last + i * 240, end_s=last + (i + 1) * 240, text=text
            )
        )
    transcript.duration_s = transcript.segments[-1].end_s  # `text` derives itself

    result = _retriever(taxonomy).retrieve(transcript)
    top = result.top(1)[0]
    assert result.unmatched_windows > 0, "the padding carries no subject vocabulary"
    # Same lecture, same dominance — but coverage is diluted by the notices
    # and share is not. Share is what confidence is built on, which is why
    # reading the register does not make a lecture look ambiguous.
    assert top.share > top.coverage
    assert top.node.id == "chem.15_1"


def test_windows_overlap_so_transitions_are_not_lost(taxonomy, stub_asr):
    transcript = _transcript(stub_asr, "lecture_halogenoalkanes")
    windows = _retriever(taxonomy).windows(transcript)
    assert len(windows) > 1
    assert windows[1].start_s < windows[0].end_s


# ── The confidence gate ───────────────────────────────────────────────────────


def _service(taxonomy) -> ClassificationService:
    return ClassificationService(
        taxonomy, ClassifySettings(), RetrievalSettings(), use_llm=False
    )


def test_unambiguous_lecture_clears_the_threshold(taxonomy, stub_asr, recording_factory):
    recording = recording_factory("raw.mp4", meeting_topic="Chemistry Class")
    transcript = _transcript(stub_asr, "lecture_halogenoalkanes")
    service = _service(taxonomy)
    classification, _ = service.run(recording, transcript)
    assert classification.primary.node_id == "chem.15_1"
    assert classification.chapter_node_id == "chem.halogenoalkanes"
    assert classification.topic_node_ids == ["chem.15_1"]
    assert not service.needs_review(classification)


def test_split_lecture_is_sent_to_review(taxonomy, stub_asr, recording_factory):
    """Insufficient certainty. Review is the right outcome — a misfiled
    lecture is worse than an unfiled one because it is invisible.

    This fixture genuinely straddles three chapters: two windows recapping
    atomic structure, two on relative atomic mass, two on the mass
    spectrometer (which Cambridge puts in 22.2, taught twelve chapters
    later). No node id is asserted, because there is no right one to assert —
    that is the whole point of the fixture. What must hold is that the system
    says so instead of picking.
    """
    recording = recording_factory("raw.mp4", meeting_topic="Chemistry Class")
    transcript = _transcript(stub_asr, "lecture_relative_atomic_mass")
    service = _service(taxonomy)
    classification, _ = service.run(recording, transcript)
    assert service.needs_review(classification)
    # Still a committed, self-consistent answer — review needs something to
    # show the reviewer, and an answer whose name disagreed with its folder
    # would be worse than no answer.
    assert classification.chapter_node_id
    assert classification.topic_node_ids
    for node_id in classification.topic_node_ids:
        assert taxonomy.require(node_id).parent_id == classification.chapter_node_id


def test_review_gate_triggers_on_low_confidence():
    c = Classification(primary=TopicMatch(node_id="x", confidence=0.5, coverage=0.9))
    assert c.needs_review(threshold=0.85, coverage_floor=0.25)


def test_review_gate_triggers_on_low_coverage():
    """Confident about a topic that occupied four minutes of ninety usually
    means the lecture's real subject is not in the taxonomy at all."""
    c = Classification(primary=TopicMatch(node_id="x", confidence=0.99, coverage=0.05))
    assert c.needs_review(threshold=0.85, coverage_floor=0.25)


def test_abstention_always_goes_to_review():
    assert Classification(abstain=True).needs_review(threshold=0.0, coverage_floor=0.0)


def test_hallucinated_node_id_is_rejected(taxonomy, stub_asr, recording_factory, monkeypatch):
    """Structured output constrains the response *shape*, not its values, so
    an invented node id must still be caught before it reaches placement."""
    from pipeline.classify.service import KeywordClassifier

    service = _service(taxonomy)

    def fake(self, recording, transcript, candidates):
        return Classification(
            primary=TopicMatch(node_id="chem.does_not_exist", confidence=0.99, coverage=0.9)
        )

    monkeypatch.setattr(KeywordClassifier, "classify", fake)
    classification, _ = service.run(
        recording_factory("raw.mp4"), _transcript(stub_asr, "lecture_halogenoalkanes")
    )
    assert classification.abstain
    assert classification.primary is None
    assert service.needs_review(classification)
