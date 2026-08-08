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
    assert top.node.id == "chem.halogenoalkanes"


def test_finds_a_subtopic_rather_than_its_parent(taxonomy, stub_asr):
    """The lecture is about relative atomic mass specifically, even though it
    is full of atomic-structure vocabulary."""
    transcript = _transcript(stub_asr, "lecture_relative_atomic_mass")
    top = _retriever(taxonomy).retrieve(transcript).top(1)[0]
    assert top.node.id == "chem.atomic_structure.rel_atomic_mass"


def test_recap_section_surfaces_as_a_secondary_topic(taxonomy, stub_asr):
    """The lecture opens with several minutes on the previous chapter, and
    the coverage profile should show that rather than hide it."""
    transcript = _transcript(stub_asr, "lecture_relative_atomic_mass")
    profile = _retriever(taxonomy).retrieve(transcript).coverage_profile()
    assert profile["chem.atomic_structure.rel_atomic_mass"] > profile["chem.atomic_structure"] > 0


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


def test_coverage_and_share_answer_different_questions(taxonomy, stub_asr):
    """`coverage` is share of the whole lecture; `share` is share of the part
    that could be classified at all. A lecture padded with admin should not
    look uncertain when its subject matter is unambiguous."""
    transcript = _transcript(stub_asr, "lecture_halogenoalkanes")
    result = _retriever(taxonomy).retrieve(transcript)
    top = result.top(1)[0]
    assert result.unmatched_windows > 0
    assert top.share > top.coverage


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
    assert classification.primary.node_id == "chem.halogenoalkanes"
    assert not service.needs_review(classification)


def test_split_lecture_is_sent_to_review(taxonomy, stub_asr, recording_factory):
    """Correct topic, insufficient certainty. Review is the right outcome —
    a misfiled lecture is worse than an unfiled one because it is invisible.
    """
    recording = recording_factory("raw.mp4", meeting_topic="Chemistry Class")
    transcript = _transcript(stub_asr, "lecture_relative_atomic_mass")
    service = _service(taxonomy)
    classification, _ = service.run(recording, transcript)
    assert classification.primary.node_id == "chem.atomic_structure.rel_atomic_mass"
    assert service.needs_review(classification)


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
