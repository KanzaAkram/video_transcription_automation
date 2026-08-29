"""What the opening of a lecture is allowed to decide.

Teachers announce the lesson in the first few minutes, in words far more
decisive than anything later on. Every opening quoted here is verbatim from
one of the seven August recordings, because the value of a cue list is
entirely a question of whether these two teachers actually talk that way.

The tests are organised around the two failures the announcement fixes and the
two it must not cause:

  * 19 August is misfiled without it — a 4-4 subtopic tie sent a lecture on
    the quadratic formula into the Number folder.
  * 5 August is misnamed without it — a course-orientation session with no
    syllabus content was named "Averages & Spread" off 12% coverage.
  * 10 August is broken *by* it if the boost is allowed to touch coverage —
    its opening announces the revision half, and the new material is Surds.
  * 11 August is broken by a naive not-a-lesson rule — its opening is thick
    with course administration and it teaches for forty-nine minutes.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from pipeline.classify.intent import IntentReader, IntentSettings
from pipeline.classify.retriever import Candidate, CandidateSet, Window
from pipeline.classify.service import KeywordClassifier, intent_ranked
from pipeline.config import ClassifySettings, NamingSettings
from pipeline.models import Recording, Source, Transcript, TranscriptSegment
from pipeline.syllabus import Taxonomy, load_taxonomy

REPO_ROOT = Path(__file__).resolve().parents[1]


# ── Verbatim openings ─────────────────────────────────────────────────────────

#: 19 Aug maths. Truth: 2.5 Equations, the quadratic formula, in Algebra and
#: Graphs. The whole lecture ties 4-4 with Estimation & Rounding; this is the
#: first sentence out of the teacher's mouth.
QUADRATIC_OPENING = (
    "So, what we have is, if we want to get the value of x in a quadratic "
    "equation, and if we want to get the value of x in a quadratic equation "
    "like this, ax squared, px plus c, one of the most important terms is "
    "this a, b and c, x is unknown, which we have to solve in a quadratic "
    "formula. So, does anyone remember the formula that we used to solve such "
    "a quadratic equation?"
)

#: 16 Aug chemistry. Truth: 2.1 Relative Atomic Mass.
RELATIVE_MASS_OPENING = (
    "Alright, so we have read the definition of IGCSC related to atomic mass. "
    "Do you remember? So, we said relative atomic mass is basically the weight "
    "average of the mass of isotope of the element when compared with carbon "
    "12 isotopes."
)

#: 5 Aug maths. Not a lesson at all: course orientation and exam logistics.
ORIENTATION_OPENING = (
    "So, in the grade 10 midterm that will happen in December, all of these "
    "grade 9 chapters will be assessed, which we will not repeat in grade 10. "
    "All 75% of the syllabus will be covered by you. After that, you will have "
    "two exams. One is a mock exam of the school. One is a tuition mock exam. "
    "And then finally a board exam. I will be sharing with you the resources "
    "and the recordings for grade 9, so I recommend that you at least watch "
    "the recordings and practice the assignments. As soon as you start your "
    "academic year, I am giving you an overview."
)

#: 11 Aug chemistry. Administration *and* a topic — the case a rule that
#: triggered on admin vocabulary alone would throw away.
ADMIN_THEN_TOPIC_OPENING = (
    "Alright, so last class can tell you, we did lesson planning and as per "
    "the lesson plan, in August, the first chapter we covered, this is atomic "
    "structure, which is chapter 1 of your book. What is an element? It is a "
    "pure substance that is not combined with anything else, only made up of "
    "one type of atom."
)

#: 10 Aug maths. The trap: the opening announces the *revision* half. The new
#: material is Surds, 33 minutes of it, and the correct name keeps both.
REVISION_OPENING = (
    "Alright, so one of the topics that we studied in grade 9, that we intend "
    "to revise in this class, is indices, right? And because this is supposed "
    "to be one of the foundational chapters, we need to revise it before we "
    "start our grade 10 long chapters."
)


@pytest.fixture
def maths() -> Taxonomy:
    return load_taxonomy(REPO_ROOT / "config" / "syllabi" / "igcse-maths-0580.yaml")


def _transcript(text: str, *, duration_s: float = 3000.0) -> Transcript:
    """A transcript whose opening is `text` and which says nothing after it.

    Enough for the intent reader, which never looks past its window. Coverage
    over the rest of the lecture is supplied separately, by `_candidates`, so
    that a test can state "the opening said X, the transcript as a whole said
    Y" without having to engineer a transcript that produces Y.
    """
    return Transcript(
        segments=[TranscriptSegment(start_s=0.0, end_s=240.0, text=text)],
        language="en",
        duration_s=duration_s,
        model="test",
        provider="test",
    )


def _candidates(taxonomy: Taxonomy, coverage: dict[str, float]) -> CandidateSet:
    """A candidate set with exactly the coverage profile asked for."""
    total = sum(coverage.values()) or 1.0
    ranked = [
        Candidate(
            node=taxonomy.require(node_id),
            score=value,
            coverage=value,
            share=value / total,
            peak_score=1.0,
            top_windows=[0],
        )
        for node_id, value in coverage.items()
    ]
    ranked.sort(key=lambda c: (c.coverage, c.score), reverse=True)
    return CandidateSet(
        candidates=ranked,
        windows=[Window(index=0, start_s=0.0, end_s=240.0, text="...")],
        unmatched_windows=0,
    )


def _classify(taxonomy: Taxonomy, opening: str, coverage: dict[str, float]):
    classifier = KeywordClassifier(taxonomy, ClassifySettings(), NamingSettings())
    transcript = _transcript(opening)
    recording = Recording(
        id="t",
        source=Source.MANUAL,
        source_id="t",
        subject_id=taxonomy.subject_id,
        original_filename="t.mp4",
        recorded_at=datetime(2026, 8, 19),
        duration_s=transcript.duration_s,
    )
    return classifier.classify(recording, transcript, _candidates(taxonomy, coverage))


# ── Extraction ────────────────────────────────────────────────────────────────


def test_the_opening_names_the_topic_the_lecture_is_about(maths):
    """19 August announces the quadratic formula in its first sentence.

    Whole-transcript retrieval cannot separate that lecture from Estimation &
    Rounding — they tie at four windows each. The opening is not ambiguous at
    all, and this is the signal that says so.
    """
    signal = IntentReader(maths).read(_transcript(QUADRATIC_OPENING))
    assert signal.strength("math.2_5") == 1.0, signal.describe()
    assert signal.intent_sentences, "no sentence was read as stating an intention"
    assert "quadratic" in signal.matches[0].quote.lower()


def test_the_opening_is_read_against_the_syllabus_it_was_given(taxonomy):
    """The same machinery, a different subject: 16 August names 2.1."""
    signal = IntentReader(taxonomy).read(_transcript(RELATIVE_MASS_OPENING))
    assert signal.strength("chem.2_1") == 1.0, signal.describe()


def test_only_sentences_that_state_an_intention_are_scored(maths):
    """A sentence with the vocabulary but no cue is not an announcement.

    Otherwise this is just a second retrieval pass over the first five
    minutes, and the distinction it exists to draw — announced versus merely
    said — is not being drawn at all.
    """
    signal = IntentReader(maths).read(
        _transcript("The bearing of B from A is 072 degrees. Measure it clockwise from north.")
    )
    assert signal.intent_sentences == ()
    assert signal.matches == ()


def test_the_window_length_is_configurable(maths):
    """Five minutes is a measurement on seven recordings, not a law."""
    late = Transcript(
        segments=[
            TranscriptSegment(start_s=0.0, end_s=400.0, text="Good morning. Can you hear me?"),
            TranscriptSegment(start_s=400.0, end_s=800.0, text=QUADRATIC_OPENING),
        ],
        language="en",
        duration_s=3000.0,
        model="test",
        provider="test",
    )
    assert IntentReader(maths).read(late).matches == ()
    wider = IntentReader(maths, IntentSettings(window_seconds=900.0)).read(late)
    assert wider.strength("math.2_5") == 1.0


# ── Not a lesson ──────────────────────────────────────────────────────────────


def test_an_admin_only_opening_is_detected_as_not_a_lesson(maths):
    """5 August is a course-orientation session. It teaches nothing."""
    signal = IntentReader(maths).read(_transcript(ORIENTATION_OPENING))
    assert signal.looks_administrative, signal.describe()
    assert not signal.opening_node_ids, "the orientation named a syllabus topic"


def test_admin_vocabulary_plus_a_topic_is_a_lesson(taxonomy):
    """11 August opens on the lesson plan, the chapter and the book — and then
    teaches atomic structure for forty-nine minutes.

    Admin cues alone must never trigger the abstain. "lesson plan" is
    deliberately in both cue lists, so this is the test that stops the
    not-a-lesson rule from eating a real lecture that happens to start with a
    word about the course.
    """
    signal = IntentReader(taxonomy).read(_transcript(ADMIN_THEN_TOPIC_OPENING))
    assert signal.admin_cues_found, "the fixture no longer contains admin vocabulary"
    assert not signal.looks_administrative, signal.describe()
    assert signal.opening_node_ids & {"chem.1_1", "chem.atomic_structure"}


def test_a_not_a_lesson_opening_abstains_rather_than_forcing_a_topic(maths):
    """The outcome that matters: no filename claiming a topic nobody taught.

    Abstaining routes the recording to a human, which is correct for a course
    orientation. Naming it "Averages & Spread" off 12% coverage — which is
    what happened — is a misfiling that is invisible in the folder listing.
    """
    result = _classify(
        maths,
        ORIENTATION_OPENING,
        {"math.9_3": 0.125, "math.1_12": 0.0625, "math.2_6": 0.0625},
    )
    assert result.abstain
    assert result.primary is None
    assert result.topic_node_ids == []
    assert result.suggested_topic_label is None, "an abstain must not propose a filename"
    assert result.needs_review(0.85, 0.25)


def test_a_lecture_that_opens_on_logistics_and_then_teaches_is_not_abstained(maths):
    """The third condition, and the one that keeps the opening subordinate.

    Same administrative opening; this time the transcript as a whole is
    unambiguous. Coverage is the primary signal, so it wins.
    """
    result = _classify(maths, ORIENTATION_OPENING, {"math.1_7": 0.62, "math.1_18": 0.08})
    assert not result.abstain
    assert result.primary is not None and result.primary.node_id == "math.1_7"


# ── Boost, not override ───────────────────────────────────────────────────────


def test_the_announcement_places_a_lecture_its_windows_could_not(maths):
    """19 August: the whole reason this exists.

    Four windows of Solving Equations against four of Estimation & Rounding,
    and Number carries two more single-window subtopics to Algebra's one, so
    the folder went to Number on a tie the transcript could not break. The
    opening breaks it.
    """
    result = _classify(
        maths,
        QUADRATIC_OPENING,
        {
            "math.2_5": 4 / 13,
            "math.1_9": 4 / 13,
            "math.1_7": 1 / 13,
            "math.2_2": 1 / 13,
            "math.1_3": 1 / 13,
        },
    )
    assert result.chapter_node_id == "math.algebra_and_graphs"
    assert result.topic_node_ids == ["math.2_5"]


def test_the_announcement_cannot_drop_a_subtopic_the_lecture_taught(maths):
    """10 August, and the property that decides where the boost is applied.

    The opening announces indices — the *revision* half. Surds is the new
    material and 21% of the lecture, and the filename has to keep it. A boost
    applied to coverage lifts Indices I to 63% and pushes Surds under the 45%
    relative bar, so "Indices I + Surds" silently becomes "Indices I". The
    boost is therefore confined to the ranking, and the rule that asks how
    much of the lecture a subtopic was worth still reads the real number.
    """
    result = _classify(
        maths,
        REVISION_OPENING,
        {"math.1_7": 0.3793, "math.1_18": 0.2069, "math.1_3": 0.0345},
    )
    assert result.topic_node_ids == ["math.1_7", "math.1_18"]


def test_the_announcement_cannot_overturn_a_decisive_transcript(maths):
    """A lecture that is plainly about one thing stays about that thing.

    The opening here names Surds and nothing else, exactly as 12 August's
    does; the lecture is indices from end to end. Naming it for the aside
    would be the announcement overriding the transcript, which is the failure
    mode this whole design is arranged around.
    """
    result = _classify(
        maths,
        "In the 10th class, which I have told you about the surds, this is grade 10.",
        {"math.1_7": 0.5333, "math.1_18": 0.0667, "math.2_13": 0.0667},
    )
    assert result.topic_node_ids == ["math.1_7"]


def test_the_boost_is_bounded_below_what_would_unsettle_a_decided_chapter():
    """The bound, stated as the relationship it has to hold rather than as a
    number that happens to work.

    `DECISIVE_CHAPTER_MARGIN` is this classifier's own claim about when a
    chapter is no longer in doubt. Two normalised weights a margin `m` apart
    are at their closest in ratio when they are the only two, at
    (1 + m)/2 : (1 - m)/2, so a multiplicative lift of (1 + b) can close a
    decisive gap once 1 + b exceeds (1 + m)/(1 - m). Moving either constant
    without the other fails here instead of quietly widening the boost.
    """
    margin = KeywordClassifier.DECISIVE_CHAPTER_MARGIN
    closest_decisive_ratio = (1 + margin) / (1 - margin)
    most_the_opening_can_lift = 1 + KeywordClassifier.INTENT_BOOST
    assert most_the_opening_can_lift < closest_decisive_ratio
    # And above the measured 1/3 that 19 August needs, or it does nothing.
    assert KeywordClassifier.INTENT_BOOST > 1 / 3, "the boost no longer places 19 August"


def test_the_opening_cannot_conjure_a_topic_the_lecture_never_taught(maths):
    """The lift is multiplicative, so a node at zero coverage stays at zero.

    This is what makes the boost safe to state as an unconditional rule rather
    than as one hedged with floors: the announcement reorders the topics a
    lecture taught and can never introduce one it did not.
    """
    signal = IntentReader(maths).read(_transcript(QUADRATIC_OPENING))
    candidates = _candidates(maths, {"math.1_7": 0.60, "math.2_5": 0.0})
    lifted = intent_ranked(candidates, signal, KeywordClassifier.INTENT_BOOST)
    assert {c.node.id: c.coverage for c in lifted.candidates}["math.2_5"] == 0.0


def test_an_opening_that_announces_nothing_changes_nothing(maths):
    """The no-op case, which is most lectures.

    Worth pinning: the boost is applied on every recording, so "no signal
    means no change" is the property that keeps five already-correct results
    correct.
    """
    candidates = _candidates(maths, {"math.1_7": 0.5, "math.1_18": 0.3})
    signal = IntentReader(maths).read(_transcript("Right. Everybody can hear me? Good."))
    assert intent_ranked(candidates, signal, KeywordClassifier.INTENT_BOOST) is candidates


def test_the_announcement_can_lead_the_filename_within_a_chapter(maths):
    """The boost promotes the primary subtopic too, not only the folder.

    Both subtopics are in Number and neither dominates; the opening announces
    the smaller one. It leads the name — and the one it displaced is refilled
    from its *true* coverage through the ordinary secondary rule, so a
    promotion widens the filename rather than truncating it.
    """
    result = _classify(
        maths,
        REVISION_OPENING,
        {"math.1_9": 0.30, "math.1_7": 0.25},
    )
    assert result.topic_node_ids == ["math.1_7", "math.1_9"]
