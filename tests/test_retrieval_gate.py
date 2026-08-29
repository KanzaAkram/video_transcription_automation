"""What a window is allowed to count as evidence for.

Both rules here come out of a six-recording audit in which the retriever
named two lectures wrongly and two incompletely. Neither failure was a
scoring failure — BM25 ranked sensibly in both. They were failures of
*attribution*: of deciding which node a window's vocabulary belongs to, and
of deciding that it belongs to nobody.
"""

from __future__ import annotations

from pipeline.classify.retriever import HybridRetriever
from pipeline.config import ClassifySettings, RetrievalSettings
from pipeline.models import Transcript, TranscriptSegment
from pipeline.syllabus import SyllabusNode, Taxonomy
from pipeline.syllabus.taxonomy import LearningOutcome


def _retriever(taxonomy) -> HybridRetriever:
    return HybridRetriever(taxonomy, ClassifySettings(), RetrievalSettings())


def _lecture(*paragraphs: str, minutes: float = 5.0) -> Transcript:
    """A transcript of one paragraph per `minutes`, long enough to window."""
    step = minutes * 60
    segments = [
        TranscriptSegment(start_s=i * step, end_s=(i + 1) * step, text=text)
        for i, text in enumerate(paragraphs)
    ]
    return Transcript(
        segments=segments,
        language="en",
        duration_s=segments[-1].end_s,
        model="test",
        provider="test",
    )


# ── Keywords are phrases, not bags of words ───────────────────────────────────

#: What the 11 and 13 August AS lectures say, thirty times in ninety minutes.
#: Every token of the Mass Spectrometry keyword "mass to charge ratio" is
#: here, in the other order, describing something else entirely.
DEFLECTION = [
    "The electron has a very large charge to mass ratio, so in the electric "
    "field it is deflected much further than anything else in the beam.",
    "A proton carries the same size of charge as the electron but is nearly "
    "two thousand times heavier, so its charge to mass ratio is tiny and it "
    "barely bends towards the negative one of the oppositely charged plates.",
    "The neutron has no charge at all. Whatever the charge to mass ratio of "
    "its neighbours, the neutron passes between the plates undeflected.",
]

#: The same three words, in the order that means a mass spectrometer.
MASS_SPECTROMETRY = [
    "Here is the mass spectrum of the compound. Along the bottom we plot the "
    "mass to charge ratio, and up the side the abundance of each ion.",
    "The molecular ion peak sits at the far right. The base peak is the "
    "tallest peak in the spectrum, and everything shorter than it comes from "
    "fragmentation.",
    "Every fragment ion gives its own line, so reading a mass spectrum means "
    "working out which fragment each mass to charge ratio belongs to.",
]


def test_a_reversed_phrase_does_not_fire_the_keyword(taxonomy):
    """Charge to mass ratio is not mass to charge ratio.

    Same three words, different physics: one is how a beam behaves between
    charged plates, the other is the axis of a mass spectrum. A keyword tested
    against an unordered set of the window's tokens cannot tell them apart,
    and on the real 13 August lecture that handed 27% of the subtopic weight —
    and a 36% runner-up chapter, in a different Drive folder — to a mass
    spectrometer nobody mentioned. "Peak", "spectrum", "spectrometer",
    "abundance" and "fragment" occur zero times in that transcript.
    """
    profile = _retriever(taxonomy).retrieve(_lecture(*DEFLECTION)).coverage_profile()
    assert "chem.22_2" not in profile, "mass spectrometry matched on word order alone"
    assert profile.get("chem.1_1", 0) > 0, "the lecture is about particles in the atom"


def test_the_genuine_phrase_still_fires(taxonomy):
    """The half of the rule that an over-eager fix would break.

    Phrase matching is only worth having if the phrase still matches when it
    is genuinely said — same keyword, same taxonomy, a lecture that really is
    about mass spectrometry.
    """
    profile = _retriever(taxonomy).retrieve(_lecture(*MASS_SPECTROMETRY)).coverage_profile()
    assert profile.get("chem.22_2", 0) > 0


# ── The gate keeps the window, not just the winner ────────────────────────────


def _prose_versus_term_taxonomy() -> Taxonomy:
    """Two nodes with the shape that broke the gate.

    `circle_theorems` is written the way syllabus outcomes really are: whole
    sentences of ordinary teaching language — value, answer, work out, in the
    exam — over technical keywords a surds lecture never says. `surds` carries
    little but its own term.

    On a window of ordinary teaching the prose node therefore wins on raw
    BM25, because it shares more of the window's vocabulary; the term node is
    the only one the lecture actually mentioned. Real taxonomies are full of
    this pair, which is why on the 10 August maths lecture "Circle Theorems I"
    took the top score in nine windows of a lecture with no geometry in it.
    """
    return Taxonomy(
        subject_id="test",
        subject_name="Test",
        nodes=[
            SyllabusNode(
                id="circle_theorems",
                title="Circle theorems",
                keywords=["cyclic quadrilateral", "angle at the centre", "tangent"],
                learning_outcomes=[
                    LearningOutcome(
                        code="1.1",
                        text=(
                            "work out the value of the angle and give a reason for "
                            "your answer, leaving the answer in exact form where the "
                            "question in the exam asks you to"
                        ),
                        verified=True,
                    )
                ],
            ),
            SyllabusNode(id="surds", title="Surds", keywords=["surd", "surd form"]),
        ],
    )


SURDS_LESSON = (
    "Now, the value of root two. If you work it out on the calculator you get "
    "1.414 and so on, but that answer is not exact, and in the exam the "
    "question will ask you to leave it in surd form."
)


def test_a_window_goes_to_the_node_that_has_its_vocabulary():
    """The gate asks the wrong question when it asks it of the winner alone.

    Taking the argmax and *then* asking whether it has keyword hits throws the
    whole window away whenever a hitless node outranks one that genuinely
    matched. On the real 10 August lecture that discarded nineteen of
    twenty-nine windows, several of them solid teaching, because circle
    theorems outscored surds on raw BM25.

    A window is evidence about whichever node had vocabulary in it. Only a
    window where *no* node has any is evidence about nothing.
    """
    result = _retriever(_prose_versus_term_taxonomy()).retrieve(_lecture(SURDS_LESSON))
    assert result.coverage_profile() == {"surds": 1.0}
    assert result.unmatched_windows == 0


def test_a_window_with_no_subject_vocabulary_still_matches_nothing():
    """The gate has to keep refusing, or the fix above is only a way of
    attributing the register to whichever node scores least badly.
    """
    result = _retriever(_prose_versus_term_taxonomy()).retrieve(
        _lecture(
            "Right, before we start, a couple of notices. The hall is being used "
            "on Thursday so we will be in the room across the corridor.",
            "Anyone who has not handed the slip back to the office, please see me "
            "at the front afterwards and I will sort it out for you.",
        )
    )
    assert result.unmatched_windows == len(result.windows)
    assert result.coverage_profile() == {}
