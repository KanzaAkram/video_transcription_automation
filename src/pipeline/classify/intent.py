"""What the teacher said the lesson was going to be about.

Coverage answers "how much of this lecture was topic X". It is the right
primary signal and it stays the primary signal. But it is a *count of
windows*, and a count cannot hear the difference between a teacher working
through a topic and a teacher happening to use its vocabulary. On the 19
August maths lecture that difference is the whole answer: "Solving Equations"
and "Estimation & Rounding" each took four of thirteen windows — an exact tie —
and the tie was broken by chapter weight, which filed the recording under
Number instead of Algebra and Graphs. The lecture's first sentence is:

    "So, what we have is, if we want to get the value of x in a quadratic
     equation ... which we have to solve in a quadratic formula."

Teachers announce the lesson. They do it in the first few minutes and they do
it in words far more decisive than anything later on, because later on they
are working examples, and the vocabulary of an example is the vocabulary of
whatever numbers happen to be in it. This module reads that announcement.

It is a **boost, never an override**, and the 10 August lecture is why. Its
opening announces the revision half:

    "one of the topics that we studied in grade 9, that we intend to revise in
     this class, is indices"

while the new material is Surds — 33 minutes of it against 31 of indices. A
rule that let the opening decide would name that lecture "Indices I" and drop
Surds, turning a correct result into a wrong one. So what leaves this module
is a *signal* — which nodes were named and how strongly — and `service.py`
applies it as a bounded lift to a ranking that is still made of coverage.

Two properties keep it honest:

  * The lift is **multiplicative on coverage**, so a node the transcript never
    supported has nothing to multiply. The opening cannot conjure a topic the
    lecture did not teach; it can only reorder topics the lecture did.
  * A "match" here means exactly what it means in `retriever.py` — the same
    tokeniser, the same `_NodeTerms`, the same ordered-adjacency phrase test,
    the same `MIN_KEYWORD_HITS` gate. Those names are private and are imported
    anyway, deliberately: a second definition of "this node's vocabulary is
    present" would be a second thing to keep in step with the first, and the
    two would drift on the first keyword anybody added.

**Not-a-lesson.** The 5 August maths recording is a course-orientation
session — no syllabus topic is taught in it at all — and the classifier used
to name it "Averages & Spread" off twelve percent coverage. Its opening is
course administration end to end: "in the grade 10 midterm that will happen in
December, all of these grade 9 chapters will be assessed ... you will have two
exams. One is a mock exam of the school ... then finally a board exam."
Detecting that needs **both** conditions. Admin vocabulary alone is not
enough: the 11 August chemistry lecture opens "as per the lesson plan, in
August, the first chapter we covered, this is atomic structure, which is
chapter 1 of your book" — lesson plan, chapters, book — and then teaches
atomic structure for forty-nine minutes. What separates the two is that one
names a syllabus topic and the other names none.

**Urdu.** These lectures code-switch, and the brief expected phonetically
rendered Urdu cues here. Measured over all seven transcripts, there are none
worth adding: decoding with ``language="en"`` renders the Urdu as English, and
what little survives phonetically is discourse filler — "beta", "yaar",
"bhai", "nahi", one "hum" in a hundred minutes — never a topic announcement.
What the Urdu does leave behind is English in Urdu word order, and those forms
*are* in the cue list below: "what we have is", "which we have to", "I have
told you about".
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from pipeline.classify.retriever import (
    MIN_KEYWORD_HITS,
    _NodeTerms,
    _WindowVocabulary,
    tokenise,
)
from pipeline.models import Transcript
from pipeline.syllabus import Taxonomy

#: Five minutes. Long enough to hold the announcement plus the sentence or two
#: that elaborates it, short enough that the first worked example — whose
#: vocabulary is the example's and not the lesson's — is still outside. All
#: seven verified openings state their topic well inside it: 19 August in its
#: first sentence, 16 August in its first two.
DEFAULT_WINDOW_SECONDS = 300.0

#: Phrasings these two teachers actually use to state an intention, collected
#: by reading the openings of all seven recordings rather than imagined. The
#: recording each came from is named, because a cue nobody says is a cue that
#: contributes nothing but false positives.
#:
#: Note that "lesson plan" and "we covered" sit here *and* in ADMIN_CUES. That
#: is not a mistake: "as per the lesson plan ... the first chapter we covered,
#: this is atomic structure" (11 Aug) is simultaneously administrative framing
#: and a topic announcement, which is exactly why the not-a-lesson test needs
#: its second condition.
INTENT_CUES: tuple[str, ...] = (
    # 11 Aug chemistry
    "as per the lesson plan",
    "lesson plan",
    "the first chapter we covered",
    "we covered",
    "we did",
    # 13 Aug chemistry
    "let's do",
    "let us do",
    "what we have to",
    "we have just done",
    "i told you",
    "i have told you",
    "i will tell you",
    # 16 Aug chemistry
    "we have read",
    "we said",
    "do you remember",
    # 10 Aug maths
    "one of the topics",
    "we intend to",
    "we intend",
    "in this class",
    "we studied",
    "we have studied",
    "we need to",
    "before we start",
    "let's talk about",
    "let us talk about",
    "moving further",
    # 12 Aug maths
    "our topic",
    "i have told you about",
    "let's learn",
    # 19 Aug maths
    "what we have is",
    "we have to",
    "if we want to",
    "we want to",
    "let's suppose",
    "we need to remember",
    "does anyone remember",
    # 5 Aug maths, and forms common to all of them
    "today we",
    "today i",
    "we will do",
    "we will cover",
    "we will study",
    "we will read",
    "we will start",
    "we will learn",
    "we are going to",
    "we're going to",
    "i am going to",
    "let's start",
    "let's begin",
    "let's move",
    "start with",
    "move on to",
)

#: Course-administration vocabulary: the brief's list, plus the terms the 5
#: August orientation session actually uses. Deliberately absent are "grade 9"
#: and "grade 10", which every maths lecture says constantly ("this is grade
#: 10, this is not grade 9" — 12 Aug), and "revise", which is what a teacher
#: says at the start of a real revision lesson.
ADMIN_CUES: tuple[str, ...] = (
    "midterm",
    "mid term",
    "mock exam",
    "mock",
    "board exam",
    "school exam",
    "exam",
    "exams",
    "assessed",
    "assessment",
    "syllabus",
    "chapter",
    "chapters",
    "marks",
    "grading",
    "timetable",
    "time table",
    "lesson plan",
    "attendance",
    "assignment",
    "assignments",
    "homework",
    "academic year",
    "overview",
    "recordings",
    "resources",
    "tuition",
)

#: Sentence-ish. Whisper punctuates, so this is enough; and a run of text with
#: no terminator at all still comes back as one chunk rather than vanishing.
_SENTENCE = re.compile(r"[^.?!]+[.?!]?")


def _terms_of(*phrases: str) -> _NodeTerms:
    """A `_NodeTerms` built over arbitrary phrases rather than over a node.

    So cue matching goes through the same singles/ordered-phrase machinery the
    node keywords do: "lesson plan" has to appear as those two words adjacent
    and in that order, exactly as "relative atomic mass" does.
    """
    singles: set[str] = set()
    ordered: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for phrase in phrases:
        parts = tuple(tokenise(phrase))
        if not parts or parts in seen:
            continue
        seen.add(parts)
        if len(parts) == 1:
            singles.add(parts[0])
        else:
            ordered.append(parts)
    return _NodeTerms(singles=frozenset(singles), phrases=tuple(ordered))


def sentences(text: str) -> list[str]:
    """Split the opening into statements.

    Intent is scored per sentence, not over the concatenated opening, for the
    same reason retrieval scores per window: a topic named in three sentences
    was stated three times, and one long sentence is not three statements.
    """
    return [s.strip() for s in _SENTENCE.findall(text) if s.strip()]


@dataclass(frozen=True)
class IntentSettings:
    """How the opening is read.

    Both numbers are measurements over seven recordings, not laws about
    teachers, which is why they are configuration rather than constants.
    """

    window_seconds: float = DEFAULT_WINDOW_SECONDS
    #: How many *distinct* admin cues the opening must carry before it can be
    #: called administrative. Measured across the seven openings: the 5 August
    #: orientation carries 15 (midterm, mock exam, mock, board exam, exam,
    #: exams, assessed, syllabus, chapters, assignments, academic year,
    #: overview, recordings, resources, tuition) and no real lesson reaches
    #: more than 2 — 11 August has "chapter" and "lesson plan", 16 August has
    #: "exams" and "marks" off a worked example about paper weightings. Six is
    #: three times the highest real lesson and well under half the orientation.
    admin_cue_minimum: int = 6


@dataclass(frozen=True)
class IntentMatch:
    """One syllabus node the opening named, and how insistently."""

    node_id: str
    #: Keyword hits summed across the intent sentences.
    hits: int
    #: `hits` relative to the strongest match, 0-1. What the boost scales by,
    #: so a lecture that announces one topic and glances off another lifts
    #: them in proportion rather than equally.
    strength: float
    #: The first intent sentence that named this node, for the audit trail.
    quote: str


@dataclass(frozen=True)
class IntentSignal:
    """What the opening announced. Reporting only — it decides nothing here."""

    matches: tuple[IntentMatch, ...] = ()
    #: The sentences the cues fired on, in order.
    intent_sentences: tuple[str, ...] = ()
    #: Every node whose vocabulary appears anywhere in the opening, cue or no
    #: cue. Wider than `matches` on purpose: it is the veto on the
    #: not-a-lesson call, and a veto should be easy to trip.
    opening_node_ids: frozenset[str] = frozenset()
    admin_cues_found: tuple[str, ...] = ()
    window_seconds: float = DEFAULT_WINDOW_SECONDS
    #: Both conditions already combined: the opening is thick with course
    #: administration **and** names no syllabus topic at all.
    looks_administrative: bool = False

    _by_id: dict[str, IntentMatch] = field(default_factory=dict, repr=False, compare=False)

    def strength(self, node_id: str) -> float:
        """0.0 for a node the opening never announced."""
        match = self._by_id.get(node_id)
        return match.strength if match else 0.0

    def describe(self) -> str:
        """One clause for the reasoning line, so a boosted result says so."""
        if self.looks_administrative:
            return (
                "its opening is course administration ("
                + ", ".join(self.admin_cues_found[:5])
                + ") and names no syllabus topic"
            )
        if not self.matches:
            return "its opening announces no syllabus topic"
        return "its opening announces " + ", ".join(
            f"{m.node_id} (x{m.hits})" for m in self.matches[:3]
        )


class IntentReader:
    """Reads the opening of a transcript against one taxonomy.

    Built once per run and reused, like `HybridRetriever`: tokenising every
    node's keywords costs the same here as it does there.
    """

    def __init__(self, taxonomy: Taxonomy, settings: IntentSettings | None = None) -> None:
        self.settings = settings or IntentSettings()
        self.nodes = taxonomy.classifiable_nodes()
        self._terms = [_NodeTerms.of(n) for n in self.nodes]
        self._intent_terms = _terms_of(*INTENT_CUES)
        self._admin_terms = [(cue, _terms_of(cue)) for cue in ADMIN_CUES]
        self._lengths = frozenset(
            len(phrase)
            for terms in (
                *self._terms,
                self._intent_terms,
                *(t for _, t in self._admin_terms),
            )
            for phrase in terms.phrases
        )

    def _vocab(self, text: str) -> _WindowVocabulary:
        return _WindowVocabulary(tokenise(text), self._lengths)

    def read(self, transcript: Transcript) -> IntentSignal:
        opening = transcript.window(0.0, self.settings.window_seconds)
        if not opening.strip():
            return IntentSignal(window_seconds=self.settings.window_seconds)

        opening_vocab = self._vocab(opening)
        opening_nodes = frozenset(
            node.id
            for node, terms in zip(self.nodes, self._terms, strict=True)
            if opening_vocab.hits(terms) >= MIN_KEYWORD_HITS
        )
        admin_found = tuple(
            cue for cue, terms in self._admin_terms if opening_vocab.hits(terms) >= 1
        )

        totals: Counter[int] = Counter()
        quotes: dict[int, str] = {}
        kept: list[str] = []
        for sentence in sentences(opening):
            vocab = self._vocab(sentence)
            if vocab.hits(self._intent_terms) < 1:
                continue
            kept.append(sentence)
            for i, terms in enumerate(self._terms):
                hits = vocab.hits(terms)
                if hits < MIN_KEYWORD_HITS:
                    continue
                totals[i] += hits
                quotes.setdefault(i, sentence)

        best = max(totals.values(), default=0)
        matches = tuple(
            IntentMatch(
                node_id=self.nodes[i].id,
                hits=hits,
                strength=hits / best,
                quote=quotes[i],
            )
            for i, hits in sorted(totals.items(), key=lambda kv: (-kv[1], self.nodes[kv[0]].id))
        )

        return IntentSignal(
            matches=matches,
            intent_sentences=tuple(kept),
            opening_node_ids=opening_nodes,
            admin_cues_found=admin_found,
            window_seconds=self.settings.window_seconds,
            # Admin cues alone must never trigger this: 11 August's opening is
            # full of them and teaches a real topic for the next forty-nine
            # minutes. The absence of any named topic is the necessary second
            # condition.
            looks_administrative=(
                len(admin_found) >= self.settings.admin_cue_minimum and not opening_nodes
            ),
            _by_id={m.node_id: m for m in matches},
        )
