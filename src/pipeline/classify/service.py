"""Turning a transcript into a syllabus location.

Three implementations behind one interface, selected by
`ClassifySettings.provider`:

  * ``OpenAIClassifier``  ChatGPT reads the whole transcript against the whole
    syllabus. The production path — it lives in `classify.openai_classifier`
    because it shares none of the retrieval machinery below.
  * ``LLMClassifier``     retrieval + Claude. The earlier production path,
    kept working.
  * ``KeywordClassifier`` retrieval only. Used by the dry run so the pipeline
    is demonstrable with no API key, and useful as a baseline to measure the
    others against.

The two in this module answer the same two questions in the same order — which
chapter, then which subtopics inside it — and both hand the result to
`assign_chapter_and_topics`, which is where the one-chapter and subtopic-cap
rules are actually enforced rather than merely requested. The order is not
cosmetic: choosing the best subtopic first and inheriting its chapter lets a
single strong subtopic drag a recording into a folder the lecture barely
visited. See `classify.chapters`.

`build_classifier` picks between the three; `ClassificationService` runs
retrieval around whichever it got and applies the confidence gate.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Protocol

from pipeline.classify.chapters import (
    ChapterChoice,
    chapter_of,
    chapter_weights,
    secondary_qualifies,
    select_chapter,
    select_subtopics,
)
from pipeline.classify.intent import IntentReader, IntentSettings, IntentSignal
from pipeline.classify.retriever import Candidate, CandidateSet, HybridRetriever
from pipeline.classify.schema import ClassificationOut, response_schema
from pipeline.config import ClassifySettings, NamingSettings, RetrievalSettings
from pipeline.models import Classification, Evidence, Recording, TopicMatch, Transcript
from pipeline.syllabus import Taxonomy

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You place a recorded lecture into the correct node of a syllabus.

Decide from what was actually taught, evidenced by the transcript. The \
recording's title, filename and meeting topic are unreliable and frequently \
generic — treat them as weak hints that transcript evidence overrides.

You will be given candidate syllabus nodes retrieved by keyword and semantic \
search, a window-by-window digest of the transcript, and a coverage profile \
showing how much of the lecture each candidate accounts for.

The syllabus tree has two levels and they mean different things. A **chapter** \
(depth 0) is one folder on disk. A **subtopic** (depth 1) is one syllabus \
subtopic, and subtopics are what the filename is made of. They do not \
correspond one-to-one: a single syllabus topic is sometimes split across two \
chapters.

Rules:

- The recording belongs to exactly ONE chapter — the one it spends the most \
time in, counting all of that chapter's subtopics together. Never split a \
recording across two chapters. A lecture that spends five minutes recapping \
the previous chapter and fifty minutes on a new one belongs to the new one.
- Choose the primary subtopic from inside that chapter.
- Report further subtopics, also from inside that chapter, only where the \
lecture genuinely taught them: at least a fifth of the lecture each, and at \
least 45% of what the primary got. A subtopic trailing the primary \
four-to-one was mentioned, not taught.
- NEVER more than three subtopics in total. A filename naming four is a table \
of contents, and an unread filename is no better than the one it replaced.
- Cite at least two verbatim quotes with their timestamps. Quotes must appear \
in the transcript you were given. This is what makes the decision auditable.
- Set is_continuation when the lecture opens mid-topic and continues the \
previous lecture rather than starting something new.
- suggested_topic_label is written straight into the filename, so it is \
KEYWORDS ONLY and short — a handful of words, under about 60 characters: \
"Shapes of Molecules", "Sigma and Pie Bonds", "Solubility in Solvents & \
Induced Dipoles". Never a syllabus sentence such as "Intermolecular forces, \
electronegativity and bond properties".
- The label names the same subtopics as the node list and no more, joined \
with " & ". The cap binds the label exactly as it binds the node list — \
naming a fourth there is the same mistake, just spelled differently.
- Abstain if none of the candidates fits — a revision session, an exam \
debrief, or a topic absent from the syllabus. Abstaining sends the lecture to \
a human, which is the correct outcome. Do not force a match: misfiling is \
worse than not filing, because it is invisible.
- When you abstain, leave suggested_topic_label as an empty string. You are \
saying this lecture has no place in the syllabus, so there is no filename to \
propose; a label supplied alongside an abstain would be used as the name of \
the very file you just said you could not place.
- Confidence should reflect genuine uncertainty. Two plausible candidates \
means a confidence below 0.7, not a coin flip reported as 0.95. Two plausible \
*chapters* is worse than two plausible subtopics: the first misfiles the \
recording, the second only misnames it.
"""


# ── Shared post-processing ────────────────────────────────────────────────────


#: How many subtopics the keyword path may name, whatever `max_subtopics`
#: allows. Two, and this is not a tuning knob.
#:
#: The cap in `NamingSettings` is three because a classifier that has *read* a
#: lecture can say it taught three subtopics. Coverage counting cannot: it
#: ranks subtopics by how much of their vocabulary was spoken, and a ranking
#: with three entries is not evidence that three things were taught. Naming the
#: third asserts something the measurement does not support, in the one place
#: nobody would ever check it.
#:
#: Stated here rather than left to `select_subtopics` stopping at two on its
#: own, because it does not hold on its own: `assign_chapter_and_topics` refills
#: the list from `secondary`, which carries reporting-only candidates, and
#: raising `max_subtopics` to 3 duly produced three-part keyword filenames.
KEYWORD_SUBTOPIC_CAP = 2


def assign_chapter_and_topics(
    classification: Classification,
    taxonomy: Taxonomy,
    weights: dict[str, float],
    settings: ClassifySettings,
    naming: NamingSettings,
    *,
    max_subtopics: int | None = None,
) -> Classification:
    """Derive the chapter and the filename subtopics from an already-chosen primary.

    Shared by the two classifiers in this module, so the rules hold whoever
    made the choice. The LLM is *told* them and mostly obeys; this is what
    makes them true. A reply naming a fourth plausible subtopic has not
    misbehaved badly enough to throw away, but it has not earned a four-part
    filename either.

    Works from `primary` and `secondary` rather than from the candidate set,
    because that is all the LLM path has to work with.

    Not used by `classify.openai_classifier`. That path chooses the chapter
    first and the subtopics inside it second, so there is no primary for a
    chapter to be inherited from, and its subtopic list is a judgement about
    what was taught rather than a coverage ranking waiting to be re-filtered
    by coverage thresholds.
    """
    classification.chapter_weights = weights
    if classification.primary is None:
        # No primary means nothing was placed, and a label without a placement
        # is a proposed filename with none of the rules behind it: no chapter,
        # no capped subtopic list, nothing that says how many topics it names.
        # Cleared here rather than at each abstaining call site so it holds for
        # whichever classifier produced the abstain.
        classification.suggested_topic_label = None
        return classification

    chapter_id = chapter_of(taxonomy, classification.primary.node_id)
    classification.chapter_node_id = chapter_id

    # Never above the configured cap, and below it when the caller knows its
    # own evidence does not support that many. See `KEYWORD_SUBTOPIC_CAP`.
    limit = min(naming.max_subtopics, max_subtopics or naming.max_subtopics)
    chosen = [classification.primary.node_id]
    for match in sorted(classification.secondary, key=lambda m: (-m.coverage, m.node_id)):
        if len(chosen) >= limit:
            break
        if match.node_id in chosen:
            continue
        # Rule 4 reaching into naming: a subtopic from another chapter in the
        # filename would contradict the folder the file is sitting in.
        if chapter_of(taxonomy, match.node_id) != chapter_id:
            continue
        # The chapter's own node is not a subtopic. It usually picks up the
        # recap minutes at the start of a lecture, and its name is the name of
        # the folder the file is already in — "Relative Atomic Mass & Atomic
        # Structure" filed under Atomic Structure says nothing twice.
        if match.node_id == chapter_id:
            continue
        if not secondary_qualifies(classification.primary.coverage, match.coverage, settings):
            continue
        chosen.append(match.node_id)
    classification.topic_node_ids = chosen

    # Only fill the label if nobody better already has. The LLM's own label is
    # written for a filename; the node labels are the dependable fallback.
    if not classification.suggested_topic_label:
        labels = [n.label for n in (taxonomy.get(i) for i in chosen) if n is not None]
        if labels:
            classification.suggested_topic_label = naming.subtopic_separator.join(labels)
    return classification


def intent_ranked(candidates: CandidateSet, signal: IntentSignal, bound: float) -> CandidateSet:
    """A ranking view of the candidate set, with announced topics lifted.

    A *view*: nothing here is reported as coverage, and nothing here reaches
    the rule that decides whether a second subtopic earned a place in the
    filename. That rule asks "did this take at least a fifth of the lecture",
    which is a question about time, and the opening is not evidence about
    time. See `KeywordClassifier.INTENT_BOOST` for the measurement that made
    that separation necessary rather than merely tidy.

    Multiplicative, so a node with no coverage has nothing to multiply: the
    opening can reorder the topics a lecture taught, never add one it did not.
    That is also what guarantees a chapter chosen off this view still has real
    candidates in it, which is what lets the caller take its subtopics from the
    unboosted set without checking.
    """
    if not signal.matches or bound <= 0:
        return candidates
    lifted = [
        replace(c, coverage=c.coverage * (1.0 + bound * signal.strength(c.node.id)))
        for c in candidates.candidates
    ]
    lifted.sort(key=lambda c: (c.coverage, c.score), reverse=True)
    return replace(candidates, candidates=lifted)


def _restate(chapter_id: str, weights: dict[str, float]) -> ChapterChoice:
    """The chosen chapter described in *unboosted* weights.

    The choice can come off the ranking view; the numbers that go into
    confidence, into the plan and onto the console must not, or a recording
    placed against the coverage grain would report the comfortable margin that
    the boost manufactured. When the opening overrules a near-tie this margin
    comes out negative, and that is the honest reading: the announcement won,
    the window count did not agree, and a human should look.
    """
    runner_up_id = max(
        (i for i in weights if i != chapter_id), key=lambda i: (weights[i], i), default=None
    )
    return ChapterChoice(
        chapter_id=chapter_id,
        weight=weights.get(chapter_id, 0.0),
        runner_up_id=runner_up_id,
        runner_up_weight=weights.get(runner_up_id, 0.0) if runner_up_id else 0.0,
    )


class Classifier(Protocol):
    def classify(
        self, recording: Recording, transcript: Transcript, candidates: CandidateSet
    ) -> Classification: ...


# ── Keyword-only baseline ─────────────────────────────────────────────────────


class KeywordClassifier:
    """Retrieval only, no LLM. Chapter first, then the subtopics inside it.

    The order matters and is the whole point. Picking the best subtopic and
    then taking its chapter lets one strong subtopic drag a recording into a
    chapter the lecture barely visited; weighing chapters first — each one the
    sum of its subtopics' coverage — asks the question the folder tree
    actually poses. See `classify.chapters`.

    Confidence comes from **share** — dominance among the windows that matched
    any topic at all — the margin over the runner-up subtopic, and now the
    margin between the top two chapters. Not from raw scores (per-window
    normalisation always puts some node at 1.0, so score margins are small and
    noisy) and not from coverage (a lecture padded with admin chatter and
    homework notices would look uncertain even when its subject matter is
    obvious).

    It is deliberately conservative and caps below the default review
    threshold on genuinely split lectures. That is correct for a baseline with
    no semantic understanding: routing an ambiguous lecture to a human is the
    right answer, and pretending to certainty would be worse than useless.
    """

    name = "keyword"
    #: Never claim more than this. Keyword matching cannot distinguish "taught
    #: this topic" from "mentioned this topic a lot", so an unqualified high
    #: confidence is not something this classifier can honestly produce.
    MAX_CONFIDENCE = 0.92
    #: A chapter margin at or above this is treated as settled. Below it the
    #: placement itself is in doubt, and no amount of certainty about *which
    #: subtopic* can repair that: a recording in the wrong folder is wrong
    #: however well it is named.
    DECISIVE_CHAPTER_MARGIN = 0.35
    #: How far an undecided chapter is allowed to pull confidence down. A
    #: dead heat between two chapters costs a third of the score, which is
    #: enough to push an otherwise-clean lecture under the review threshold.
    CHAPTER_DOUBT_WEIGHT = 0.35

    #: How far the announcement in the opening may lift a node in the ranking,
    #: as a fraction of that node's own coverage. See `classify.intent`.
    #:
    #: Swept over the seven August recordings in steps of 0.005 to 0.4 and 0.05
    #: to 10, and bounded above by this class's own definition of a decisive
    #: chapter. Three numbers came out of that and this is the fourth.
    #:
    #: *0.3333 — below this, 19 August stays misfiled.* That lecture teaches
    #: the quadratic formula and is the recording this exists for. Its windows
    #: split 4-4 between "Solving Equations" (Algebra and Graphs) and
    #: "Estimation & Rounding" (Number), and Number carries two more
    #: single-window subtopics to Algebra's one, so the folder went to Number
    #: 55% / 45% on a subtopic tie. The opening announces math.2_5 — and also
    #: math.1_3 Powers & Roots, off "square root of b square", which sits in
    #: Number and is lifted along with it. Chapter sums become (5 + 4b)/13 for
    #: Algebra against (6 + b)/13 for Number, so the placement turns over at
    #: exactly b > 1/3; measured, 0.3333 keeps Number and 0.3334 takes Algebra.
    #:
    #: *1.0769 — above this the boost could overturn a chapter this classifier
    #: already calls settled.* DECISIVE_CHAPTER_MARGIN says a 0.35 lead is not
    #: in doubt. The tightest weight ratio consistent with that lead is
    #: 0.675 : 0.325, so a lift of (1 + b) can close it once 1 + b > 2.0769.
    #: `test_intent` holds this relationship, so moving either constant without
    #: the other fails rather than quietly widening the boost.
    #:
    #: *7.05 — where the seven recordings themselves break.* 12 August opens
    #: "which I have told you about the surds" and then teaches indices for 53%
    #: of the lecture; a lift that large renames it "Surds + Indices I". Far
    #: outside the bound above, but it is the empirical confirmation that the
    #: derived cap is the binding one and not the loose one.
    #:
    #: 0.664 is the geometric midpoint of 1.3333 and 2.0769 as multipliers —
    #: the furthest point, in the ratio that actually does the work, from both
    #: the result that needs the boost and the point at which the boost would
    #: start doing damage. Every one of the seven is stable across that whole
    #: interval, so the value inside it is chosen for margin, not for fit.
    #:
    #: None of it can touch the 10 August trap, which is the point of putting
    #: the boost in the ranking rather than in the coverage: that opening
    #: announces the *revision* half ("we intend to revise ... is indices")
    #: while Surds is the new material, and no value of this constant drops
    #: Surds from the name, because the boost never reaches the coverage the
    #: secondary-subtopic rule reads.
    INTENT_BOOST = 0.664

    def __init__(
        self,
        taxonomy: Taxonomy,
        settings: ClassifySettings | None = None,
        naming: NamingSettings | None = None,
        intent: IntentSettings | None = None,
    ) -> None:
        self.taxonomy = taxonomy
        self.settings = settings or ClassifySettings()
        self.naming = naming or NamingSettings()
        self.intent = IntentReader(taxonomy, intent)

    def classify(
        self, recording: Recording, transcript: Transcript, candidates: CandidateSet
    ) -> Classification:
        signal = self.intent.read(transcript)
        weights = chapter_weights(candidates, self.taxonomy)

        # A course-orientation session, not a lesson. Both of the intent
        # module's conditions have already been met — the opening is course
        # administration and names no syllabus topic — and this adds the third,
        # which is the one that keeps the opening subordinate to the whole
        # transcript: the rest of the lecture must fail to contradict it. A
        # lecture that spends five minutes on exam logistics and then teaches
        # for fifty has coverage well over the floor and is untouched here.
        top_coverage = max((c.coverage for c in candidates.candidates), default=0.0)
        if signal.looks_administrative and top_coverage < self.settings.coverage_floor:
            return Classification(
                abstain=True,
                abstain_reason=(
                    f"course administration, not a lesson: {signal.describe()}, and no "
                    f"topic reaches the coverage floor (best {top_coverage:.0%})"
                ),
                method=self.name,
                chapter_weights=weights,
            )

        # The announcement decides the ranking; coverage decides everything
        # reported. See `intent_ranked`.
        ranked = intent_ranked(candidates, signal, self.INTENT_BOOST)
        chosen = select_chapter(ranked, self.taxonomy)
        if chosen is None:
            return Classification(
                abstain=True,
                abstain_reason="no syllabus node matched any part of the transcript",
                method=self.name,
            )
        chapter = _restate(chosen.chapter_id, weights)

        picked = select_subtopics(
            candidates,
            self.taxonomy,
            chapter.chapter_id,
            self.settings,
            max_subtopics=KEYWORD_SUBTOPIC_CAP,
        )
        if not picked:
            return Classification(
                abstain=True,
                abstain_reason=(
                    f"chapter {chapter.chapter_id!r} won the weighting but has no "
                    "namable subtopic among the candidates"
                ),
                method=self.name,
            )
        picked = self._promote_announced(picked, ranked, candidates, chapter.chapter_id)

        best = picked[0]
        rivals = [c for c in candidates.candidates if c.node.id != best.node.id and c.coverage > 0]
        subtopic_margin = best.share - (rivals[0].share if rivals else 0.0)
        raw = 0.35 + best.share * 0.4 + subtopic_margin * 0.5
        # A near-tie between chapters is genuine uncertainty even when one
        # subtopic dominates inside the winner, so it scales the whole score
        # rather than being subtracted from it.
        # Clamped at zero because the margin can now be negative: an
        # intent-boosted placement is measured against the chapter that won on
        # coverage, and losing that comparison is exactly the case that should
        # bottom out the certainty rather than push confidence below zero,
        # which TopicMatch rejects outright.
        chapter_certainty = min(1.0, max(0.0, chapter.margin) / self.DECISIVE_CHAPTER_MARGIN)
        confidence = min(
            self.MAX_CONFIDENCE,
            raw * (1 - self.CHAPTER_DOUBT_WEIGHT + self.CHAPTER_DOUBT_WEIGHT * chapter_certainty),
        )

        evidence = [
            Evidence(quote=candidates.windows[i].text[:180], start_s=candidates.windows[i].start_s)
            for i in best.top_windows[:2]
            if i < len(candidates.windows)
        ]

        # The second pick leads the secondary list so `assign_chapter_and_topics`
        # reaches the same conclusion from the Classification alone; the other
        # candidates follow as reporting, not as naming input.
        picked_ids = {c.node.id for c in picked}
        secondary = [
            TopicMatch(node_id=c.node.id, confidence=0.0, coverage=c.coverage) for c in picked[1:]
        ]
        secondary += [
            TopicMatch(node_id=c.node.id, confidence=0.0, coverage=c.coverage)
            for c in candidates.top(4)
            if c.node.id not in picked_ids and c.coverage >= self.settings.secondary_coverage_min
        ]

        chapter_node = self.taxonomy.get(chapter.chapter_id)
        chapter_title = chapter_node.title if chapter_node else chapter.chapter_id
        runner_up = self.taxonomy.get(chapter.runner_up_id) if chapter.runner_up_id else None
        # "ahead of" and "behind" as separate branches, because the margin is
        # signed now: a chapter the announcement pulled past the coverage
        # leader trails it, and "-9% ahead" is a sentence that reads as a typo
        # in precisely the case a reviewer most needs to understand.
        if runner_up is None:
            against = " with no rival chapter. "
        elif chapter.margin >= 0:
            against = f", {chapter.margin:.0%} ahead of {runner_up.title!r}. "
        else:
            against = (
                f", {-chapter.margin:.0%} behind {runner_up.title!r} on coverage but "
                f"announced at the top of the lecture. "
            )
        reasoning = (
            f"Chapter {chapter_title!r} carries {chapter.weight:.0%} of the classifiable "
            f"content"
            + against
            + f"Inside it, {best.node.label!r} took {best.coverage:.0%} of the lecture "
            f"and {best.share:.0%} of its classifiable content"
            + (f", alongside {picked[1].node.label!r}" if len(picked) > 1 else "")
            + (f"; {signal.describe()}." if signal.matches else ".")
        )

        classification = Classification(
            primary=TopicMatch(
                node_id=best.node.id,
                confidence=confidence,
                coverage=best.coverage,
                evidence=evidence,
            ),
            secondary=secondary,
            reasoning=reasoning,
            method=self.name,
        )
        return assign_chapter_and_topics(
            classification,
            self.taxonomy,
            weights,
            self.settings,
            self.naming,
            max_subtopics=KEYWORD_SUBTOPIC_CAP,
        )

    def _promote_announced(
        self,
        picked: list[Candidate],
        ranked: CandidateSet,
        candidates: CandidateSet,
        chapter_id: str,
    ) -> list[Candidate]:
        """Let the announcement choose the primary subtopic, not the second one.

        `picked` came off true coverage. If the ranking view names a different
        subtopic first, that one leads — but the slot behind it is refilled
        from true coverage through the same `secondary_qualifies` the unboosted
        path uses, so the boost decides *which* topic the file is named for and
        never how much of the lecture a topic has to be worth to be named at
        all. Keeping those two apart is what leaves 10 August as "Indices I +
        Surds": its opening announces indices and only indices, and a boost
        that reached the coverage on both sides of that comparison would push
        Surds under the 45% bar and out of the filename.

        Returns `picked` unchanged whenever the two agree, which is every one
        of the seven August recordings — the placement is what the opening
        moves on real data, not the name inside it.
        """
        promoted = select_subtopics(
            ranked, self.taxonomy, chapter_id, self.settings, max_subtopics=1
        )
        if not promoted or promoted[0].node.id == picked[0].node.id:
            return picked
        primary = next((c for c in candidates.candidates if c.node.id == promoted[0].node.id), None)
        if primary is None:  # pragma: no cover - the view holds the same nodes
            return picked
        rest = [c for c in picked if c.node.id != primary.node.id][:1]
        return [primary] + [
            c for c in rest if secondary_qualifies(primary.coverage, c.coverage, self.settings)
        ]


# ── LLM classifier ────────────────────────────────────────────────────────────


class LLMClassifier:
    """Retrieval narrows the field; Claude makes the call and justifies it."""

    name = "hybrid+llm"

    def __init__(
        self,
        taxonomy: Taxonomy,
        settings: ClassifySettings,
        api_key: str | None = None,
        naming: NamingSettings | None = None,
    ) -> None:
        self.taxonomy = taxonomy
        self.settings = settings
        self.naming = naming or NamingSettings()
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic

            # A bare constructor also resolves an `ant auth login` profile, so
            # do not force an explicit key.
            self._client = (
                anthropic.Anthropic(api_key=self.api_key)
                if self.api_key
                else anthropic.Anthropic()
            )
        return self._client

    def classify(
        self, recording: Recording, transcript: Transcript, candidates: CandidateSet
    ) -> Classification:
        top = candidates.top(self.settings.top_k_candidates)
        if not top:
            return Classification(
                abstain=True,
                abstain_reason="retrieval returned no candidates",
                method=self.name,
            )

        response = self._get_client().messages.create(
            model=self.settings.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            thinking={"type": "adaptive"},
            output_config={
                "effort": self.settings.effort,
                "format": {"type": "json_schema", "schema": response_schema()},
            },
            messages=[{"role": "user", "content": self._build_prompt(recording, candidates, top)}],
        )

        if response.stop_reason == "refusal":
            log.warning("classifier refused: %s", getattr(response, "stop_details", None))
            return Classification(
                abstain=True,
                abstain_reason="model declined to answer",
                method=self.name,
                model=self.settings.model,
                chapter_weights=chapter_weights(candidates, self.taxonomy),
            )

        # output_config.format guarantees the first text block is valid JSON
        # matching the schema.
        text = next(b.text for b in response.content if b.type == "text")
        parsed = ClassificationOut.model_validate(json.loads(text))
        return self._to_domain(parsed, candidates)

    # ── Prompt ───────────────────────────────────────────────────────────────

    def _build_prompt(self, recording: Recording, candidates: CandidateSet, top) -> str:
        parts: list[str] = []

        parts.append("## Candidate syllabus nodes\n")
        for candidate in top:
            node = candidate.node
            parts.append(f"### {node.id}")
            parts.append(f"Title path: {node.title_path}")
            chapter_id = chapter_of(self.taxonomy, node.id)
            if chapter_id and chapter_id != node.id:
                parts.append(f"Chapter (the folder this would file into): {chapter_id}")
            if node.short_title:
                parts.append(f"Filename label: {node.short_title}")
            if node.keywords:
                parts.append(f"Keywords: {', '.join(node.keywords[:20])}")
            if node.aliases:
                parts.append(f"Also called: {', '.join(node.aliases)}")
            for outcome in node.learning_outcomes[:6]:
                parts.append(f"  - [{outcome.code}] {outcome.text}")
            parts.append(
                f"Retrieval: coverage {candidate.coverage:.0%}, "
                f"mean score {candidate.score:.3f}, peak {candidate.peak_score:.3f}"
            )
            parts.append("")

        # Chapters, not just nodes: rule 4 is decided at this level, and the
        # model cannot weigh chapters it was never shown.
        weights = chapter_weights(candidates, self.taxonomy)
        if weights:
            parts.append("## Chapter weighting (sum of each chapter's subtopic coverage)\n")
            for chapter_id, weight in sorted(weights.items(), key=lambda kv: -kv[1]):
                node = self.taxonomy.get(chapter_id)
                parts.append(f"- {chapter_id} ({node.title if node else '?'}): {weight:.0%}")
            parts.append("")
            parts.append(
                "The recording goes in exactly one of these. Pick the heaviest unless "
                "the transcript says otherwise, and say so if it does."
            )
            parts.append("")

        parts.append("## Transcript by window\n")
        for window in candidates.windows:
            minutes = int(window.start_s // 60)
            seconds = int(window.start_s % 60)
            # Trim each window; the full transcript is neither needed nor
            # affordable, and the decisive content is early in each window.
            excerpt = window.text[:1200]
            parts.append(f"[{minutes:02d}:{seconds:02d}] {excerpt}")
        parts.append("")

        parts.append("## Weak signals (transcript evidence overrides these)\n")
        parts.append(f"Recorded: {recording.recorded_at:%Y-%m-%d %H:%M}")
        if recording.meeting_topic:
            parts.append(f"Zoom meeting topic: {recording.meeting_topic}")
        if recording.teacher:
            parts.append(f"Teacher: {recording.teacher}")
        parts.append(f"Original filename: {recording.original_filename}")
        parts.append("")
        parts.append(
            "Choose the one chapter this lecture belongs to and the one or two "
            "subtopics inside it that should name the file, or abstain. Cite "
            "timestamped quotes from the windows above."
        )
        return "\n".join(parts)

    def _to_domain(self, out: ClassificationOut, candidates: CandidateSet) -> Classification:
        classification = Classification(
            primary=TopicMatch(
                node_id=out.primary.node_id,
                confidence=max(0.0, min(1.0, out.primary.confidence)),
                coverage=max(0.0, min(1.0, out.primary.coverage)),
                evidence=[
                    Evidence(quote=e.quote, start_s=e.start_s) for e in out.primary.evidence
                ],
            )
            if not out.abstain
            else None,
            secondary=[
                TopicMatch(node_id=s.node_id, confidence=0.0, coverage=s.coverage)
                for s in out.secondary
            ],
            is_continuation=out.is_continuation,
            # Not carried through an abstain. `schema._strictify` marks every
            # property required, so the model must emit a label even on the
            # reply that says it could not place the lecture — and that label
            # is what naming reads first once there are no topic node ids.
            suggested_topic_label=None if out.abstain else (out.suggested_topic_label or None),
            reasoning=out.reasoning,
            abstain=out.abstain,
            abstain_reason=out.abstain_reason or None,
            method=self.name,
            model=self.settings.model,
        )
        # The prompt states the one-chapter and two-subtopic rules; this is
        # what enforces them. A reply that named a third subtopic still yields
        # a usable classification, just not a three-part filename.
        return assign_chapter_and_topics(
            classification,
            self.taxonomy,
            chapter_weights(candidates, self.taxonomy),
            self.settings,
            self.naming,
        )


# ── Choosing one ──────────────────────────────────────────────────────────────


def build_classifier(
    taxonomy: Taxonomy,
    settings: ClassifySettings,
    naming: NamingSettings,
    *,
    provider: str,
    api_key: str | None = None,
    allowed_ids: set[str] | None = None,
) -> Classifier:
    """The one place a provider name turns into a classifier.

    Shared by `ClassificationService` and by `scripts/rename_folder.py` so the
    unattended pipeline and the folder tool cannot drift into classifying the
    same recording two different ways.
    """
    if provider == "keyword":
        return KeywordClassifier(taxonomy, settings, naming)
    if provider == "anthropic":
        return LLMClassifier(taxonomy, settings, api_key, naming)
    if provider == "openai":
        # Imported here rather than at module scope: the openai package is an
        # optional extra, and the keyword path has to keep working without it.
        from pipeline.classify.openai_classifier import OpenAIClassifier

        return OpenAIClassifier(
            taxonomy, settings, naming, api_key=api_key, allowed_ids=allowed_ids
        )
    raise ValueError(
        f"unknown classifier provider {provider!r}; expected 'openai', 'anthropic' or 'keyword'"
    )


# ── Service ───────────────────────────────────────────────────────────────────


class ClassificationService:
    """Retrieval + classifier + the confidence gate.

    Which classifier comes from `ClassifySettings.provider`, defaulting to
    ChatGPT. Retrieval still runs on every path — the OpenAI classifier ignores
    the shortlist and reads the whole syllabus, but the candidate set is what
    the console line, `rename_plan.json` and the sidecar report as the keyword
    evidence, and being able to see the two disagree is worth the BM25 pass.
    """

    def __init__(
        self,
        taxonomy: Taxonomy,
        settings: ClassifySettings,
        retrieval: RetrievalSettings,
        *,
        use_llm: bool = True,
        api_key: str | None = None,
        naming: NamingSettings | None = None,
        allowed_ids: set[str] | None = None,
    ) -> None:
        self.taxonomy = taxonomy
        self.settings = settings
        self.retriever = HybridRetriever(taxonomy, settings, retrieval)
        self.naming = naming or NamingSettings()
        self.classifier = build_classifier(
            taxonomy,
            settings,
            self.naming,
            # `use_llm=False` still means "no network", whatever the provider
            # says. The dry run and the tests pass it, and a default that
            # reached for an API key would break both on a fresh clone.
            provider=settings.provider if use_llm else "keyword",
            api_key=api_key,
            allowed_ids=allowed_ids,
        )

    def run(
        self, recording: Recording, transcript: Transcript
    ) -> tuple[Classification, CandidateSet]:
        candidates = self.retriever.retrieve(transcript)
        classification = self.classifier.classify(recording, transcript, candidates)

        # Guard against a hallucinated node id. Structured output constrains
        # the shape, not the values, so this check is still necessary.
        if classification.primary and self.taxonomy.get(classification.primary.node_id) is None:
            log.error(
                "classifier returned unknown node %r; routing to review",
                classification.primary.node_id,
            )
            classification.abstain = True
            classification.abstain_reason = (
                f"classifier returned unknown node id {classification.primary.node_id!r}"
            )
            classification.primary = None
            classification.chapter_node_id = None
            classification.topic_node_ids = []
            # The label goes too. Naming falls back to it as soon as
            # `topic_node_ids` is empty, so leaving it set would name the file
            # after the node this branch just rejected as non-existent — the
            # review marker on the end being the only sign anything was wrong.
            classification.suggested_topic_label = None

        return classification, candidates

    def needs_review(self, classification: Classification) -> bool:
        return classification.needs_review(
            self.settings.review_threshold, self.settings.coverage_floor
        )
