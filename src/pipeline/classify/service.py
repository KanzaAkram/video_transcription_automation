"""Turning a transcript into a syllabus location.

Two implementations behind one interface:

  * ``LLMClassifier``     retrieval + Claude. The production path.
  * ``KeywordClassifier`` retrieval only, top candidate wins. Used by the dry
    run so the pipeline is demonstrable with no API key, and useful as a
    baseline to measure the LLM against.

`ClassificationService` picks between them and applies the confidence gate.
"""

from __future__ import annotations

import json
import logging
from typing import Protocol

from pipeline.classify.retriever import CandidateSet, HybridRetriever
from pipeline.classify.schema import ClassificationOut, response_schema
from pipeline.config import ClassifySettings, RetrievalSettings
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

Rules:

- Choose the node the lecture predominantly covers, not every node it touches. \
A lecture that spends five minutes recapping the previous chapter and fifty \
minutes on a new one belongs to the new one.
- Cite at least two verbatim quotes with their timestamps. Quotes must appear \
in the transcript you were given. This is what makes the decision auditable.
- Report any other node with meaningful coverage as secondary.
- Set is_continuation when the lecture opens mid-topic and continues the \
previous lecture rather than starting something new.
- suggested_topic_label goes in the filename. Prefer the specific thing taught \
("Relative Atomic Mass") over the broad chapter ("Atomic Structure") when the \
lecture is clearly about the narrower thing.
- Abstain if none of the candidates fits — a revision session, an exam \
debrief, or a topic absent from the syllabus. Abstaining sends the lecture to \
a human, which is the correct outcome. Do not force a match: misfiling is \
worse than not filing, because it is invisible.
- Confidence should reflect genuine uncertainty. Two plausible candidates \
means a confidence below 0.7, not a coin flip reported as 0.95.
"""


class Classifier(Protocol):
    def classify(
        self, recording: Recording, transcript: Transcript, candidates: CandidateSet
    ) -> Classification: ...


# ── Keyword-only baseline ─────────────────────────────────────────────────────


class KeywordClassifier:
    """Top retrieval candidate wins. No LLM.

    Confidence comes from **share** — dominance among the windows that matched
    any topic at all — and the margin over the runner-up. Not from raw scores
    (per-window normalisation always puts some node at 1.0, so score margins
    are small and noisy) and not from coverage (a lecture padded with admin
    chatter and homework notices would look uncertain even when its subject
    matter is obvious).

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

    def __init__(self, taxonomy: Taxonomy) -> None:
        self.taxonomy = taxonomy

    def classify(
        self, recording: Recording, transcript: Transcript, candidates: CandidateSet
    ) -> Classification:
        top = candidates.top(3)
        if not top or top[0].coverage == 0:
            return Classification(
                abstain=True,
                abstain_reason="no syllabus node matched any part of the transcript",
                method=self.name,
            )

        best = top[0]
        runner_up_share = top[1].share if len(top) > 1 else 0.0
        margin = best.share - runner_up_share
        confidence = min(self.MAX_CONFIDENCE, 0.35 + best.share * 0.4 + margin * 0.5)

        evidence = [
            Evidence(quote=candidates.windows[i].text[:180], start_s=candidates.windows[i].start_s)
            for i in best.top_windows[:2]
            if i < len(candidates.windows)
        ]

        return Classification(
            primary=TopicMatch(
                node_id=best.node.id,
                confidence=confidence,
                coverage=best.coverage,
                evidence=evidence,
            ),
            secondary=[
                TopicMatch(node_id=c.node.id, confidence=0.0, coverage=c.coverage)
                for c in top[1:]
                if c.coverage >= 0.15
            ],
            suggested_topic_label=best.node.title,
            reasoning=(
                f"Retrieval put {best.node.title!r} top: {best.coverage:.0%} of the "
                f"lecture, {best.share:.0%} of its classifiable content, "
                f"{margin:.0%} ahead of {top[1].node.title!r}."
                if len(top) > 1
                else f"Retrieval put {best.node.title!r} top with "
                f"{best.coverage:.0%} coverage and no rival."
            ),
            method=self.name,
        )


# ── LLM classifier ────────────────────────────────────────────────────────────


class LLMClassifier:
    """Retrieval narrows the field; Claude makes the call and justifies it."""

    name = "hybrid+llm"

    def __init__(
        self, taxonomy: Taxonomy, settings: ClassifySettings, api_key: str | None = None
    ) -> None:
        self.taxonomy = taxonomy
        self.settings = settings
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
            )

        # output_config.format guarantees the first text block is valid JSON
        # matching the schema.
        text = next(b.text for b in response.content if b.type == "text")
        parsed = ClassificationOut.model_validate(json.loads(text))
        return self._to_domain(parsed)

    # ── Prompt ───────────────────────────────────────────────────────────────

    def _build_prompt(self, recording: Recording, candidates: CandidateSet, top) -> str:
        parts: list[str] = []

        parts.append("## Candidate syllabus nodes\n")
        for candidate in top:
            node = candidate.node
            parts.append(f"### {node.id}")
            parts.append(f"Title path: {node.title_path}")
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
            "Choose the syllabus node this lecture belongs to, or abstain. "
            "Cite timestamped quotes from the windows above."
        )
        return "\n".join(parts)

    def _to_domain(self, out: ClassificationOut) -> Classification:
        return Classification(
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
            suggested_topic_label=out.suggested_topic_label or None,
            reasoning=out.reasoning,
            abstain=out.abstain,
            abstain_reason=out.abstain_reason or None,
            method=self.name,
            model=self.settings.model,
        )


# ── Service ───────────────────────────────────────────────────────────────────


class ClassificationService:
    """Retrieval + classifier + the confidence gate."""

    def __init__(
        self,
        taxonomy: Taxonomy,
        settings: ClassifySettings,
        retrieval: RetrievalSettings,
        *,
        use_llm: bool = True,
        api_key: str | None = None,
    ) -> None:
        self.taxonomy = taxonomy
        self.settings = settings
        self.retriever = HybridRetriever(taxonomy, settings, retrieval)
        self.classifier: Classifier = (
            LLMClassifier(taxonomy, settings, api_key)
            if use_llm
            else KeywordClassifier(taxonomy)
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

        return classification, candidates

    def needs_review(self, classification: Classification) -> bool:
        return classification.needs_review(
            self.settings.review_threshold, self.settings.coverage_floor
        )
