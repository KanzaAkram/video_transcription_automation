"""Choosing one chapter, then the subtopics inside it — the keyword path.

Two decisions that look like one and are not.

**Which chapter** is a placement decision. A chapter here is a *Drive folder*,
not a syllabus topic — the two do not correspond. Cambridge 9701 topic 1,
"Atomic structure", is split across two of the user's folders: "1. Atomic
Structure (Chapter 1)" holds 1.1 and 1.2, "2. Electrons in Atoms (Chapter 2)"
holds 1.3 and 1.4. So the tree is chapter (depth 0) over subtopic (depth 1),
and a recording belongs to exactly one chapter. It is one lesson; it goes in
one folder. Splitting it across two would double the archive without adding a
single minute of teaching to it.

**Which subtopics** is a naming decision, capped at `MAX_SUBTOPICS_IN_NAME`.
Most lectures cover exactly one subtopic and are named for it; a lecture that
genuinely straddles two or three gets each of them. `select_subtopics` here
only ever returns one or two, and that is a statement about keyword coverage
rather than about filenames: coverage can rank subtopics but cannot tell
"taught this" from "said this word", so a third name chosen that way is a
guess wearing the same clothes as a fact. The ChatGPT path
(`classify.openai_classifier`) reads the lecture and can honestly return
three.

Everything here is a pure function over the CandidateSet retrieval already
produced — no I/O, no model calls — so the decisions are testable in isolation
and reproducible from a stored candidate set.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.classify.retriever import Candidate, CandidateSet
from pipeline.config import ClassifySettings
from pipeline.models import MAX_SUBTOPICS_IN_NAME
from pipeline.syllabus import Taxonomy

#: The rule stated where it cannot be configured away. `NamingSettings`
#: enforces the same bound on input; this one bounds the output. One number,
#: imported rather than repeated, so raising the cap cannot half-happen.
HARD_SUBTOPIC_CAP = MAX_SUBTOPICS_IN_NAME


@dataclass(frozen=True)
class ChapterChoice:
    """Which Drive folder a recording belongs to, and how close the call was.

    The runner-up is carried rather than discarded because it is the only
    thing that distinguishes a confident placement from a coin flip, and the
    two are indistinguishable in the winner alone.
    """

    chapter_id: str
    weight: float
    runner_up_id: str | None
    runner_up_weight: float

    @property
    def margin(self) -> float:
        return self.weight - self.runner_up_weight


def chapter_of(taxonomy: Taxonomy, node_id: str) -> str | None:
    """The depth-0 ancestor of `node_id`, or the node itself if it is depth 0.

    Returns None for a node the taxonomy does not know, so a hallucinated id
    from a model reply cannot silently invent a chapter.
    """
    node = taxonomy.get(node_id)
    if node is None:
        return None
    ancestors = taxonomy.ancestors(node_id)
    return ancestors[0].id if ancestors else node.id


def chapter_weights(candidates: CandidateSet, taxonomy: Taxonomy) -> dict[str, float]:
    """How much of the lecture each chapter accounts for, summing to 1.

    Every candidate's coverage is attributed to its chapter and the totals are
    normalised. A chapter's weight is therefore the sum of its subtopics'
    coverage: four subtopics at 15% each beat one at 50%, which is the right
    answer when the lecture worked through most of a chapter.

    Normalising discards the windows that matched nothing — admin, silence,
    homework notices. That is deliberate: those windows say nothing about
    *which* chapter, and leaving them in would make every chapter look weak on
    a lecture that happened to start late.
    """
    totals: dict[str, float] = {}
    for candidate in candidates.candidates:
        if candidate.coverage <= 0:
            continue
        chapter_id = chapter_of(taxonomy, candidate.node.id) or candidate.node.id
        totals[chapter_id] = totals.get(chapter_id, 0.0) + candidate.coverage
    grand_total = sum(totals.values())
    if grand_total <= 0:
        return {}
    return {chapter_id: value / grand_total for chapter_id, value in totals.items()}


def select_chapter(candidates: CandidateSet, taxonomy: Taxonomy) -> ChapterChoice | None:
    """Pick the one chapter this recording belongs to.

    The tie-break is fully ordered — weight, then the best single subtopic in
    the chapter, then the node id — because a coin flip here misfiles a
    lecture, and a misfiled lecture is invisible in a way an unfiled one is
    not. The second key is the one that matters in practice: two chapters at
    identical weight, one carried by a subtopic the lecture clearly taught and
    the other by three it brushed past, are not equally good answers.
    """
    weights = chapter_weights(candidates, taxonomy)
    if not weights:
        return None

    best_single: dict[str, float] = {}
    for candidate in candidates.candidates:
        if candidate.coverage <= 0:
            continue
        chapter_id = chapter_of(taxonomy, candidate.node.id) or candidate.node.id
        best_single[chapter_id] = max(best_single.get(chapter_id, 0.0), candidate.coverage)

    ranked = sorted(
        weights,
        key=lambda chapter_id: (
            -weights[chapter_id],
            -best_single.get(chapter_id, 0.0),
            chapter_id,
        ),
    )
    winner = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    return ChapterChoice(
        chapter_id=winner,
        weight=weights[winner],
        runner_up_id=runner_up,
        runner_up_weight=weights[runner_up] if runner_up else 0.0,
    )


def secondary_qualifies(
    primary_coverage: float, candidate_coverage: float, settings: ClassifySettings
) -> bool:
    """Whether a second subtopic has earned a place in the filename.

    Both bars, because they fail differently. The absolute one rejects a
    subtopic that took four minutes of ninety however lopsided the lecture
    was; the relative one rejects a subtopic that looks substantial only
    because the winner is weak too.
    """
    if candidate_coverage < settings.secondary_subtopic_min_coverage:
        return False
    return candidate_coverage >= settings.secondary_subtopic_ratio * primary_coverage


def select_subtopics(
    candidates: CandidateSet,
    taxonomy: Taxonomy,
    chapter_id: str,
    settings: ClassifySettings,
    *,
    max_subtopics: int = HARD_SUBTOPIC_CAP,
) -> list[Candidate]:
    """The one or two subtopics of `chapter_id` that name the file.

    Two, never three, whatever `max_subtopics` allows: only the runner-up is
    ever considered. Keyword coverage ranks subtopics, and a ranking is not
    evidence that a third one was taught — see the module docstring.

    The thresholds live on ClassifySettings and the cap on NamingSettings, so
    the cap arrives as an argument rather than off `settings`: this module has
    no business reading naming configuration, and the caller has both objects.

    Falls back to the chapter node itself when the chapter has no subtopics
    among the candidates. Some chapters have no subtopic layer at all, and a
    chapter that won the weighting has to be namable by something.
    """
    cap = min(max_subtopics, HARD_SUBTOPIC_CAP)

    inside = [
        candidate
        for candidate in candidates.candidates
        if candidate.coverage > 0
        and (chapter_of(taxonomy, candidate.node.id) or candidate.node.id) == chapter_id
    ]
    subtopics = [c for c in inside if c.node.id != chapter_id] or inside
    if not subtopics:
        return []

    subtopics.sort(key=lambda c: (-c.coverage, -c.score, c.node.id))
    chosen = [subtopics[0]]
    # Only the runner-up is ever considered. The list is coverage-ordered, so
    # nothing below it can clear a bar it failed, and nothing below it could
    # displace it if it passed.
    runner_up = subtopics[1] if len(subtopics) > 1 else None
    if (
        cap > 1
        and runner_up is not None
        and secondary_qualifies(chosen[0].coverage, runner_up.coverage, settings)
    ):
        chosen.append(runner_up)

    assert len(chosen) <= HARD_SUBTOPIC_CAP, (
        f"rule violated: {len(chosen)} subtopics selected for {chapter_id!r}"
    )
    return chosen
