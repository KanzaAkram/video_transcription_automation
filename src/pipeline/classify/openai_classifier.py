"""Classification by ChatGPT: the whole lecture read against the whole syllabus.

The keyword path (`classify.service.KeywordClassifier`) decides from vocabulary
overlap — which syllabus terms appear in which four-minute window, and how
often. That is cheap and auditable and it is wrong in a specific, recurring
way: a teacher who spends fifty minutes *explaining* glycolysis without saying
"glycolysis" loses to a chapter whose keywords happen to be said out loud, and
a lecture that opens on a recap gets named after the recap. Neither failure is
visible in the filename it produces.

So this classifier is not "retrieval plus a model to pick the winner". It is
the model reading the lecture. Two consequences shape everything below:

  * **The candidate list is the syllabus, not a retrieval shortlist.** Filtering
    the options by keyword score before the model sees them would reintroduce
    the exact failure being replaced — the right answer would already have been
    dropped. Chapters are one folder each and a subject is at most a few
    hundred nodes, so all of them fit in the prompt with room to spare.

  * **The transcript goes in whole.** Not window digests, not the first five
    minutes. What a lecture is about is a fact about all of it.

What comes back is constrained twice over. The reply schema declares the legal
node ids as a JSON Schema ``enum``, so a strict structured output *cannot*
contain an id the syllabus does not have; and `_to_domain` re-checks every id
against the taxonomy anyway, because a schema constrains the reply and not the
world. The filename is then rendered from the syllabus' own titles for those
ids — the model never writes a label. That is what "no inventions" means here:
not that the model is asked nicely, but that there is no path from a made-up
name to a file on disk.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from pipeline.classify.chapters import chapter_of
from pipeline.classify.retriever import CandidateSet
from pipeline.config import ClassifySettings, NamingSettings
from pipeline.models import Classification, Evidence, Recording, TopicMatch, Transcript
from pipeline.syllabus import SyllabusNode, Taxonomy

log = logging.getLogger(__name__)

#: What the model says instead of naming a chapter. The schema pins `chapter_id`
#: to an enum of real chapters, so abstaining needs a legal value to be spelled
#: with — an empty string would have to be legal everywhere, and then every
#: reply could quietly carry one.
NO_CHAPTER = "NONE"

#: Above this many options, or this many characters of them, ids are sent as a
#: plain string rather than an enum. Structured outputs cap enum size, and a
#: request rejected for a schema that is too large fails the whole run; the id
#: check in `_to_domain` is the one that actually holds, so dropping the enum
#: degrades the guarantee from "impossible" to "caught", not to "absent".
MAX_ENUM_VALUES = 500
MAX_ENUM_CHARS = 7000

#: How much transcript to send. A full lecture is on the order of 50,000 to
#: 100,000 characters, so this is headroom rather than a limit anyone meets; it
#: exists so a pathological input — a Whisper loop repeating one sentence for
#: forty minutes, which is the failure `condition_on_previous_text: false`
#: exists to prevent — costs a warning instead of a 400.
TRANSCRIPT_CHAR_BUDGET = 240_000

#: One line of transcript per this many seconds. Long enough that timestamps do
#: not dominate the text, short enough that a quote can be cited to the minute.
TRANSCRIPT_LINE_SECONDS = 45


#: The rules the model is given. A template rather than a constant because two
#: of the numbers in it are `ClassifySettings` fields, and a prompt that states
#: a bar the configuration does not hold is worse than one that states no bar:
#: it reads as though the two agree. See `system_prompt`.
SYSTEM_PROMPT_TEMPLATE = """\
You file one recorded lecture into a syllabus.

You are given the complete syllabus for one subject and the complete \
transcript of one lecture. Decide which chapter the lecture belongs to, then \
which subtopics inside that chapter the teacher actually taught.

## Rules

1. **One chapter.** A recording is one lesson and goes in one folder. Never \
split it across two chapters. If the lecture recaps the previous chapter and \
then teaches a new one, it belongs to the new one.

2. **One to three subtopics, all inside that chapter**, ordered by how much of \
the lecture each took. Name every subtopic the teacher genuinely taught and no \
others. Most lectures teach one; a lecture that works through two or three \
gets two or three.

3. **Taught, not mentioned.** A subtopic named in passing, used as an example, \
or recapped from last lesson is not a subtopic this lecture covered. If you \
would not tell a student "this recording covers X", do not name X.

   Past the first, a subtopic earns its place only if **both** are true: it \
took at least {min_coverage:.0%} of the lecture, and at least \
{ratio:.0%} of what the first subtopic took. The two bars catch different \
mistakes — the first rejects a five-minute aside in a fifty-minute lesson, the \
second rejects a topic that only looks substantial next to a lesson with no \
real focus. A subtopic trailing the first four-to-one was mentioned, not \
taught.

   The test of whether you have cleared this bar is simple: **if you cannot \
quote the teacher teaching it, you may not name it.** Adding a name you have \
no evidence for makes the filename longer and less true at the same time, \
which is the worst trade available — the cost is paid on every glance at the \
folder, and the benefit is a topic no student would have looked for there.

4. **Use only the ids listed in the syllabus below.** Never invent an id, a \
chapter name or a subtopic name, and never merge two subtopics into one. The \
filename is built from the syllabus' own titles for the ids you return, so an \
id that is not in the list is a file nobody can find again.

5. **Judge from the whole lecture.** The first five minutes are usually a \
recap of the previous lesson, attendance, or homework, and the last few are \
often questions. What a lecture is about is what the middle of it is spent on.

6. `coverage` is the fraction of lecture *time* spent on that subtopic, \
between 0 and 1. The coverages you report should sum to at most 1 — the \
remainder is admin, recap and digression.

7. **Cite at least two verbatim quotes** with their timestamps, copied exactly \
from the transcript. This is what makes the decision checkable by a human.

8. **Abstain** — `chapter_id` "NONE", `subtopics` empty — when the lecture is \
not teaching anything in this syllabus: a revision or past-paper session, an \
exam debrief, or a topic the syllabus does not contain. Abstaining sends the \
recording to a human, which is the right outcome. Misfiling is worse than not \
filing, because it is invisible.

9. **`session_type`.** Almost every recording is a `lesson`. Set it to \
`orientation` for the class that is *about the course rather than about the \
subject*: the first session of the year, the plan for the term, which papers \
are sat and when, calculator and exam-board rules, what to bring, where the \
resources are. The signature is a teacher listing topics they are going to \
cover later instead of teaching one now.

   An orientation session also abstains — there is no subtopic in it — but the \
two are not the same thing and must not be spelled the same way. "I could not \
place this" leaves a recording called *Unclassified*; "this is the \
introduction class" names it for what it is. Do not reach for `orientation` \
just because a lecture was hard to place: a revision session and a past-paper \
walkthrough are lessons you are abstaining on, not orientation.

10. `confidence` is your genuine certainty that the chapter and the first \
subtopic are right, between 0 and 1 — or, for an orientation session, that it \
is one. Two plausible chapters is below 0.7. Do not report a coin flip as 0.95.

## About the audio

The transcript comes from Whisper and these lectures are taught in mixed Urdu \
and English, so technical terms are usually English inside Urdu sentences and \
are sometimes mis-transcribed phonetically. Read through the transcription \
errors. Judge from the subject matter being explained, not from whether the \
exact syllabus wording appears.
"""


def system_prompt(settings: ClassifySettings) -> str:
    """The rules, with the secondary-subtopic bars filled in from settings.

    The bars are the same two `secondary_qualifies` applies on the keyword
    path, and they are stated to the model rather than enforced after it on
    purpose. Its coverage figures are estimates; hard-filtering an estimate
    drops a subtopic the model was right about because it guessed 18% instead
    of 20%. Telling it the standard lets it apply the judgement it was hired
    for — and the first run without them named a third subtopic at 10% in three
    lectures out of five, with no quote behind any of them.
    """
    return SYSTEM_PROMPT_TEMPLATE.format(
        min_coverage=settings.secondary_subtopic_min_coverage,
        ratio=settings.secondary_subtopic_ratio,
    )


# ── Reply ─────────────────────────────────────────────────────────────────────


class _SubtopicOut(BaseModel):
    node_id: str
    coverage: float


class _EvidenceOut(BaseModel):
    quote: str
    start_s: float


#: `session_type` values. A lecture is about the subject; an orientation class
#: is about the course.
LESSON = "lesson"
ORIENTATION = "orientation"


class _ReplyOut(BaseModel):
    chapter_id: str
    subtopics: list[_SubtopicOut] = Field(default_factory=list)
    session_type: str = LESSON
    confidence: float = 0.0
    evidence: list[_EvidenceOut] = Field(default_factory=list)
    reasoning: str = ""
    abstain: bool = False
    abstain_reason: str = ""


def response_schema(chapter_ids: list[str], subtopic_ids: list[str]) -> dict[str, Any]:
    """The reply schema, with the syllabus' own ids as the legal values.

    Built per subject rather than declared once, because the whole point is
    that the enumeration *is* the syllabus. A schema with `node_id: string`
    would accept "bio.12_2.krebs_cycle_and_link_reaction" from a syllabus that
    only has "bio.12_2.the_link_reaction_and_the_krebs_cycle", and the two are
    indistinguishable to a reader of the reply.
    """

    def id_field(values: list[str], description: str) -> dict[str, Any]:
        field: dict[str, Any] = {"type": "string", "description": description}
        total = sum(len(v) for v in values)
        if values and len(values) <= MAX_ENUM_VALUES and total <= MAX_ENUM_CHARS:
            field["enum"] = values
        else:
            log.debug(
                "%d ids (%d chars) is past what a structured-output enum takes; ids "
                "will be checked after the reply instead",
                len(values),
                total,
            )
        return field

    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "chapter_id",
            "subtopics",
            "session_type",
            "confidence",
            "evidence",
            "reasoning",
            "abstain",
            "abstain_reason",
        ],
        "properties": {
            "chapter_id": id_field(
                [*chapter_ids, NO_CHAPTER],
                f"Id of the one chapter this lecture belongs to, or {NO_CHAPTER!r} "
                "when abstaining.",
            ),
            "subtopics": {
                "type": "array",
                "description": (
                    "One to three subtopics of that chapter, most-taught first. "
                    "Empty when abstaining."
                ),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["node_id", "coverage"],
                    "properties": {
                        "node_id": id_field(subtopic_ids, "Id of a subtopic in that chapter."),
                        "coverage": {
                            "type": "number",
                            "description": "Fraction of lecture time on this subtopic, 0 to 1.",
                        },
                    },
                },
            },
            "session_type": {
                "type": "string",
                "enum": [LESSON, ORIENTATION],
                "description": (
                    "'lesson' for a class about the subject. 'orientation' for one about "
                    "the course itself — the first session of the year, the term plan, "
                    "exam papers and dates, calculator rules, where the resources are."
                ),
            },
            "confidence": {
                "type": "number",
                "description": "Genuine certainty in the chapter and first subtopic, 0 to 1.",
            },
            "evidence": {
                "type": "array",
                "description": "At least two verbatim quotes from the transcript, with times.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["quote", "start_s"],
                    "properties": {
                        "quote": {"type": "string"},
                        "start_s": {
                            "type": "number",
                            "description": "Seconds from the start of the recording.",
                        },
                    },
                },
            },
            "reasoning": {
                "type": "string",
                "description": "Two or three sentences: what was taught and how you know.",
            },
            "abstain": {"type": "boolean"},
            "abstain_reason": {"type": "string"},
        },
    }


# ── Prompt ────────────────────────────────────────────────────────────────────


def default_allowed_ids(taxonomy: Taxonomy) -> set[str]:
    """Which nodes may name a file when the caller does not say.

    The deepest layer the syllabus actually has, which is the rule
    `scripts/rename_folder.py` applies with `--level`: a 9701 subtopic is one
    lecture, a 9700 subtopic is six, and the hand-curated lesson layer under it
    exists precisely because of that.
    """
    nodes = taxonomy.classifiable_nodes()
    lessons = {n.id for n in nodes if n.depth > 1 and not n.children}
    if lessons:
        return lessons
    return {n.id for n in nodes if n.depth == 1} | {
        n.id for n in nodes if n.depth == 0 and not n.children
    }


def chapter_nodes(taxonomy: Taxonomy) -> list[SyllabusNode]:
    """Every depth-0 node, including the ones with no Drive folder yet.

    `Taxonomy.pending_nodes` holds topics the syllabus has and the folder tree
    does not, and it is not a rare corner: all nine IGCSE Maths chapters are
    there, and one of the twenty-four Chemistry ones. Walking `nodes` alone
    showed the model an empty syllabus for Maths — every lecture would have
    abstained, correctly according to the prompt and for a reason no console
    line would have explained.

    Naming a recording is not placing it. A pending chapter can name a file
    perfectly well; what it cannot do is receive one, and that is the folder
    resolver's decision to take later.
    """
    return [
        node
        for node in (*taxonomy.nodes, *taxonomy.pending_nodes)
        if not node.exclude_from_classification
    ]


def orientation_node(taxonomy: Taxonomy, label: str) -> SyllabusNode | None:
    """The chapter a course-orientation session is filed under, if there is one.

    Found by title rather than by a dedicated flag, because the title is
    `naming.orientation_label` — one string doing both jobs, so the name on the
    file and the folder it goes into cannot drift apart. Renaming the folder is
    a config change, not a code change.

    Deliberately searches nodes that are `exclude_from_classification`. That
    flag means "this must not compete with real lessons for a transcript",
    which is a different claim from "this can never be the answer": an intro
    session name-checks half the syllabus in forty minutes and would outscore
    the lecture that actually taught it, which is exactly why it is kept out of
    the candidate list and exactly why it is reachable here, where the model
    has already said what kind of session this is.
    """
    wanted = label.strip().lower()
    for node in (*taxonomy.nodes, *taxonomy.pending_nodes):
        if node.title.strip().lower() == wanted:
            return node
    return None


def _node_block(node: SyllabusNode, marker: str, *, choosable: bool) -> list[str]:
    """One syllabus node as the model sees it.

    Keywords and aliases are included because they are what the teacher says
    out loud — "AOD" appears in the user's own filenames and in no syllabus
    document — and learning outcomes because they are often the only text that
    distinguishes two subtopics whose titles are both three words long.
    """
    lines = [f"{marker} {node.id} — {node.title}"]
    if node.short_title and node.short_title != node.title:
        lines.append(f"    filename label: {node.short_title}")
    if not choosable:
        lines.append("    (grouping only — do not return this id)")
    if node.aliases:
        lines.append(f"    also called: {', '.join(node.aliases[:10])}")
    if node.keywords:
        lines.append(f"    keywords: {', '.join(node.keywords[:24])}")
    for outcome in node.learning_outcomes[:8]:
        lines.append(f"    - {outcome.text}")
    return lines


def syllabus_block(taxonomy: Taxonomy, allowed: set[str]) -> str:
    """The whole syllabus, chapter by chapter, with the choosable ids marked.

    Nested rather than flat because the one-chapter rule is a claim about this
    tree: the model has to see which subtopics share a folder to know that
    naming two of them is one decision and naming two from different chapters
    is a contradiction.
    """
    heading = f"# Syllabus — {taxonomy.subject_name}"
    if taxonomy.syllabus_code:
        heading += f" ({taxonomy.syllabus_code})"
    lines = [heading, ""]

    def walk(nodes: list[SyllabusNode], depth: int) -> None:
        for node in nodes:
            if node.exclude_from_classification:
                continue
            marker = "##" if depth == 0 else ("###" if depth == 1 else "####")
            label = {0: "CHAPTER", 1: "SUBTOPIC"}.get(depth, "LESSON")
            lines.extend(_node_block(node, f"{marker} {label}", choosable=node.id in allowed))
            lines.append("")
            walk(node.children, depth + 1)

    walk(chapter_nodes(taxonomy), 0)
    return "\n".join(lines)


def transcript_block(
    transcript: Transcript,
    *,
    line_seconds: int = TRANSCRIPT_LINE_SECONDS,
    char_budget: int = TRANSCRIPT_CHAR_BUDGET,
) -> str:
    """The whole lecture, one timestamped line per `line_seconds`.

    Timestamps are what make rule 7 checkable and rule 5 possible: the model
    cannot tell a recap from the subject of the lecture without knowing where
    in the hour it happened.

    Over budget, every line is trimmed by the same fraction rather than the
    tail being dropped. Losing the end of a lecture would silently delete the
    material a two-topic lesson spends its second half on — the exact thing
    this classifier exists to notice.
    """
    buckets: dict[int, list[str]] = {}
    for segment in transcript.segments:
        text = segment.text.strip()
        if text:
            buckets.setdefault(int(segment.start_s // line_seconds), []).append(text)

    lines = []
    for bucket in sorted(buckets):
        start = bucket * line_seconds
        lines.append(f"[{start // 60:02d}:{start % 60:02d}] {' '.join(buckets[bucket])}")

    rendered = "\n".join(lines)
    if len(rendered) > char_budget and lines:
        ratio = char_budget / len(rendered)
        log.warning(
            "transcript is %d characters; trimming every line to %.0f%% to fit the prompt",
            len(rendered),
            ratio * 100,
        )
        lines = [line[: max(40, int(len(line) * ratio))] for line in lines]
        rendered = "\n".join(lines)
    return rendered or "(the transcript is empty)"


# ── Classifier ────────────────────────────────────────────────────────────────


class OpenAIClassifier:
    """ChatGPT reads the lecture and names it out of the syllabus.

    Satisfies the same `classify.service.Classifier` protocol as the keyword
    baseline, so the pipeline, the dry run and `rename_folder.py` swap one for
    the other. `candidates` is accepted and ignored: it is a retrieval
    shortlist, and narrowing the syllabus by keyword before the model reads it
    is the failure this class exists to replace.
    """

    name = "openai"

    def __init__(
        self,
        taxonomy: Taxonomy,
        settings: ClassifySettings | None = None,
        naming: NamingSettings | None = None,
        *,
        api_key: str | None = None,
        allowed_ids: set[str] | None = None,
    ) -> None:
        self.taxonomy = taxonomy
        self.settings = settings or ClassifySettings()
        self.naming = naming or NamingSettings()
        self.api_key = api_key
        self.allowed = allowed_ids or default_allowed_ids(taxonomy)
        self._client = None

    # ── API ──────────────────────────────────────────────────────────────────

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise SystemExit(
                    'The OpenAI client is not installed. pip install -e ".[classify]"'
                ) from None
            options: dict[str, Any] = {
                # Above the SDK default of two. One lecture is one call, so a
                # transient 429 halfway through a folder should cost seconds
                # rather than a re-run of everything after it.
                "max_retries": 5,
                "timeout": self.settings.openai_timeout_s,
            }
            # A bare constructor reads OPENAI_API_KEY from the environment,
            # which is how the .env path supplies it.
            if self.api_key:
                options["api_key"] = self.api_key
            if self.settings.openai_base_url:
                options["base_url"] = self.settings.openai_base_url
            self._client = OpenAI(**options)
        return self._client

    def classify(
        self,
        recording: Recording,
        transcript: Transcript,
        candidates: CandidateSet | None = None,
    ) -> Classification:
        chapters = [n.id for n in chapter_nodes(self.taxonomy)]
        subtopics = [n.id for n in self.taxonomy.classifiable_nodes() if n.id in self.allowed]
        if not chapters or not subtopics:
            # Loudly, and without a network call. An empty option list is a
            # loaded-the-wrong-file problem, and a model asked to choose from
            # nothing abstains — which reads as "this lecture is off-syllabus"
            # rather than as "the syllabus never arrived".
            return Classification(
                abstain=True,
                abstain_reason=(
                    f"the syllabus offered nothing to choose from: {len(chapters)} chapters "
                    f"and {len(subtopics)} selectable subtopics"
                ),
                method=self.name,
            )

        reply = self._ask(recording, transcript, response_schema(chapters, subtopics))
        if reply is None:
            return Classification(
                abstain=True,
                abstain_reason="the model declined to answer",
                method=self.name,
                model=self.settings.openai_model,
            )
        return self._to_domain(reply)

    def _ask(
        self, recording: Recording, transcript: Transcript, schema: dict[str, Any]
    ) -> _ReplyOut | None:
        """One call. Returns None when the model refused to answer."""
        extra: dict[str, Any] = {}
        if self.settings.openai_reasoning_effort:
            extra["reasoning_effort"] = self.settings.openai_reasoning_effort

        response = self._get_client().chat.completions.create(
            model=self.settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt(self.settings)},
                {"role": "user", "content": self._build_prompt(recording, transcript)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "lecture_classification",
                    "strict": True,
                    "schema": schema,
                },
            },
            max_completion_tokens=self.settings.openai_max_output_tokens,
            **extra,
        )

        choice = response.choices[0]
        if getattr(choice.message, "refusal", None):
            log.warning("model refused: %s", choice.message.refusal)
            return None
        if choice.finish_reason == "length":
            # The JSON is emitted last, after any reasoning, so a truncated
            # reply is not partially usable — it is not JSON at all.
            raise RuntimeError(
                f"the reply hit the {self.settings.openai_max_output_tokens}-token output "
                "limit before finishing. Raise classify.openai_max_output_tokens, or "
                "lower classify.openai_reasoning_effort."
            )
        return _ReplyOut.model_validate(json.loads(choice.message.content or "{}"))

    def _build_prompt(self, recording: Recording, transcript: Transcript) -> str:
        minutes = (transcript.duration_s or 0) / 60
        parts = [
            syllabus_block(self.taxonomy, self.allowed),
            "",
            f"# Transcript ({minutes:.0f} minutes; timestamps are mm:ss from the start)",
            "",
            transcript_block(transcript),
            "",
            "# Weak signals — the transcript overrides all of these",
            "",
            f"Recorded: {recording.recorded_at:%Y-%m-%d %H:%M}",
            f"Original filename: {recording.original_filename}",
        ]
        if recording.meeting_topic:
            parts.append(f"Zoom meeting topic: {recording.meeting_topic}")
        if recording.teacher:
            parts.append(f"Teacher: {recording.teacher}")
        parts += [
            "",
            "Filenames and meeting topics on these recordings are generic "
            '("class 4", "Bio lecture") and frequently wrong. Do not classify from them.',
            "",
            "Now return the one chapter, and the one-to-three subtopics inside it that "
            "this lecture taught — or abstain.",
        ]
        return "\n".join(parts)

    # ── Reply -> domain ──────────────────────────────────────────────────────

    def _orientation(self, reply: _ReplyOut) -> Classification:
        """The class about the course rather than about the subject.

        It teaches no subtopic, so on the old path it abstained and came out
        named *Unclassified* — which says "I could not work out what this is"
        about the one recording whose nature is least in doubt. It is the first
        session of the year, and it has a folder: "0. Introduction".

        Two outcomes, and the difference is whether this syllabus has that
        folder. With one, the recording is filed and named from the node like
        any other. Without one, it still gets the name — that is the part the
        reader wants — but it goes to review, because there is genuinely
        nowhere to put it and pretending otherwise would file it into a folder
        chosen at random.
        """
        node = orientation_node(self.taxonomy, self.naming.orientation_label)
        evidence = [Evidence(quote=e.quote, start_s=e.start_s) for e in reply.evidence[:4]]
        confidence = max(0.0, min(1.0, reply.confidence))

        if node is None:
            log.info(
                "orientation session, but %r has no %r chapter to file it under; "
                "naming it and routing to review",
                self.taxonomy.subject_id,
                self.naming.orientation_label,
            )
            classification = Classification(
                abstain=True,
                abstain_reason=(
                    f"course orientation, not a lesson — and this syllabus has no "
                    f"{self.naming.orientation_label!r} chapter to file it under. Add one "
                    f"to the overlay to have these filed automatically."
                ),
                reasoning=reply.reasoning,
                method=self.name,
                model=self.settings.openai_model,
            )
            # Set after construction and deliberately kept through the abstain,
            # unlike the retrieval paths. There the label is a *proposed
            # placement* and carrying one past an abstain would name a file
            # after the node the classifier just rejected. This label makes no
            # claim about placement at all — it says what kind of session this
            # is, which is the one thing the model was sure of.
            classification.suggested_topic_label = self.naming.orientation_label
            return classification

        return Classification(
            primary=TopicMatch(
                node_id=node.id,
                confidence=confidence,
                # The whole recording is the orientation session. Stated rather
                # than taken from the reply, which reports coverage per
                # subtopic and there are none: left at 0 it would fail the
                # coverage floor and route to review a decision nothing doubts.
                coverage=1.0,
                evidence=evidence,
            ),
            chapter_node_id=node.id,
            chapter_weights={node.id: 1.0},
            topic_node_ids=[node.id],
            reasoning=reply.reasoning,
            suggested_topic_label=node.label,
            method=self.name,
            model=self.settings.openai_model,
        )

    def _to_domain(self, reply: _ReplyOut) -> Classification:
        """Turn the reply into a Classification, believing none of its ids.

        The enum in the schema already makes an invented id impossible in a
        conforming reply. This is the check that holds when the enum was too
        large to send, when a future model is pointed at a stale schema, and
        when the reply conforms but names a subtopic belonging to a different
        chapter than the one it just chose.
        """
        model_name = self.settings.openai_model

        def abstaining(reason: str) -> Classification:
            return Classification(
                abstain=True,
                abstain_reason=reason,
                reasoning=reply.reasoning,
                method=self.name,
                model=model_name,
            )

        if reply.session_type == ORIENTATION:
            return self._orientation(reply)

        if reply.abstain or reply.chapter_id == NO_CHAPTER or not reply.subtopics:
            return abstaining(reply.abstain_reason or "the model found no subtopic this fits")

        chapter = self.taxonomy.get(reply.chapter_id)
        if chapter is None:
            return abstaining(f"the model returned unknown chapter id {reply.chapter_id!r}")

        chosen: list[_SubtopicOut] = []
        for item in reply.subtopics:
            if any(item.node_id == already.node_id for already in chosen):
                continue
            if self.taxonomy.get(item.node_id) is None:
                log.warning("dropping unknown node id %r from the reply", item.node_id)
                continue
            if chapter_of(self.taxonomy, item.node_id) != chapter.id:
                # A subtopic from another chapter contradicts the folder the
                # file is about to sit in, so the name would disagree with the
                # placement — and only one of the two is visible in a listing.
                log.warning(
                    "dropping %r: it belongs to chapter %r, not the chosen %r",
                    item.node_id,
                    chapter_of(self.taxonomy, item.node_id),
                    chapter.id,
                )
                continue
            chosen.append(item)
            if len(chosen) >= self.naming.max_subtopics:
                break

        if not chosen:
            return abstaining(
                "none of the subtopics the model named ("
                + ", ".join(s.node_id for s in reply.subtopics)
                + f") is a known node inside chapter {chapter.id!r}"
            )

        def clamp(value: float) -> float:
            return max(0.0, min(1.0, value))

        primary = chosen[0]
        classification = Classification(
            primary=TopicMatch(
                node_id=primary.node_id,
                confidence=clamp(reply.confidence),
                coverage=clamp(primary.coverage),
                evidence=[Evidence(quote=e.quote, start_s=e.start_s) for e in reply.evidence[:4]],
            ),
            secondary=[
                TopicMatch(node_id=s.node_id, confidence=0.0, coverage=clamp(s.coverage))
                for s in chosen[1:]
            ],
            chapter_node_id=chapter.id,
            # How much of the lecture the chosen chapter accounts for, which is
            # what a reviewer wants and is the model's own number rather than a
            # keyword count. There is no runner-up: the model was asked for one
            # chapter, so a second weight here would be an invention.
            chapter_weights={chapter.id: clamp(sum(s.coverage for s in chosen))},
            topic_node_ids=[s.node_id for s in chosen],
            reasoning=reply.reasoning,
            method=self.name,
            model=model_name,
        )
        # From the syllabus, never from the model. `naming.template` prefers
        # `topic_node_ids` and renders the same labels from them, so the two
        # agree by construction; this is what a reviewer reads in
        # rename_plan.json, and what naming falls back to if the ids are lost.
        classification.suggested_topic_label = self.naming.subtopic_separator.join(
            node.label
            for node in (self.taxonomy.get(s.node_id) for s in chosen)
            if node is not None
        )
        return classification
