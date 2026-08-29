"""Domain models and the processing state machine.

These types are the contract between stages. Every stage takes a `Recording`
plus the artifacts of prior stages and returns a new artifact; nothing mutates
shared state in place, which is what makes retries safe.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utcnow() -> datetime:
    """Timezone-aware UTC now.

    `datetime.utcnow()` returns a naive datetime and is deprecated; naive UTC
    timestamps are also a recurring source of off-by-hours bugs once a second
    deployment region appears.
    """
    return datetime.now(UTC)


# ── State machine ─────────────────────────────────────────────────────────────


class Stage(StrEnum):
    """Where a recording is in the pipeline.

    Forward transitions only, except that any stage can fail. The happy path is
    DISCOVERED -> ... -> DONE; CLASSIFIED can branch to NEEDS_REVIEW, which a
    human resolves by moving the file in Drive (see docs/OPERATIONS.md).
    """

    DISCOVERED = "discovered"
    FETCHED = "fetched"
    AUDIO_READY = "audio_ready"
    TRANSCRIBED = "transcribed"
    CLASSIFIED = "classified"
    NAMED = "named"
    PLACED = "placed"
    PUBLISHED = "published"
    DONE = "done"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


#: Legal transitions. Enforced by `Recording.can_transition_to` so a bug in the
#: orchestrator surfaces as a loud error rather than a job in a nonsense state.
ALLOWED_TRANSITIONS: dict[Stage, set[Stage]] = {
    Stage.DISCOVERED: {Stage.FETCHED, Stage.FAILED},
    Stage.FETCHED: {Stage.AUDIO_READY, Stage.FAILED},
    Stage.AUDIO_READY: {Stage.TRANSCRIBED, Stage.FAILED},
    Stage.TRANSCRIBED: {Stage.CLASSIFIED, Stage.FAILED},
    Stage.CLASSIFIED: {Stage.NAMED, Stage.NEEDS_REVIEW, Stage.FAILED},
    Stage.NAMED: {Stage.PLACED, Stage.FAILED},
    Stage.PLACED: {Stage.PUBLISHED, Stage.FAILED},
    Stage.PUBLISHED: {Stage.DONE, Stage.FAILED},
    Stage.DONE: set(),
    # A human moving the file out of _Needs Review re-enters the pipeline at
    # NAMED, because the classification has effectively been supplied by hand.
    Stage.NEEDS_REVIEW: {Stage.NAMED, Stage.FAILED},
    Stage.FAILED: {Stage.DISCOVERED},  # manual retry restarts from the top
}

TERMINAL_STAGES = {Stage.DONE, Stage.FAILED}


class Source(StrEnum):
    ZOOM = "zoom"
    DRIVE = "drive"
    MANUAL = "manual"


# ── Media ─────────────────────────────────────────────────────────────────────


class MediaKind(StrEnum):
    VIDEO = "video"
    AUDIO = "audio"


class MediaAsset(BaseModel):
    """A downloadable file belonging to a recording.

    Zoom exposes both an MP4 and a much smaller audio-only M4A for the same
    meeting; we prefer the latter for transcription. See ADR-003.
    """

    kind: MediaKind
    file_extension: str
    size_bytes: int | None = None
    download_url: str | None = None
    #: Zoom download tokens are short-lived, so this is fetched at use time.
    requires_token: bool = False
    drive_file_id: str | None = None
    local_path: str | None = None


# ── Transcription ─────────────────────────────────────────────────────────────


class TranscriptSegment(BaseModel):
    start_s: float
    end_s: float
    text: str
    speaker: str | None = None
    #: High values flag segments the model was unsure were speech at all —
    #: useful for spotting hallucination over silence.
    no_speech_prob: float | None = None

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class Transcript(BaseModel):
    segments: list[TranscriptSegment]
    language: str
    duration_s: float
    model: str
    provider: str
    #: Wall-clock seconds the ASR run took. Tracked because it is the main
    #: capacity input when sizing GPU workers.
    runtime_s: float | None = None
    created_at: datetime = Field(default_factory=utcnow)

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def window(self, start_s: float, end_s: float) -> str:
        """Text of all segments overlapping [start_s, end_s)."""
        return " ".join(
            s.text.strip()
            for s in self.segments
            if s.end_s > start_s and s.start_s < end_s and s.text.strip()
        )


class TranscriptionHints(BaseModel):
    """Subject-specific priming for the ASR model.

    Populated from the syllabus taxonomy. Materially improves how specialist
    vocabulary is transcribed, which matters because classification depends on
    exactly those terms.
    """

    language: str | None = None
    initial_prompt: str | None = None
    hotwords: list[str] = Field(default_factory=list)


# ── Classification ────────────────────────────────────────────────────────────


class Evidence(BaseModel):
    """A quote from the transcript supporting a classification.

    Required (see docs/CLASSIFICATION.md): a classification without evidence is
    unauditable, and an unauditable classifier degrades a folder tree silently.
    """

    quote: str
    start_s: float


class TopicMatch(BaseModel):
    node_id: str
    #: How sure the classifier is that this node is correct, 0-1.
    confidence: float = Field(ge=0.0, le=1.0)
    #: Share of the lecture attributable to this node, 0-1.
    coverage: float = Field(ge=0.0, le=1.0, default=0.0)
    evidence: list[Evidence] = Field(default_factory=list)


#: A filename may name at most this many syllabus subtopics. Stated once, here,
#: because it has to hold for every path that can build a Classification — the
#: keyword classifier, an LLM reply, a hand-written fixture — and a rule that
#: lives in only one of those is not a rule.
MAX_SUBTOPICS_IN_NAME = 2


class Classification(BaseModel):
    #: Validated on assignment, not only on construction. The one code path
    #: that writes `topic_node_ids` writes it by assignment — `classify/
    #: service.py` builds the object first and fills the field afterwards —
    #: and pydantic v2 skips validators on assignment unless asked, so without
    #: this the never-three rule was enforced everywhere except where the code
    #: actually goes.
    model_config = ConfigDict(validate_assignment=True)

    primary: TopicMatch | None = None
    secondary: list[TopicMatch] = Field(default_factory=list)

    #: The one Drive folder this recording belongs in, as a depth-0 node id.
    #: A recording is never split across two chapters: it is one lesson, it
    #: goes in one place, and a shortcut in a second chapter would make the
    #: archive look larger than it is.
    chapter_node_id: str | None = None
    #: chapter node id -> share of the lecture, summing to 1. Kept even when
    #: the winner is obvious, because a near-tie here is the signature of a
    #: misfiling and is invisible in the final answer.
    chapter_weights: dict[str, float] = Field(default_factory=dict)
    #: The one or two subtopics that name the file, in coverage order. Empty
    #: for callers that only set `primary` — see naming.template.
    topic_node_ids: list[str] = Field(default_factory=list)

    #: True when this looks like the continuation of the previous lecture —
    #: drives "Part 2" naming rather than a duplicate-name collision.
    is_continuation: bool = False
    #: What to put in the filename. Often more specific than the node title.
    suggested_topic_label: str | None = None
    reasoning: str = ""
    abstain: bool = False
    abstain_reason: str | None = None
    #: Which classifier produced this, for reproducibility.
    method: str = "hybrid+llm"
    model: str | None = None
    created_at: datetime = Field(default_factory=utcnow)

    @field_validator("topic_node_ids")
    @classmethod
    def _at_most_two_subtopics(cls, v: list[str]) -> list[str]:
        """Refuse a third subtopic at the type boundary.

        Truncating silently would be friendlier and worse: the caller that
        supplied three — most likely a model reply — would keep supplying
        three, and nobody would ever see it.
        """
        if len(v) > MAX_SUBTOPICS_IN_NAME:
            raise ValueError(
                f"a filename may name at most {MAX_SUBTOPICS_IN_NAME} subtopics; "
                f"got {len(v)}: {v}"
            )
        return v

    def needs_review(self, threshold: float, coverage_floor: float) -> bool:
        """Whether this result must go to a human rather than being filed.

        Two independent gates. Low confidence is the obvious one. High
        confidence with low coverage is the subtler one: the model is sure
        about a topic that occupied very little of the lecture, which usually
        means the lecture's real subject is not in the taxonomy at all.
        """
        if self.abstain or self.primary is None:
            return True
        if self.primary.confidence < threshold:
            return True
        return self.primary.coverage < coverage_floor


# ── Naming and placement ──────────────────────────────────────────────────────


class NamePlan(BaseModel):
    filename: str
    original_filename: str
    template_used: str
    is_low_confidence: bool = False
    part_number: int | None = None


class PlacementPlan(BaseModel):
    target_folder_id: str
    target_folder_path: str
    node_id: str
    #: How the folder was found: exact | normalised | fuzzy | created | review.
    #: Anything other than `exact` is logged for confirmation.
    resolution_method: str
    resolution_confidence: float = 1.0
    #: Folders for secondary topics, if shortcuts are enabled.
    shortcut_folder_ids: list[str] = Field(default_factory=list)


# ── The job row ───────────────────────────────────────────────────────────────


class Recording(BaseModel):
    """One lecture recording moving through the pipeline.

    Deduplicated on (source, source_id). A replayed webhook, a reconciler
    re-detection, and a re-uploaded file all resolve to the same row.
    """

    id: str
    source: Source
    source_id: str
    subject_id: str

    # As discovered
    original_filename: str
    recorded_at: datetime
    duration_s: float | None = None
    host_email: str | None = None
    teacher: str | None = None
    #: Zoom meeting topic. A weak signal only — see docs/CLASSIFICATION.md.
    meeting_topic: str | None = None

    # Drive identity
    drive_file_id: str | None = None
    drive_parent_id: str | None = None
    content_hash: str | None = None

    # Artifacts, filled in as stages complete
    assets: list[MediaAsset] = Field(default_factory=list)
    audio_path: str | None = None
    transcript: Transcript | None = None
    classification: Classification | None = None
    name_plan: NamePlan | None = None
    placement_plan: PlacementPlan | None = None

    # Bookkeeping
    stage: Stage = Stage.DISCOVERED
    attempts: dict[str, int] = Field(default_factory=dict)
    last_error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    @field_validator("subject_id")
    @classmethod
    def _subject_required(cls, v: str) -> str:
        if not v:
            raise ValueError("subject_id is required; a recording must belong to a subject")
        return v

    @property
    def recorded_on(self) -> date:
        return self.recorded_at.date()

    @property
    def dedupe_key(self) -> str:
        return f"{self.source}:{self.source_id}"

    def can_transition_to(self, target: Stage) -> bool:
        return target in ALLOWED_TRANSITIONS.get(self.stage, set())

    def transition_to(self, target: Stage) -> None:
        """Move to `target`, refusing illegal transitions.

        Deliberately strict: a pipeline that silently accepts a nonsense
        transition hides orchestration bugs until they corrupt data.
        """
        if not self.can_transition_to(target):
            raise ValueError(
                f"illegal transition {self.stage} -> {target} for recording {self.id}"
            )
        self.stage = target
        self.updated_at = utcnow()

    def record_attempt(self, step: str) -> int:
        self.attempts[step] = self.attempts.get(step, 0) + 1
        return self.attempts[step]

    def is_terminal(self) -> bool:
        return self.stage in TERMINAL_STAGES


class ProcessingResult(BaseModel):
    """What the pipeline did, for logging and the dry-run report."""

    recording_id: str
    final_stage: Stage
    filename: str | None = None
    folder_path: str | None = None
    node_id: str | None = None
    confidence: float | None = None
    error: str | None = None
    stage_timings: dict[str, float] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)

