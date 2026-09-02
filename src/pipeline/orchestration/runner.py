"""The pipeline runner.

Walks one recording through the stages, committing after each. Every stage is
idempotent, so a worker that dies mid-flight simply has its job resumed from
the last committed state — nothing needs unwinding.

This class holds the sequencing and nothing else — no queue, no broker, no
scheduler. It is called synchronously, which is what the dry run and the tests
do, and is what a cron job will do. One machine processes one recording at a
time; a queue is the answer to many workers competing for work, and there is
one worker.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from pipeline.asr import ASRProvider
from pipeline.classify import ClassificationService
from pipeline.config import Settings, SubjectSettings
from pipeline.media import extract_audio
from pipeline.models import (
    Classification,
    PlacementPlan,
    ProcessingResult,
    Recording,
    Stage,
    TranscriptionHints,
)
from pipeline.naming import build_name
from pipeline.placement import FolderResolver
from pipeline.publish import render_markdown, render_sidecar, render_srt, render_vtt
from pipeline.storage import DrivePort
from pipeline.syllabus import Taxonomy

log = logging.getLogger(__name__)


class PipelineRunner:
    def __init__(
        self,
        *,
        settings: Settings,
        subject: SubjectSettings,
        taxonomy: Taxonomy,
        drive: DrivePort,
        asr: ASRProvider,
        classifier: ClassificationService,
        resolver: FolderResolver,
        review_folder_id: str | None = None,
        work_dir: Path | None = None,
    ) -> None:
        self.settings = settings
        self.subject = subject
        self.taxonomy = taxonomy
        self.drive = drive
        self.asr = asr
        self.classifier = classifier
        self.resolver = resolver
        self.review_folder_id = review_folder_id
        self.work_dir = work_dir or settings.runtime.media_cache_dir
        self.work_dir = Path(self.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    # ── Entry point ──────────────────────────────────────────────────────────

    def process(self, recording: Recording) -> ProcessingResult:
        timings: dict[str, float] = {}
        try:
            for stage_name, step in (
                ("fetch", self.fetch),
                ("prepare_audio", self.prepare_audio),
                ("transcribe", self.transcribe),
                ("classify", self.classify),
                ("plan_name", self.plan_name),
                ("place", self.place),
                ("publish", self.publish),
            ):
                started = time.monotonic()
                step(recording)
                timings[stage_name] = round(time.monotonic() - started, 2)
                # A review decision ends this run; a human resumes it by
                # moving the file in Drive.
                if recording.stage == Stage.NEEDS_REVIEW:
                    break

            if recording.stage == Stage.PUBLISHED:
                recording.transition_to(Stage.DONE)

        except Exception as exc:  # noqa: BLE001 - the boundary logs and records
            log.exception("processing failed for %s", recording.id)
            recording.last_error = f"{type(exc).__name__}: {exc}"
            if recording.can_transition_to(Stage.FAILED):
                recording.transition_to(Stage.FAILED)
            return ProcessingResult(
                recording_id=recording.id,
                final_stage=recording.stage,
                error=recording.last_error,
                stage_timings=timings,
            )

        classification = recording.classification
        return ProcessingResult(
            recording_id=recording.id,
            final_stage=recording.stage,
            filename=recording.name_plan.filename if recording.name_plan else None,
            folder_path=(
                recording.placement_plan.target_folder_path if recording.placement_plan else None
            ),
            node_id=(
                classification.primary.node_id
                if classification and classification.primary
                else None
            ),
            confidence=(
                classification.primary.confidence
                if classification and classification.primary
                else None
            ),
            stage_timings=timings,
            details={
                "reasoning": classification.reasoning if classification else "",
                "abstain_reason": classification.abstain_reason if classification else None,
            },
        )

    # ── Stages ───────────────────────────────────────────────────────────────

    def _local_source(self, recording: Recording) -> Path:
        """Where the downloaded media lives. Derived, not remembered.

        Deriving it from the recording id means a resumed job finds the file a
        previous worker downloaded, instead of re-fetching it.
        """
        suffix = Path(recording.original_filename).suffix or ".mp4"
        return self.work_dir / f"{recording.id}{suffix}"

    def fetch(self, recording: Recording) -> None:
        """Bring the media within reach. Idempotent."""
        if recording.stage != Stage.DISCOVERED:
            return
        if not recording.drive_file_id:
            raise ValueError(f"recording {recording.id} has no drive_file_id to fetch")

        local = self._local_source(recording)
        if not local.exists():
            self.drive.download(recording.drive_file_id, local)
        recording.transition_to(Stage.FETCHED)

    def prepare_audio(self, recording: Recording) -> None:
        if recording.stage != Stage.FETCHED:
            return
        source = self._local_source(recording)
        if not source.exists():
            raise RuntimeError(f"media not present at {source}; fetch() must run first")

        audio = self.work_dir / f"{recording.id}.wav"
        if source.suffix.lower() == ".wav":
            audio = source
        else:
            try:
                audio = extract_audio(source, audio)
            except Exception:
                # A stub ASR provider reads a fixture and needs no real audio,
                # so a missing ffmpeg must not block the dry run.
                if self.asr.name == "local_stub":
                    audio = source
                else:
                    raise
        recording.audio_path = str(audio)
        recording.transition_to(Stage.AUDIO_READY)

    def transcribe(self, recording: Recording) -> None:
        if recording.stage != Stage.AUDIO_READY:
            return
        available, reason = self.asr.is_available()
        if not available:
            raise RuntimeError(f"ASR provider {self.asr.name} unavailable: {reason}")

        language = self.subject.languages[0] if self.subject.languages else "auto"
        hints = TranscriptionHints(
            language=None if language == "auto" else language,
            initial_prompt=(
                self.taxonomy.initial_prompt() if self.settings.asr.prime_from_syllabus else None
            ),
            hotwords=(
                self.taxonomy.vocabulary(self.settings.asr.max_hotwords)
                if self.settings.asr.prime_from_syllabus
                else []
            ),
        )
        recording.transcript = self.asr.transcribe(Path(recording.audio_path), hints)
        recording.transition_to(Stage.TRANSCRIBED)

        if not self.settings.runtime.keep_audio_after_transcription:
            Path(recording.audio_path).unlink(missing_ok=True)

    def classify(self, recording: Recording) -> None:
        if recording.stage != Stage.TRANSCRIBED:
            return
        if recording.transcript is None:
            raise RuntimeError("cannot classify without a transcript")

        classification, _candidates = self.classifier.run(recording, recording.transcript)
        recording.classification = classification
        recording.transition_to(Stage.CLASSIFIED)

        if self.classifier.needs_review(classification):
            log.info(
                "recording %s routed to review: %s",
                recording.id,
                classification.abstain_reason or "below confidence threshold",
            )
            recording.transition_to(Stage.NEEDS_REVIEW)
            self._place_in_review(recording, classification)

    def plan_name(self, recording: Recording) -> None:
        if recording.stage != Stage.CLASSIFIED:
            return
        assert recording.classification is not None

        target = self.resolver.resolve(recording.classification.primary.node_id)  # type: ignore[union-attr]
        existing = (
            {f.name for f in self.drive.list_children(target.folder_id)}
            if target.resolved
            else set()
        )
        recording.name_plan = build_name(
            recording,
            recording.classification,
            self.taxonomy,
            self.settings.naming,
            existing_names=existing,
            is_low_confidence=False,
        )
        recording.transition_to(Stage.NAMED)
        self._target = target

    def place(self, recording: Recording) -> None:
        if recording.stage != Stage.NAMED:
            return
        assert recording.name_plan is not None
        target = getattr(self, "_target", None)
        if target is None or not target.resolved:
            raise RuntimeError(
                f"no Drive folder resolved for recording {recording.id}: "
                f"{target.note if target else 'resolver not run'}"
            )

        # Rename first, then move. If the move fails, the file keeps a correct
        # name in its original location rather than a stale name in a new one.
        #
        # Always chain through the returned handle rather than reusing the
        # original id. Google Drive ids survive a rename, but that is a
        # property of Google's adapter, not of the interface — the local
        # adapter keys on path. Threading the handle keeps the runner correct
        # against both.
        renamed = self.drive.rename(recording.drive_file_id, recording.name_plan.filename)  # type: ignore[arg-type]
        moved = self.drive.move(renamed.id, target.folder_id, recording.drive_parent_id)

        recording.drive_file_id = moved.id
        recording.drive_parent_id = target.folder_id
        recording.placement_plan = PlacementPlan(
            target_folder_id=target.folder_id,
            target_folder_path=target.folder_path,
            node_id=recording.classification.primary.node_id,  # type: ignore[union-attr]
            resolution_method=target.method,
            resolution_confidence=target.confidence,
        )
        recording.transition_to(Stage.PLACED)

    def publish(self, recording: Recording) -> None:
        """Upload transcript artifacts alongside the recording."""
        if recording.stage != Stage.PLACED:
            return
        assert recording.transcript and recording.name_plan and recording.placement_plan

        stem = Path(recording.name_plan.filename).stem
        parent_id = recording.placement_plan.target_folder_id
        cfg = self.settings.transcripts

        if cfg.location == "subfolder":
            existing = self.drive.find_child_by_name(parent_id, cfg.subfolder_name)
            parent_id = (
                existing.id
                if existing
                else self.drive.create_folder(cfg.subfolder_name, parent_id).id
            )

        renderers = {
            "md": lambda: render_markdown(
                recording,
                recording.transcript,  # type: ignore[arg-type]
                title=stem,
                heading_interval_s=cfg.timestamp_heading_interval_s,
                include_timestamps=cfg.include_timestamps,
            ),
            "srt": lambda: render_srt(recording.transcript),  # type: ignore[arg-type]
            "vtt": lambda: render_vtt(recording.transcript),  # type: ignore[arg-type]
            "txt": lambda: recording.transcript.text,  # type: ignore[union-attr]
        }
        for fmt in cfg.formats:
            if fmt in renderers:
                self._upload_text(renderers[fmt](), f"{stem}.{fmt}", parent_id)

        if cfg.write_sidecar:
            meta_parent = self._ensure_folder(
                cfg.sidecar_folder, recording.placement_plan.target_folder_id
            )
            self._upload_text(render_sidecar(recording), f"{stem}.json", meta_parent)

        recording.transition_to(Stage.PUBLISHED)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _place_in_review(self, recording: Recording, classification: Classification) -> None:
        """Move an uncertain recording to _Needs Review with a marked name.

        Named with the best guess plus the low-confidence marker, so a human
        can see what the system thought without opening the sidecar. Never
        filed silently — see ADR-007.
        """
        if not self.review_folder_id or not recording.drive_file_id:
            return
        name_plan = build_name(
            recording,
            classification,
            self.taxonomy,
            self.settings.naming,
            existing_names={f.name for f in self.drive.list_children(self.review_folder_id)},
            is_low_confidence=True,
        )
        recording.name_plan = name_plan
        renamed = self.drive.rename(recording.drive_file_id, name_plan.filename)
        moved = self.drive.move(renamed.id, self.review_folder_id, recording.drive_parent_id)
        recording.drive_file_id = moved.id
        recording.drive_parent_id = self.review_folder_id

        # The sidecar goes with it, so the reviewer can see the reasoning and
        # the evidence quotes.
        if self.settings.transcripts.write_sidecar:
            self._upload_text(
                render_sidecar(recording),
                f"{Path(name_plan.filename).stem}.json",
                self.review_folder_id,
            )

    def _ensure_folder(self, name: str, parent_id: str) -> str:
        existing = self.drive.find_child_by_name(parent_id, name)
        return existing.id if existing else self.drive.create_folder(name, parent_id).id

    def _upload_text(self, content: str, filename: str, parent_id: str) -> None:
        tmp = self.work_dir / filename
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(content, encoding="utf-8")
        try:
            self.drive.upload(tmp, filename, parent_id, mime_type="text/plain")
        finally:
            tmp.unlink(missing_ok=True)
