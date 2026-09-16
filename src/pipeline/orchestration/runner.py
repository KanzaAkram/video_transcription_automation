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

import json
import logging
import time
from pathlib import Path

from pipeline.asr import ASRProvider
from pipeline.classify import ClassificationService
from pipeline.config import Settings, SubjectSettings
from pipeline.ingest.drive import recording_from_drive_file
from pipeline.media import extract_audio
from pipeline.media.audio import FFmpegMissing
from pipeline.m4a.discover import find_existing_video
from pipeline.m4a.routing import detect_subject
from pipeline.models import (
    Classification,
    PlacementPlan,
    ProcessingResult,
    Recording,
    Stage,
    Transcript,
    TranscriptionHints,
    TranscriptSegment,
)
from pipeline.naming import build_name
from pipeline.placement import FolderResolver, hierarchy_labels
from pipeline.publish import render_markdown, render_sidecar, render_srt, render_vtt
from pipeline.storage import DriveFile, DrivePort
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
        rename_only: bool = False,
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
        #: Rename the Drive file in place and stop. No move, no folder
        #: creation, no transcript upload. Used when the recording is already
        #: in the right folder and only the name is wrong.
        self.rename_only = rename_only

    # ── Entry point ──────────────────────────────────────────────────────────

    def process(self, recording: Recording) -> ProcessingResult:
        timings: dict[str, float] = {}
        try:
            # A previous Whisper run is enough: classify still needs the text,
            # but fetch/audio/ASR must not run again.
            self._hydrate_from_transcript_cache(recording)
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

    def process_drive_file(self, file: DriveFile) -> ProcessingResult:
        """The Drive → Recording → process() connection.

        Class and subject are read from the filename when present, then handed
        to the OpenAI classifier as weak context. Taxonomy still comes from the
        runner — the matching syllabus is chosen when the runner is built.
        """
        recording = recording_from_drive_file(file, subject_id=self.subject.id)
        try:
            hint = detect_subject(file.name)
        except (ValueError, FileNotFoundError):
            hint = None
        if hint is not None:
            recording.meeting_topic = f"{hint.class_name} {hint.subject}"
        return self.process(recording)

    def process_date_folder(self, folder_id: str, stem: str | None = None) -> ProcessingResult:
        """Find the .mp4 in a date folder (ignore chat.txt) and process it."""
        video = find_existing_video(self.drive, folder_id, stem)
        return self.process_drive_file(video)

    # ── Stages ───────────────────────────────────────────────────────────────

    def _local_source(self, recording: Recording) -> Path:
        """Where the downloaded media lives. Derived, not remembered.

        Deriving it from the recording id means a resumed job finds the file a
        previous worker downloaded, instead of re-fetching it.
        """
        suffix = Path(recording.original_filename).suffix or ".mp4"
        return self.work_dir / f"{recording.id}{suffix}"

    def _transcript_cache_path(self, recording: Recording) -> Path:
        return self.work_dir / f"{recording.id}.transcript.json"

    def _load_cached_transcript(self, recording: Recording) -> Transcript | None:
        path = self._transcript_cache_path(recording)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        try:
            return Transcript.model_validate(data)
        except Exception:  # noqa: BLE001 - fixture-style JSON is also accepted
            segments = [TranscriptSegment.model_validate(s) for s in data["segments"]]
            return Transcript(
                segments=segments,
                language=data.get("language", "en"),
                duration_s=data.get("duration_s", 0.0),
                model=data.get("model", "cached"),
                provider=data.get("provider", "cache"),
                runtime_s=data.get("runtime_s"),
            )

    def _save_cached_transcript(self, recording: Recording) -> None:
        if recording.transcript is None:
            return
        path = self._transcript_cache_path(recording)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(recording.transcript.model_dump_json(indent=1), encoding="utf-8")

    def _hydrate_from_transcript_cache(self, recording: Recording) -> None:
        """Skip download and Whisper when this recording was already transcribed.

        Walks DISCOVERED → TRANSCRIBED so fetch/prepare/transcribe become
        no-ops. Classification still runs: the cache is the transcript, not
        the filing decision.
        """
        if recording.stage != Stage.DISCOVERED:
            return
        cached = self._load_cached_transcript(recording)
        if cached is None:
            return
        recording.transcript = cached
        recording.transition_to(Stage.FETCHED)
        recording.transition_to(Stage.AUDIO_READY)
        recording.transition_to(Stage.TRANSCRIBED)
        log.info("reusing cached transcript for %s; skipping fetch and ASR", recording.id)

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
            except Exception as exc:
                # A stub ASR provider reads a fixture and needs no real audio,
                # so a missing ffmpeg must not block the dry run. faster-whisper
                # decodes the original container through PyAV, so a missing
                # ffmpeg binary is also not fatal for a downloaded mp4.
                if self.asr.name == "local_stub" or isinstance(exc, FFmpegMissing):
                    audio = source
                else:
                    raise
        recording.audio_path = str(audio)
        recording.transition_to(Stage.AUDIO_READY)

    def transcribe(self, recording: Recording) -> None:
        if recording.stage != Stage.AUDIO_READY:
            return
        cached = self._load_cached_transcript(recording)
        if cached is not None:
            recording.transcript = cached
            recording.transition_to(Stage.TRANSCRIBED)
            log.info("reusing cached transcript for %s", recording.id)
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
        self._save_cached_transcript(recording)

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

        # Confidence is reported, never a stop. A low score still files the
        # best node the classifier produced so folder creation and rename run.
        if self.classifier.needs_review(classification):
            primary = classification.primary
            log.info(
                "recording %s below review threshold (%s, confidence %s); "
                "filing best result",
                recording.id,
                classification.abstain_reason or "low confidence or coverage",
                f"{primary.confidence:.2f}" if primary else "n/a",
            )

    def plan_name(self, recording: Recording) -> None:
        if recording.stage != Stage.CLASSIFIED:
            return
        assert recording.classification is not None

        primary = recording.classification.primary
        use_hierarchy = (
            not self.rename_only
            and self.subject.placement.allow_create_folders
            and primary is not None
        )
        serial: int | None = None
        labels = None
        planned_path = ""
        if use_hierarchy:
            labels = hierarchy_labels(
                self.taxonomy, primary.node_id, subject_name=self.subject.name
            )
            # Preview only: do not create folders before the Detected: print.
            target = self.resolver.ensure_hierarchy(
                primary.node_id, subject_name=self.subject.name, create=False
            )
            planned_path = target.folder_path
            existing, serial = self._sibling_videos(
                target.folder_id, exclude_id=recording.drive_file_id
            )
        elif self.rename_only and recording.drive_parent_id:
            # Collisions are against siblings in the current folder — we are
            # not moving this file into the chapter folder.
            target = None
            existing = {f.name for f in self.drive.list_children(recording.drive_parent_id)}
        else:
            target = self.resolver.resolve(primary.node_id) if primary else None
            existing = (
                {f.name for f in self.drive.list_children(target.folder_id)}
                if target is not None and target.resolved
                else set()
            )
        recording.name_plan = build_name(
            recording,
            recording.classification,
            self.taxonomy,
            self.settings.naming,
            existing_names=existing,
            is_low_confidence=False,
            serial=serial,
        )
        if labels is not None:
            self._print_detected(labels, planned_path, recording.name_plan.filename)
        recording.transition_to(Stage.NAMED)
        self._target = target

    def place(self, recording: Recording) -> None:
        if recording.stage != Stage.NAMED:
            return
        assert recording.name_plan is not None

        # Always chain through the returned handle rather than reusing the
        # original id. Google Drive ids survive a rename, but that is a
        # property of Google's adapter, not of the interface — the local
        # adapter keys on path. Threading the handle keeps the runner correct
        # against both.
        handle = recording.drive_file_id  # type: ignore[assignment]
        if self.rename_only:
            renamed = self.drive.rename(handle, recording.name_plan.filename)
            recording.drive_file_id = renamed.id
            parent_id = recording.drive_parent_id or ""
            recording.placement_plan = PlacementPlan(
                target_folder_id=parent_id,
                target_folder_path=parent_id,
                node_id=(
                    recording.classification.primary.node_id
                    if recording.classification and recording.classification.primary
                    else ""
                ),
                resolution_method="in_place",
                resolution_confidence=1.0,
            )
            recording.transition_to(Stage.PLACED)
            return

        primary = (
            recording.classification.primary if recording.classification else None
        )
        if self.subject.placement.allow_create_folders and primary is not None:
            # Find or create missing folders now — after the Detected: print.
            target = self.resolver.ensure_hierarchy(
                primary.node_id, subject_name=self.subject.name, create=True
            )
        else:
            target = getattr(self, "_target", None)

        if target is None or not target.resolved:
            raise RuntimeError(
                f"no Drive folder resolved for recording {recording.id}: "
                f"{target.note if target else 'resolver not run'}"
            )

        if recording.drive_parent_id != target.folder_id:
            moved = self.drive.move(handle, target.folder_id, recording.drive_parent_id)
            handle = moved.id
            recording.drive_parent_id = target.folder_id

        occupied = {
            f.name for f in self.drive.list_children(target.folder_id) if f.id != handle
        }
        if recording.name_plan.filename in occupied:
            raise RuntimeError(
                f"refusing to overwrite existing file {recording.name_plan.filename!r} "
                f"in {target.folder_path}"
            )

        current = self.drive.get_file(handle)
        if current.name != recording.name_plan.filename:
            renamed = self.drive.rename(handle, recording.name_plan.filename)
            handle = renamed.id

        recording.drive_file_id = handle
        recording.drive_parent_id = target.folder_id
        recording.placement_plan = PlacementPlan(
            target_folder_id=target.folder_id,
            target_folder_path=target.folder_path,
            node_id=primary.node_id if primary else "",
            resolution_method=target.method,
            resolution_confidence=target.confidence,
        )
        recording.transition_to(Stage.PLACED)

    def publish(self, recording: Recording) -> None:
        """Upload transcript artifacts alongside the recording."""
        if recording.stage != Stage.PLACED:
            return
        if self.rename_only:
            # Metadata rename is the whole write. Do not upload transcripts
            # or create a sidecar folder alongside the recording.
            recording.transition_to(Stage.PUBLISHED)
            return
        cfg = self.settings.transcripts
        if not cfg.formats and not cfg.write_sidecar:
            recording.transition_to(Stage.PUBLISHED)
            return
        assert recording.transcript and recording.name_plan and recording.placement_plan

        stem = Path(recording.name_plan.filename).stem
        parent_id = recording.placement_plan.target_folder_id

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

    def _sibling_videos(
        self, folder_id: str | None, *, exclude_id: str | None
    ) -> tuple[set[str], int]:
        """Names already in `folder_id`, and the next per-folder video serial.

        Serial counts video files in this folder only — never a global counter.
        The recording being processed is excluded so a re-run does not skip a
        number because it counted itself.
        """
        if not folder_id:
            return set(), 1
        video_exts = {ext.lower() for ext in self.settings.ingest.video_extensions}
        names: set[str] = set()
        video_count = 0
        for child in self.drive.list_children(folder_id):
            if child.id == exclude_id:
                continue
            names.add(child.name)
            if not child.is_folder and child.extension in video_exts:
                video_count += 1
        return names, video_count + 1

    def _print_detected(self, labels, folder_path: str, filename: str) -> None:
        print(
            "Detected:\n"
            f"Class: {labels.class_name}\n"
            f"Subject: {labels.subject}\n"
            f"Chapter: {labels.chapter}\n"
            f"Taught topic: {labels.subtopic}\n"
            "\n"
            f"Target folder: {folder_path}\n"
            f"Final filename: {filename}",
            flush=True,
        )

    def _upload_text(self, content: str, filename: str, parent_id: str) -> None:
        tmp = self.work_dir / filename
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(content, encoding="utf-8")
        try:
            self.drive.upload(tmp, filename, parent_id, mime_type="text/plain")
        finally:
            tmp.unlink(missing_ok=True)
