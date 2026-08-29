"""End-to-end pipeline behaviour, and the safety guarantees around it."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from pipeline.classify import ClassificationService
from pipeline.models import Stage
from pipeline.orchestration import PipelineRunner
from pipeline.placement import FolderResolver
from pipeline.storage import DriveWriteRefused

RECORDINGS_ROOT = "5. AS Chemistry/1. Notes, Assignments & Class Recordings"


@pytest.fixture
def runner_factory(settings, taxonomy, drive, stub_asr, tmp_path):
    def _make(fixture_name: str, *, use_llm: bool = False) -> PipelineRunner:
        subject = settings.subjects[0]
        return PipelineRunner(
            settings=settings,
            subject=subject,
            taxonomy=taxonomy,
            drive=drive,
            asr=stub_asr(fixture_name),
            classifier=ClassificationService(
                taxonomy, settings.classify, subject.retrieval, use_llm=use_llm
            ),
            resolver=FolderResolver(
                drive, taxonomy, subject.placement, recordings_root_id=RECORDINGS_ROOT
            ),
            review_folder_id="_Needs Review",
            work_dir=tmp_path / "work",
        )

    return _make


# ── The happy path ────────────────────────────────────────────────────────────


def test_confident_lecture_is_filed_end_to_end(runner_factory, recording_factory, drive):
    recording = recording_factory(
        "GMT20260914-090114_Recording.mp4",
        recorded_at=datetime(2026, 9, 14, 9, 1),
        meeting_topic="Chemistry Class",
    )
    result = runner_factory("lecture_halogenoalkanes").process(recording)

    assert result.final_stage == Stage.DONE
    assert result.error is None
    assert result.node_id == "chem.15_1"
    assert result.filename == "Sep, 14 - Halogenoalkanes.mp4"

    chapter = Path(drive.root) / RECORDINGS_ROOT / "9. Halogenoalkanes (Chapter 16)"
    assert (chapter / "Sep, 14 - Halogenoalkanes.mp4").exists()


def test_transcript_artifacts_land_beside_the_recording(
    runner_factory, recording_factory, drive
):
    recording = recording_factory(
        "raw.mp4", recorded_at=datetime(2026, 9, 14, 9, 1), meeting_topic="Chemistry Class"
    )
    runner_factory("lecture_halogenoalkanes").process(recording)

    chapter = Path(drive.root) / RECORDINGS_ROOT / "9. Halogenoalkanes (Chapter 16)"
    assert (chapter / "Sep, 14 - Halogenoalkanes.md").exists()
    assert (chapter / "Sep, 14 - Halogenoalkanes.srt").exists()
    assert (chapter / ".lecture-meta" / "Sep, 14 - Halogenoalkanes.json").exists()


def test_sidecar_records_provenance(runner_factory, recording_factory, drive):
    import json

    recording = recording_factory(
        "raw.mp4", recorded_at=datetime(2026, 9, 14, 9, 1), meeting_topic="Chemistry Class"
    )
    runner_factory("lecture_halogenoalkanes").process(recording)

    sidecar = (
        Path(drive.root)
        / RECORDINGS_ROOT
        / "9. Halogenoalkanes (Chapter 16)"
        / ".lecture-meta"
        / "Sep, 14 - Halogenoalkanes.json"
    )
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    # Enough to reclassify later without re-transcribing, and to audit the
    # decision a year from now.
    assert payload["recording"]["original_filename"] == "raw.mp4"
    assert payload["classification"]["primary"]["node_id"] == "chem.15_1"
    assert payload["transcript"]["segment_count"] > 0
    assert payload["placement"]["resolution_method"] == "exact"


def test_original_file_leaves_the_inbox(runner_factory, recording_factory, drive):
    recording = recording_factory("raw.mp4", meeting_topic="Chemistry Class")
    runner_factory("lecture_halogenoalkanes").process(recording)
    assert not (Path(drive.root) / "_Inbox" / "raw.mp4").exists()


# ── The review path ───────────────────────────────────────────────────────────


def test_uncertain_lecture_goes_to_review_not_a_chapter(
    runner_factory, recording_factory, drive
):
    recording = recording_factory(
        "raw.mp4", recorded_at=datetime(2026, 8, 10, 14, 2), meeting_topic="Chemistry Class"
    )
    result = runner_factory("lecture_relative_atomic_mass").process(recording)

    assert result.final_stage == Stage.NEEDS_REVIEW
    review = Path(drive.root) / "_Needs Review"
    assert (review / "Aug, 10 - Relative Atomic Mass (review).mp4").exists()
    # Emphatically not filed into a chapter -- any chapter. The fixture spans
    # three, so naming the one it came closest to would under-test this.
    root = Path(drive.root) / RECORDINGS_ROOT
    assert not list(root.glob("*/Aug, 10*.mp4"))


def test_review_item_carries_its_reasoning(runner_factory, recording_factory, drive):
    """A reviewer needs to see what the classifier thought and why."""
    import json

    recording = recording_factory(
        "raw.mp4", recorded_at=datetime(2026, 8, 10, 14, 2), meeting_topic="Chemistry Class"
    )
    runner_factory("lecture_relative_atomic_mass").process(recording)

    sidecar = Path(drive.root) / "_Needs Review" / "Aug, 10 - Relative Atomic Mass (review).json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["classification"]["reasoning"]
    assert payload["classification"]["primary"]["node_id"]


# ── State machine ─────────────────────────────────────────────────────────────


def test_illegal_transitions_are_refused(recording_factory):
    recording = recording_factory("raw.mp4")
    assert recording.stage == Stage.DISCOVERED
    with pytest.raises(ValueError, match="illegal transition"):
        recording.transition_to(Stage.PUBLISHED)


def test_stages_are_idempotent(runner_factory, recording_factory):
    """Re-running a completed stage is a no-op, which is what makes crash
    recovery a matter of restarting rather than unwinding."""
    recording = recording_factory("raw.mp4", meeting_topic="Chemistry Class")
    runner = runner_factory("lecture_halogenoalkanes")
    runner.process(recording)
    assert recording.stage == Stage.DONE

    for step in (runner.fetch, runner.transcribe, runner.classify, runner.place):
        step(recording)
    assert recording.stage == Stage.DONE


# ── Safety ────────────────────────────────────────────────────────────────────


def test_dry_run_makes_no_changes(settings, taxonomy, stub_asr, tmp_path, recording_factory):
    """The rehearsal mode: log every intended change, perform none."""
    from pipeline.storage.local_drive import LocalDriveAdapter

    subject = settings.subjects[0]
    root = tmp_path / "drive2"
    (root / RECORDINGS_ROOT / "9. Halogenoalkanes (Chapter 16)").mkdir(parents=True)
    (root / "_Inbox").mkdir(parents=True)
    (root / "_Inbox" / "raw.mp4").write_text("x", encoding="utf-8")
    dry_drive = LocalDriveAdapter(root, dry_run=True)

    runner = PipelineRunner(
        settings=settings,
        subject=subject,
        taxonomy=taxonomy,
        drive=dry_drive,
        asr=stub_asr("lecture_halogenoalkanes"),
        classifier=ClassificationService(
            taxonomy, settings.classify, subject.retrieval, use_llm=False
        ),
        resolver=FolderResolver(
            dry_drive, taxonomy, subject.placement, recordings_root_id=RECORDINGS_ROOT
        ),
        review_folder_id=None,
        work_dir=tmp_path / "work2",
    )
    recording = recording_factory("raw.mp4", meeting_topic="Chemistry Class")
    recording.drive_file_id = "_Inbox/raw.mp4"
    runner.process(recording)

    # Intent recorded, nothing moved.
    assert any(e["action"] == "rename" for e in dry_drive.audit)
    assert (root / "_Inbox" / "raw.mp4").exists()


def test_local_adapter_refuses_to_escape_its_root(drive):
    with pytest.raises(DriveWriteRefused):
        drive.get_file("../../etc/passwd")


def test_drive_port_exposes_no_delete():
    """Enforced by the interface, not by convention. See ADR-009."""
    from pipeline.storage import DrivePort

    assert not any("delete" in name.lower() for name in dir(DrivePort))


def test_every_mutation_is_audited(runner_factory, recording_factory, drive):
    recording = recording_factory("raw.mp4", meeting_topic="Chemistry Class")
    runner_factory("lecture_halogenoalkanes").process(recording)
    actions = [e["action"] for e in drive.audit]
    assert "rename" in actions and "move" in actions
    renames = [e for e in drive.audit if e["action"] == "rename"]
    # Before and after both recorded, so the change is reversible.
    assert renames[0]["before"] and renames[0]["after"]
