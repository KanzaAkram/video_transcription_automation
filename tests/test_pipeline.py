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


def test_uncertain_lecture_is_filed_on_best_result(
    runner_factory, recording_factory, drive
):
    """Low confidence is logged, but the best classification is still filed."""
    recording = recording_factory(
        "raw.mp4", recorded_at=datetime(2026, 8, 10, 14, 2), meeting_topic="Chemistry Class"
    )
    result = runner_factory("lecture_relative_atomic_mass").process(recording)

    assert result.final_stage == Stage.DONE
    assert result.node_id
    assert result.filename
    assert result.confidence is not None
    review = Path(drive.root) / "_Needs Review"
    assert not list(review.glob("*.mp4"))
    root = Path(drive.root) / RECORDINGS_ROOT
    matches = list(root.glob(f"*/{result.filename}"))
    assert len(matches) == 1


def test_filed_item_carries_its_reasoning(runner_factory, recording_factory, drive):
    """Confidence no longer blocks filing; the sidecar still records why."""
    import json

    recording = recording_factory(
        "raw.mp4", recorded_at=datetime(2026, 8, 10, 14, 2), meeting_topic="Chemistry Class"
    )
    result = runner_factory("lecture_relative_atomic_mass").process(recording)
    assert result.final_stage == Stage.DONE

    sidecars = list((Path(drive.root) / RECORDINGS_ROOT).glob("*/.lecture-meta/*.json"))
    assert sidecars
    payload = json.loads(sidecars[0].read_text(encoding="utf-8"))
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


def test_process_drive_file_renames_in_place(
    settings, taxonomy, stub_asr, tmp_path, recording_factory, drive
):
    """DriveFile → Recording → process(): same file, new name, original folder."""
    subject = settings.subjects[0]
    runner = PipelineRunner(
        settings=settings,
        subject=subject,
        taxonomy=taxonomy,
        drive=drive,
        asr=stub_asr("lecture_halogenoalkanes"),
        classifier=ClassificationService(
            taxonomy, settings.classify, subject.retrieval, use_llm=False
        ),
        resolver=FolderResolver(
            drive, taxonomy, subject.placement, recordings_root_id=RECORDINGS_ROOT
        ),
        review_folder_id="_Needs Review",
        work_dir=tmp_path / "work-rename-only",
        rename_only=True,
    )
    recording_factory(
        "Copy of Sep, 9 - A2 Chemistry M_J 27.mp4",
        meeting_topic="Chemistry Class",
    )
    file = drive.get_file("_Inbox/Copy of Sep, 9 - A2 Chemistry M_J 27.mp4")
    result = runner.process_drive_file(file)

    assert result.error is None
    assert result.final_stage == Stage.DONE
    assert result.filename == "Sep, 9 - Halogenoalkanes.mp4"
    inbox = Path(drive.root) / "_Inbox"
    assert (inbox / "Sep, 9 - Halogenoalkanes.mp4").exists()
    assert not (inbox / "Copy of Sep, 9 - A2 Chemistry M_J 27.mp4").exists()
    chapter = Path(drive.root) / RECORDINGS_ROOT / "9. Halogenoalkanes (Chapter 16)"
    assert not (chapter / "Sep, 9 - Halogenoalkanes.mp4").exists()
    actions = [e["action"] for e in drive.audit]
    assert "rename" in actions
    assert "move" not in actions
    assert "upload" not in actions
    assert "create_folder" not in actions


def test_new_recording_is_filed_into_taxonomy_hierarchy(
    settings, taxonomy, stub_asr, tmp_path, recording_factory, drive
):
    """Classify → find/create folders → move → rename. Other files stay put."""
    from pipeline.config import NamingSettings, PlacementSettings
    from pipeline.placement import hierarchy_labels

    subject = settings.subjects[0].model_copy(
        update={"placement": PlacementSettings(allow_create_folders=True)}
    )
    settings = settings.model_copy(
        update={
            "naming": NamingSettings(
                template="{serial} - {month_abbr} {day} - {subtopic}{ext}"
            ),
            "transcripts": settings.transcripts.model_copy(
                update={"formats": [], "write_sidecar": False}
            ),
        }
    )
    recordings_root = Path(drive.root) / "r"
    recordings_root.mkdir()
    labels = hierarchy_labels(taxonomy, "chem.15_1", subject_name=subject.name)
    sibling_folder = recordings_root / "untouched extra folder"
    sibling_folder.mkdir()

    # One video already in the chapter folder → this recording is serial 2.
    leaf = recordings_root
    for part in labels.folder_names():
        leaf = leaf / part
    leaf.mkdir(parents=True)
    (leaf / "older.mp4").write_text("x", encoding="utf-8")
    other_inbox = Path(drive.root) / "_Inbox" / "Aug, 1 - leave me.mp4"
    other_inbox.write_text("x", encoding="utf-8")

    runner = PipelineRunner(
        settings=settings,
        subject=subject,
        taxonomy=taxonomy,
        drive=drive,
        asr=stub_asr("lecture_halogenoalkanes"),
        classifier=ClassificationService(
            taxonomy, settings.classify, subject.retrieval, use_llm=False
        ),
        resolver=FolderResolver(
            drive, taxonomy, subject.placement, recordings_root_id="r"
        ),
        review_folder_id="_Needs Review",
        work_dir=tmp_path / "work-hierarchy",
    )
    recording_factory(
        "Copy of Sep, 9.mp4",
        recorded_at=datetime(2026, 9, 9, 14, 0),
        meeting_topic="Chemistry Class",
        id="hier-001",
    )
    file = drive.get_file("_Inbox/Copy of Sep, 9.mp4")
    result = runner.process_drive_file(file)

    assert result.error is None
    assert result.final_stage == Stage.DONE
    assert result.filename == "2 - Sep 9 - Halogenoalkanes.mp4"
    assert (leaf / "2 - Sep 9 - Halogenoalkanes.mp4").exists()
    assert (leaf / "older.mp4").exists()
    assert other_inbox.exists()
    assert sibling_folder.is_dir()
    inbox = Path(drive.root) / "_Inbox"
    assert not (inbox / "Copy of Sep, 9.mp4").exists()
    assert not (inbox / "2 - Sep 9 - Halogenoalkanes.mp4").exists()

    actions = [e["action"] for e in drive.audit]
    assert actions.index("move") < actions.index("rename")
    assert "create_folder" not in actions  # leaf already existed


def test_hierarchy_creates_folders_only_when_missing(
    settings, taxonomy, stub_asr, tmp_path, recording_factory, drive
):
    from pipeline.config import NamingSettings, PlacementSettings

    subject = settings.subjects[0].model_copy(
        update={"placement": PlacementSettings(allow_create_folders=True)}
    )
    settings = settings.model_copy(
        update={
            "naming": NamingSettings(
                template="{serial} - {month_abbr} {day} - {subtopic}{ext}"
            ),
            "transcripts": settings.transcripts.model_copy(
                update={"formats": [], "write_sidecar": False}
            ),
        }
    )
    runner = PipelineRunner(
        settings=settings,
        subject=subject,
        taxonomy=taxonomy,
        drive=drive,
        asr=stub_asr("lecture_halogenoalkanes"),
        classifier=ClassificationService(
            taxonomy, settings.classify, subject.retrieval, use_llm=False
        ),
        resolver=FolderResolver(
            drive, taxonomy, subject.placement, recordings_root_id="r"
        ),
        review_folder_id="_Needs Review",
        work_dir=tmp_path / "work-create",
    )
    (Path(drive.root) / "r").mkdir()
    recording = recording_factory(
        "raw.mp4",
        recorded_at=datetime(2026, 9, 14, 9, 1),
        meeting_topic="Chemistry Class",
    )
    result = runner.process(recording)
    assert result.error is None
    assert result.filename == "1 - Sep 14 - Halogenoalkanes.mp4"
    leaf = (
        Path(drive.root)
        / "r"
        / "AS Chemistry"
        / "9. Halogenoalkanes (Chapter 16)"
        / "1 - Sep 14 - Halogenoalkanes.mp4"
    )
    assert leaf.exists()
    assert not (leaf.parent / "Halogenoalkanes").exists()
    assert "create_folder" in [e["action"] for e in drive.audit]


def test_transcribe_writes_transcript_cache(runner_factory, recording_factory):
    recording = recording_factory(
        "raw.mp4", recorded_at=datetime(2026, 9, 14, 9, 1), meeting_topic="Chemistry Class"
    )
    runner = runner_factory("lecture_halogenoalkanes")
    runner.process(recording)
    cache = Path(runner.work_dir) / f"{recording.id}.transcript.json"
    assert cache.exists()
    payload = cache.read_text(encoding="utf-8")
    assert "halogenoalkane" in payload.lower()


def test_cached_transcript_skips_asr_and_is_filed(
    settings, taxonomy, tmp_path, recording_factory, drive
):
    """A previous Whisper run is reused: classify → chapter folder → rename."""
    from pipeline.asr.base import ASRProvider
    from pipeline.config import NamingSettings, PlacementSettings
    from pipeline.placement import hierarchy_labels

    class BoomASR(ASRProvider):
        name = "boom"

        def is_available(self) -> tuple[bool, str]:
            return True, ""

        def transcribe(self, audio_path, hints):
            raise AssertionError("ASR must not run when a transcript cache exists")

    subject = settings.subjects[0].model_copy(
        update={"placement": PlacementSettings(allow_create_folders=True)}
    )
    settings = settings.model_copy(
        update={
            "naming": NamingSettings(
                template="{serial} - {month_abbr} {day} - {subtopic}{ext}"
            ),
            "transcripts": settings.transcripts.model_copy(
                update={"formats": [], "write_sidecar": False}
            ),
        }
    )
    (Path(drive.root) / "r").mkdir()
    work = tmp_path / "work-cached"
    work.mkdir()
    recording = recording_factory(
        "Copy of Sep, 9.mp4",
        recorded_at=datetime(2026, 9, 9, 14, 0),
        meeting_topic="Chemistry Class",
        id="cached-001",
    )
    fixture = Path(__file__).parent / "fixtures" / "lecture_halogenoalkanes.json"
    (work / "cached-001.transcript.json").write_text(
        fixture.read_text(encoding="utf-8"), encoding="utf-8"
    )

    runner = PipelineRunner(
        settings=settings,
        subject=subject,
        taxonomy=taxonomy,
        drive=drive,
        asr=BoomASR(),
        classifier=ClassificationService(
            taxonomy, settings.classify, subject.retrieval, use_llm=False
        ),
        resolver=FolderResolver(
            drive, taxonomy, subject.placement, recordings_root_id="r"
        ),
        review_folder_id="_Needs Review",
        work_dir=work,
    )
    result = runner.process(recording)

    assert result.error is None
    assert result.final_stage == Stage.DONE
    assert result.filename == "1 - Sep 9 - Halogenoalkanes.mp4"
    labels = hierarchy_labels(taxonomy, "chem.15_1", subject_name=subject.name)
    leaf = Path(drive.root) / "r"
    for part in labels.folder_names():
        leaf = leaf / part
    assert (leaf / "1 - Sep 9 - Halogenoalkanes.mp4").exists()
    assert "download" not in [e["action"] for e in drive.audit]


def test_every_mutation_is_audited(runner_factory, recording_factory, drive):
    recording = recording_factory("raw.mp4", meeting_topic="Chemistry Class")
    runner_factory("lecture_halogenoalkanes").process(recording)
    actions = [e["action"] for e in drive.audit]
    assert "rename" in actions and "move" in actions
    renames = [e for e in drive.audit if e["action"] == "rename"]
    # Before and after both recorded, so the change is reversible.
    assert renames[0]["before"] and renames[0]["after"]
