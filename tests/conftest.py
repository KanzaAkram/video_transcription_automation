from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline.asr.stub import StubProvider  # noqa: E402
from pipeline.config import ASRSettings, Settings, load_settings  # noqa: E402
from pipeline.models import Recording, Source  # noqa: E402
from pipeline.storage.local_drive import LocalDriveAdapter  # noqa: E402
from pipeline.syllabus import Taxonomy, load_taxonomy  # noqa: E402

FIXTURES = REPO_ROOT / "tests" / "fixtures"
RECORDINGS_ROOT = "5. AS Chemistry/1. Notes, Assignments & Class Recordings"

# Mirrors the real Drive structure, including the deliberate mismatch between
# the leading ordinal and the "(Chapter N)" suffix.
CHAPTER_FOLDERS = [
    "0. Introduction",
    "1. Atomic Structure (Chapter 1)",
    "2. Electrons in Atoms (Chapter 2)",
    "3. Chemical Bonding (Chapter 4)",
    "4. Moles and Stoichiometry (Chapter 3)",
    "5. States of Matter (Chapter 5)",
    "6. Intro to Organic Chemistry (Chapter 14)",
    "7. Hydrocarbons (Chapter 15)",
    "8. Alcohols (Chapter 17)",
    "9. Halogenoalkanes (Chapter 16)",
    "10. Carboxylic Acids & Esters (Chapter 17)",
]


@pytest.fixture
def settings() -> Settings:
    return load_settings(REPO_ROOT / "config" / "settings.example.yaml")


@pytest.fixture
def taxonomy() -> Taxonomy:
    return load_taxonomy(REPO_ROOT / "config" / "syllabi" / "as-chemistry-9701.yaml")


@pytest.fixture
def drive(tmp_path: Path) -> LocalDriveAdapter:
    root = tmp_path / "drive"
    for folder in CHAPTER_FOLDERS:
        (root / RECORDINGS_ROOT / folder).mkdir(parents=True, exist_ok=True)
    (root / "_Inbox").mkdir(parents=True, exist_ok=True)
    (root / "_Needs Review").mkdir(parents=True, exist_ok=True)
    return LocalDriveAdapter(root)


@pytest.fixture
def stub_asr():
    def _make(fixture_name: str) -> StubProvider:
        return StubProvider(ASRSettings(), fixture=FIXTURES / f"{fixture_name}.json")

    return _make


@pytest.fixture
def recording_factory(drive: LocalDriveAdapter):
    def _make(filename: str, *, recorded_at: datetime | None = None, **kwargs) -> Recording:
        (Path(drive.root) / "_Inbox" / filename).write_text("x", encoding="utf-8")
        source_id = f"_Inbox/{filename}"
        return Recording(
            id=kwargs.pop("id", "test-001"),
            source=Source.DRIVE,
            source_id=source_id,
            subject_id=kwargs.pop("subject_id", "as_chemistry"),
            original_filename=filename,
            recorded_at=recorded_at or datetime(2026, 8, 10, 14, 0, 0),
            duration_s=kwargs.pop("duration_s", 3600),
            drive_file_id=source_id,
            drive_parent_id="_Inbox",
            **kwargs,
        )

    return _make
