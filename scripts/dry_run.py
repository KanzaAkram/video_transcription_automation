#!/usr/bin/env python
"""End-to-end pipeline run with no credentials.

Builds a miniature copy of the AS Chemistry tree on local disk, drops an
unclassified recording into the inbox, and runs the full pipeline over it:
transcribe (from a fixture) -> classify -> name -> place -> publish.

The point is to make the whole design inspectable before any Zoom or Google
account is connected. Everything you see it do here is what it will do to real
Drive folders, through the same code path — only the storage adapter differs.

    python scripts/dry_run.py            keyword classifier, no API key needed
    python scripts/dry_run.py --ai       the configured provider (ChatGPT by default)
    python scripts/dry_run.py --keep     leave .dryrun/ in place afterwards
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# The Windows console is cp1252, and this script prints box-drawing rules. The
# first thing it printed was therefore a UnicodeEncodeError traceback, which is
# a poor first impression from the command whose whole job is to show that the
# pipeline works. Elsewhere the convention is to keep console output ASCII —
# see `scripts/rename_folder.py` — and only this script needs the exception.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from pipeline.asr.stub import StubProvider  # noqa: E402
from pipeline.classify import ClassificationService  # noqa: E402
from pipeline.config import Settings, load_secrets, load_settings  # noqa: E402
from pipeline.models import Recording, Source, Stage  # noqa: E402
from pipeline.orchestration import PipelineRunner  # noqa: E402
from pipeline.placement import FolderResolver  # noqa: E402
from pipeline.storage.local_drive import LocalDriveAdapter  # noqa: E402
from pipeline.syllabus import load_taxonomy  # noqa: E402

DRYRUN_DIR = REPO_ROOT / ".dryrun"
TAXONOMY = REPO_ROOT / "config" / "syllabi" / "as-chemistry-9701.yaml"
FIXTURES = REPO_ROOT / "tests" / "fixtures"

# Two lectures, chosen to exercise both outcomes:
#
#   * halogenoalkanes  — one topic throughout, distinctive vocabulary.
#     Retrieval alone is confident, so it gets filed automatically.
#   * relative atomic mass — genuinely split, opening with several minutes
#     recapping the previous chapter. The keyword baseline is not confident
#     enough to file it, so it goes to review. That is the safety mechanism
#     working, not a failure.
LECTURES = [
    {
        "id": "dryrun-halogenoalkanes",
        "fixture": FIXTURES / "lecture_halogenoalkanes.json",
        "filename": "GMT20260914-090114_Recording_1920x1080.mp4",
        "recorded_at": datetime(2026, 9, 14, 9, 1, 0),
        "meeting_topic": "Chemistry Class",
        "note": "single topic throughout",
    },
    {
        "id": "dryrun-relative-atomic-mass",
        "fixture": FIXTURES / "lecture_relative_atomic_mass.json",
        "filename": "GMT20260810-140203_Recording_1920x1080.mp4",
        "recorded_at": datetime(2026, 8, 10, 14, 2, 0),
        "meeting_topic": "Chemistry Class",
        "note": "opens with a recap of the previous chapter",
    },
]

# Mirrors the structure in your Drive screenshots, including the ordinal /
# chapter-number mismatch that the taxonomy has to cope with.
SUBJECT_ROOT = "5. AS Chemistry"
RECORDINGS_ROOT = f"{SUBJECT_ROOT}/1. Notes, Assignments & Class Recordings"

CHAPTERS = [
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

FOLDERS = [
    *(f"{RECORDINGS_ROOT}/{chapter}" for chapter in CHAPTERS),
    f"{SUBJECT_ROOT}/2. Syllabus",
    "_Inbox",
    "_Needs Review",
]

# A couple of files already in place, so collision handling has something to
# collide with and the "before" tree looks realistic.
EXISTING = [
    f"{RECORDINGS_ROOT}/1. Atomic Structure (Chapter 1)/Aug, 5 - Atomic Structure.mp4",
    f"{RECORDINGS_ROOT}/1. Atomic Structure (Chapter 1)/0. Atomic Structure.pdf",
]


def build_tree() -> None:
    if DRYRUN_DIR.exists():
        shutil.rmtree(DRYRUN_DIR)
    for folder in FOLDERS:
        (DRYRUN_DIR / folder).mkdir(parents=True, exist_ok=True)
    for path in EXISTING:
        (DRYRUN_DIR / path).write_text("(placeholder)", encoding="utf-8")

    # Placeholder media in the inbox. The stub ASR provider reads the fixture
    # transcript instead of touching these, so their contents do not matter.
    for lecture in LECTURES:
        (DRYRUN_DIR / "_Inbox" / lecture["filename"]).write_text(
            "(placeholder video)", encoding="utf-8"
        )


def print_tree(drive: LocalDriveAdapter, title: str) -> None:
    print(f"\n{title}")
    print("─" * len(title))
    print(drive.tree() or "(empty)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ai",
        "--llm",
        dest="ai",
        action="store_true",
        help="classify with the provider in settings (classify.provider, ChatGPT by "
        "default) instead of the keyword baseline. Needs that provider's API key.",
    )
    parser.add_argument("--keep", action="store_true", help="keep .dryrun/ afterwards")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)-7s %(name)s: %(message)s",
    )

    for lecture in LECTURES:
        if not lecture["fixture"].exists():
            print(f"missing fixture transcript: {lecture['fixture']}", file=sys.stderr)
            return 1

    print("Lecture Pipeline — dry run")
    print("No credentials used. Drive is a local directory; ASR reads a fixture.\n")

    build_tree()

    settings: Settings = load_settings(REPO_ROOT / "config" / "settings.example.yaml")
    subject = settings.subjects[0] if settings.subjects else None
    if subject is None:
        print("no subject configured in settings.example.yaml", file=sys.stderr)
        return 1

    taxonomy = load_taxonomy(TAXONOMY)
    drive = LocalDriveAdapter(DRYRUN_DIR)

    print_tree(drive, "BEFORE")

    resolver = FolderResolver(
        drive, taxonomy, subject.placement, recordings_root_id=RECORDINGS_ROOT
    )
    secrets = load_secrets()
    classifier = ClassificationService(
        taxonomy,
        settings.classify,
        subject.retrieval,
        use_llm=args.ai,
        api_key=(
            secrets.openai_api_key
            if settings.classify.provider == "openai"
            else secrets.anthropic_api_key
        ),
    )

    threshold = settings.classify.review_threshold
    models = {
        "openai": settings.classify.openai_model,
        "anthropic": settings.classify.model,
    }
    which = (
        f"{settings.classify.provider} ({models.get(settings.classify.provider, '-')})"
        if args.ai
        else "keyword baseline (no API key needed)"
    )
    print(f"\nClassifier: {which}")
    print(f"Review threshold: {threshold:.2f} — below this, a lecture goes to a human.\n")

    results = []
    for lecture in LECTURES:
        runner = PipelineRunner(
            settings=settings,
            subject=subject,
            taxonomy=taxonomy,
            drive=drive,
            asr=StubProvider(settings.asr, fixture=lecture["fixture"]),
            classifier=classifier,
            resolver=resolver,
            review_folder_id="_Needs Review",
            work_dir=DRYRUN_DIR / ".work",
        )

        source_id = f"_Inbox/{lecture['filename']}"
        recording = Recording(
            id=lecture["id"],
            source=Source.DRIVE,
            source_id=source_id,
            subject_id=subject.id,
            original_filename=lecture["filename"],
            # Deliberately uninformative: the whole point is that
            # classification comes from the transcript, not the title.
            meeting_topic=lecture["meeting_topic"],
            recorded_at=lecture["recorded_at"],
            duration_s=3600,
            teacher="Ms Rahman",
            drive_file_id=source_id,
            drive_parent_id="_Inbox",
        )

        print(f"── {lecture['filename']}")
        print(f"   Zoom topic {recording.meeting_topic!r} carries no information; "
              f"fixture: {lecture['note']}")

        result = runner.process(recording)
        results.append((lecture, result))

        if result.error:
            print(f"   ERROR       {result.error}")
        else:
            print(f"   syllabus    {result.node_id}")
            if result.confidence is not None:
                verdict = "filed" if result.confidence >= threshold else "-> review"
                print(f"   confidence  {result.confidence:.2f}  {verdict}")
            print(f"   filename    {result.filename}")
            print(f"   folder      {result.folder_path or '_Needs Review'}")
            if result.details.get("reasoning"):
                print(f"   reasoning   {result.details['reasoning']}")
        print(f"   timings     {json.dumps(result.stage_timings)}")
        print()

    print_tree(drive, "AFTER")

    print("\nDrive operations performed")
    print("──────────────────────────")
    for entry in drive.audit:
        before = entry.get("before", "")
        after = entry.get("after", "")
        arrow = f"{before} -> {after}" if before else after
        print(f"  {entry['action']:<14} {arrow}")

    print("\nSUMMARY")
    print("───────")
    ok = True
    for lecture, result in results:
        if result.error:
            ok = False
            print(f"  FAIL   {lecture['id']}: {result.error}")
        elif result.final_stage == Stage.DONE:
            print(f"  filed  {lecture['id']}: {result.filename}")
        elif result.final_stage == Stage.NEEDS_REVIEW:
            print(f"  review {lecture['id']}: below the {threshold:.2f} threshold")
        else:
            ok = False
            print(f"  FAIL   {lecture['id']}: unexpected stage {result.final_stage}")

    if any(r.final_stage == Stage.NEEDS_REVIEW for _, r in results):
        print(
            "\n  A lecture routed to review is the safety mechanism working, not a\n"
            "  failure: a misfiled lecture is worse than an unfiled one, because it\n"
            "  is invisible. To resolve it, a teacher moves the file into the right\n"
            "  chapter folder in Drive and the pipeline learns from the correction.\n"
            "  The keyword baseline is deliberately conservative — run with --ai\n"
            "  for the production classifier."
        )

    if not args.keep:
        shutil.rmtree(DRYRUN_DIR, ignore_errors=True)
    else:
        print(f"\nTree kept at {DRYRUN_DIR}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
