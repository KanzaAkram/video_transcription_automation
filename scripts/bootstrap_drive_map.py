#!/usr/bin/env python
"""Step 2 of connecting the real accounts: resolve syllabus nodes to Drive folders.

Read-only. Walks the taxonomy and reports, for every node, which Drive folder
it resolves to and how the match was made. Run this before anything writes
anything — it is the cheapest way to confirm the taxonomy's folder names match
what is actually in Drive, and it catches the ordinal/chapter-number confusion
that would otherwise file lectures into the wrong chapter.

    python scripts/bootstrap_drive_map.py --dry-run
    python scripts/bootstrap_drive_map.py --subject as_chemistry
    python scripts/bootstrap_drive_map.py --local .dryrun    # against a local tree
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from _common import load_settings_with_fallback, select_subjects  # noqa: E402

from pipeline.config import load_secrets  # noqa: E402
from pipeline.placement import FolderResolver  # noqa: E402
from pipeline.storage import DrivePort  # noqa: E402
from pipeline.syllabus import load_taxonomy  # noqa: E402

# How much to trust each resolution method. Anything other than `exact`
# deserves a human glance before the pipeline starts writing.
METHOD_NOTE = {
    "exact": "",
    "normalised": "matched after stripping the ordinal / chapter suffix",
    "fuzzy": "FUZZY MATCH — confirm this is right",
    "created": "folder was created",
    "review": "unresolved; lectures for this node will go to review",
}


def build_drive(args, secrets, subject) -> tuple[DrivePort, str]:
    if args.local:
        from pipeline.storage.local_drive import LocalDriveAdapter

        root = Path(args.local).resolve()
        recordings_root = subject.recordings_subfolder or ""
        subject_root = args.subject_folder or "5. AS Chemistry"
        return LocalDriveAdapter(root), f"{subject_root}/{recordings_root}".rstrip("/")

    from pipeline.storage.google_drive import GoogleDriveAdapter

    secrets.require_google()
    root_id = subject.drive_root_folder_id or secrets.google_drive_root_folder_id
    if not root_id:
        raise SystemExit(
            f"no Drive root folder id for subject {subject.id!r}. "
            "Set drive_root_folder_id in settings.yaml or "
            "GOOGLE_DRIVE_ROOT_FOLDER_ID in .env."
        )
    drive = GoogleDriveAdapter(
        secrets,
        root_folder_id=root_id,
        shared_drive_id=secrets.google_shared_drive_id,
        dry_run=True,  # this script never writes
    )
    recordings_root = root_id
    if subject.recordings_subfolder:
        child = drive.find_child_by_name(root_id, subject.recordings_subfolder)
        if child is None:
            raise SystemExit(
                f"could not find {subject.recordings_subfolder!r} under the "
                f"subject root. Check recordings_subfolder in settings.yaml."
            )
        recordings_root = child.id
    return drive, recordings_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", help="only this subject id")
    parser.add_argument(
        "--dry-run", action="store_true", help="accepted for symmetry; always read-only"
    )
    parser.add_argument("--local", help="resolve against a local directory instead of Drive")
    parser.add_argument("--subject-folder", help="subject folder name when using --local")
    args = parser.parse_args()

    secrets = load_secrets()
    settings, settings_path, fell_back = load_settings_with_fallback(secrets)
    if fell_back:
        print(f"note: {secrets.settings_file} not found; using {settings_path}\n")
    subjects = select_subjects(settings, args.subject)

    exit_code = 0
    for subject in subjects:
        taxonomy = load_taxonomy(subject.taxonomy_file)
        drive, recordings_root = build_drive(args, secrets, subject)
        resolver = FolderResolver(
            drive, taxonomy, subject.placement, recordings_root_id=recordings_root
        )

        print(f"\n{subject.name} ({subject.id})")
        print("=" * (len(subject.name) + len(subject.id) + 3))

        results = resolver.resolve_all()
        unresolved, inexact = 0, 0

        for node_id, result in results.items():
            node = taxonomy.require(node_id)
            if result.resolved:
                status = "->"
                detail = result.folder_path
            else:
                status = "!!"
                detail = result.note
                unresolved += 1
            note = METHOD_NOTE.get(result.method, result.method)
            if result.method not in ("exact",):
                inexact += 1
            line = f"  {status} {node.title:<34} {detail}"
            print(line if not note else f"{line}\n       {note}")

        pending = [n for n in taxonomy.all_nodes() if n.drive_folder_name is None]
        print(f"\n  {len(results)} nodes with a folder name: "
              f"{len(results) - unresolved} resolved, {unresolved} unresolved, "
              f"{inexact} not an exact match")
        if pending:
            print(f"  {len(pending)} node(s) have no Drive folder yet "
                  f"(lectures on those topics will route to review)")
        if unresolved or inexact:
            exit_code = 1

    if exit_code:
        print(
            "\nSome nodes did not resolve exactly. Fix by editing "
            "`drive_folder_name` in the taxonomy YAML to match the folder name "
            "in Drive exactly, then re-run. Do not proceed to live processing "
            "until this is clean."
        )
    else:
        print("\nAll nodes resolved exactly. Safe to proceed.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
