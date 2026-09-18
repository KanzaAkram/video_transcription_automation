"""Preview one Drive video through the production path. No rename.

    python scripts/preview_m4a.py \
        --name "2026-09-14 19.08.53 RA A2 Chemistry M_J 27"

Finds the .mp4, extracts audio, transcribes with faster-whisper base, classifies
with OpenAI against the matching syllabus, and prints the generated filename.
Drive is not written.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline.config import load_secrets  # noqa: E402
from pipeline.m4a.preview import DEFAULT_FOLDER_ID, DEFAULT_STEM, preview_one  # noqa: E402
from pipeline.storage.google_drive import GoogleDriveAdapter  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder-id", default=DEFAULT_FOLDER_ID)
    parser.add_argument("--name", default=DEFAULT_STEM)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    secrets = load_secrets()
    secrets.require_google()
    if not secrets.openai_api_key:
        print("ERROR: OPENAI_API_KEY is not set.", flush=True)
        return 2
    drive = GoogleDriveAdapter(
        secrets,
        root_folder_id=args.folder_id,
        shared_drive_id=secrets.google_shared_drive_id,
        dry_run=True,
        confine_to_root=True,
    )
    cache_dir = REPO_ROOT / "media_cache" / "prod_preview"
    print(
        preview_one(
            drive,
            args.folder_id,
            args.name,
            secrets=secrets,
            repo_root=REPO_ROOT,
            cache_dir=cache_dir,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
