"""Preview gpt-audio placement for one Drive recording. No rename, no whisper.

    python scripts/preview_gpt_audio.py \
        --name "2026-09-14 19.08.53 RA A2 Chemistry M_J 27"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline.config import load_secrets, load_settings  # noqa: E402
from pipeline.m4a.discover import find_existing_m4a  # noqa: E402
from pipeline.m4a.gpt_audio import (  # noqa: E402
    DEFAULT_MODEL,
    classify_lecture_audio,
    encode_mp3,
    generated_video_name,
)
from pipeline.m4a.routing import detect_subject  # noqa: E402
from pipeline.m4a.transcribe import files_to_transcribe  # noqa: E402
from pipeline.storage.google_drive import GoogleDriveAdapter  # noqa: E402
from pipeline.syllabus import load_taxonomy  # noqa: E402

DEFAULT_FOLDER_ID = "1KdYxE8YXDaxifYyoGsTNIqsoLaOmFtzF"
DEFAULT_STEM = "2026-09-14 19.08.53 RA A2 Chemistry M_J 27"


def format_preview(
    *,
    model: str,
    named_name: str,
    named_id: str,
    audio_name: str,
    audio_size_mb: float,
    chunks: int,
    fields: dict[str, str],
    filename: str,
) -> str:
    return "\n".join(
        [
            f"MODEL: {model}",
            f"FILE: {named_name}",
            f"FILE ID: {named_id}",
            f"AUDIO FILE: {audio_name}",
            f"AUDIO SIZE: {audio_size_mb:.1f} MB",
            f"CHUNKS: {chunks}",
            "",
            f"CLASS: {fields['class']}",
            f"SUBJECT: {fields['subject']}",
            f"CHAPTER: {fields['chapter']}",
            f"TOPIC: {fields['topic']}",
            f"SUBTOPIC: {fields['subtopic']}",
            "",
            f"GENERATED VIDEO FILENAME: {filename}",
            "",
            "Stopped before any Drive write. Waiting for approval.",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder-id", default=DEFAULT_FOLDER_ID)
    parser.add_argument("--name", default=DEFAULT_STEM)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    secrets = load_secrets()
    secrets.require_google()
    api_key = secrets.openai_api_key
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set.", flush=True)
        return 2

    drive = GoogleDriveAdapter(
        secrets,
        root_folder_id=args.folder_id,
        shared_drive_id=secrets.google_shared_drive_id,
        dry_run=True,
        confine_to_root=True,
    )
    named, audio = find_existing_m4a(drive, args.folder_id, args.name)
    hint = detect_subject(named.name, repo_root=REPO_ROOT)
    taxonomy = load_taxonomy(hint.taxonomy_path)
    settings = load_settings(secrets.settings_file)

    cache_dir = REPO_ROOT / "media_cache" / "m4a"
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / f"{audio.id}.m4a"
    if not local.exists():
        drive.download(audio.id, local)

    pieces = files_to_transcribe(local)
    mp3_paths: list[Path] = []
    for piece, _offset in pieces:
        mp3 = cache_dir / f"{piece.stem}.gpt-audio.mp3"
        if not mp3.exists() or mp3.stat().st_size == 0:
            encode_mp3(piece, mp3)
        mp3_paths.append(mp3)

    fields = classify_lecture_audio(
        mp3_paths,
        taxonomy=taxonomy,
        hint=hint,
        api_key=api_key,
        model=args.model,
    )
    filename = generated_video_name(
        fields,
        hint,
        taxonomy,
        settings.naming,
        display_name=f"{named.name}.mp4",
        audio_id=audio.id,
        parent_id=audio.parents[0] if audio.parents else None,
        existing_names={named.name, audio.name},
    )
    print(
        format_preview(
            model=args.model,
            named_name=named.name,
            named_id=named.id,
            audio_name=audio.name,
            audio_size_mb=local.stat().st_size / (1024 * 1024),
            chunks=len(pieces),
            fields=fields,
            filename=filename,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
