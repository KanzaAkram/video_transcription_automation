"""Find the recording media in a Drive date folder. Ignore chat and sidecars."""

from __future__ import annotations

from pathlib import Path

from pipeline.ingest.drive import NON_RECORDING_NAMES, is_recording_file
from pipeline.storage.base import DriveFile, DrivePort

AUDIO_EXT = ".m4a"
VIDEO_EXT = ".mp4"


def _matches(name: str, stem: str) -> bool:
    lowered = name.lower()
    target = stem.lower()
    return target in lowered or Path(name).stem.lower() == target


def _is_m4a(file: DriveFile) -> bool:
    return file.extension == AUDIO_EXT or file.name.lower().endswith(AUDIO_EXT)


def _is_video(file: DriveFile) -> bool:
    if file.is_folder or file.name.lower() in NON_RECORDING_NAMES:
        return False
    return file.extension == VIDEO_EXT or file.name.lower().endswith(VIDEO_EXT)


def find_video_in(children: list[DriveFile], *, stem: str | None = None) -> DriveFile:
    """The one .mp4 in a date folder. chat.txt and other sidecars are ignored."""
    videos = [child for child in children if _is_video(child) and is_recording_file(child)]
    if stem:
        named = [item for item in videos if _matches(item.name, stem)]
        if named:
            videos = named
    if not videos:
        names = ", ".join(child.name for child in children) or "(empty folder)"
        raise FileNotFoundError(f"no .mp4 file found. Children: {names}")
    if len(videos) > 1:
        raise FileNotFoundError(
            "multiple .mp4 files found: " + ", ".join(item.name for item in videos)
        )
    return videos[0]


def find_existing_video(drive: DrivePort, folder_id: str, stem: str | None = None) -> DriveFile:
    """Return the lecture .mp4 inside `folder_id`. Never selects chat.txt."""
    children = drive.list_children(folder_id)
    try:
        named = find_named_item(children, stem) if stem else None
    except FileNotFoundError:
        named = None
    if named is not None:
        if _is_video(named):
            return named
        if named.is_folder:
            return find_video_in(drive.list_children(named.id), stem=stem)
    return find_video_in(children, stem=stem)


def find_named_item(children: list[DriveFile], stem: str) -> DriveFile:
    matches = [child for child in children if _matches(child.name, stem)]
    if not matches:
        names = ", ".join(child.name for child in children) or "(empty folder)"
        raise FileNotFoundError(f"no Drive item matching {stem!r}. Children: {names}")
    if len(matches) > 1:
        m4as = [item for item in matches if _is_m4a(item)]
        if len(m4as) == 1:
            return m4as[0]
        raise FileNotFoundError(
            f"multiple Drive items match {stem!r}: "
            + ", ".join(item.name for item in matches)
        )
    return matches[0]


def find_m4a_in(children: list[DriveFile], *, stem: str | None = None) -> DriveFile:
    audio = [child for child in children if _is_m4a(child)]
    if stem:
        named = [child for child in audio if _matches(child.name, stem)]
        if named:
            audio = named
    if not audio:
        raise FileNotFoundError("no .m4a file found")
    if len(audio) > 1:
        raise FileNotFoundError(
            "multiple .m4a files found: " + ", ".join(item.name for item in audio)
        )
    return audio[0]


def find_existing_m4a(drive: DrivePort, folder_id: str, stem: str) -> tuple[DriveFile, DriveFile]:
    """Return (named item, .m4a file). The named item may be a folder or the audio itself.

    The recording system's folder is named like the lecture; the audio inside is
    often a generic ``audio*.m4a``. `folder_id` may be that lecture folder.
    """
    children = drive.list_children(folder_id)
    try:
        named = find_named_item(children, stem)
    except FileNotFoundError:
        audio = find_m4a_in(children)
        try:
            named = drive.get_file(folder_id)
        except Exception:  # noqa: BLE001 - LocalDriveAdapter cannot describe its own root
            named = DriveFile(
                id=folder_id,
                name=stem,
                mime_type="application/vnd.google-apps.folder",
            )
        if not _matches(named.name, stem):
            raise FileNotFoundError(
                f"no Drive item matching {stem!r} under {named.name!r}"
            ) from None
        return named, audio
    if _is_m4a(named):
        return named, named
    if named.is_folder:
        return named, find_m4a_in(drive.list_children(named.id))
    sibling_name = f"{Path(named.name).stem}{AUDIO_EXT}"
    sibling = drive.find_child_by_name(folder_id, sibling_name)
    if sibling is not None and _is_m4a(sibling):
        return named, sibling
    parent_children = drive.list_children(folder_id)
    try:
        return named, find_m4a_in(parent_children, stem=stem)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{named.name!r} is not an .m4a and no sibling .m4a was found"
        ) from exc
