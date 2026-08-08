"""A DrivePort backed by a local directory tree.

This is what makes the pipeline runnable with no credentials at all. Point it
at a directory, and every rename, move and upload the pipeline would perform
against Google Drive happens against real files you can inspect.

Two uses:

  * The dry run — demonstrate the whole pipeline before connecting accounts.
  * Rehearsal — mirror your real folder structure locally and watch exactly
    what the system would do to it.

File IDs are stable paths relative to the root, which keeps them readable in
logs. Google's opaque IDs are the only difference callers can observe.
"""

from __future__ import annotations

import hashlib
import mimetypes
import shutil
from datetime import datetime
from pathlib import Path

from pipeline.storage.base import DriveChange, DriveFile, DrivePort, DriveWriteRefused

FOLDER_MIME = "application/vnd.google-apps.folder"


class LocalDriveAdapter(DrivePort):
    def __init__(self, root: Path | str, *, dry_run: bool = False) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run
        #: Every mutation, for the audit trail and the dry-run report.
        self.audit: list[dict[str, str]] = []
        self._change_cursor = 0

    # ── ID <-> path ──────────────────────────────────────────────────────────

    def _to_path(self, file_id: str) -> Path:
        if file_id in ("", "root"):
            return self.root
        path = (self.root / file_id).resolve()
        # Defence in depth: a crafted id must not escape the root.
        if not self._within_root(path):
            raise DriveWriteRefused(f"path {file_id!r} escapes the configured root")
        return path

    def _to_id(self, path: Path) -> str:
        resolved = path.resolve()
        if resolved == self.root:
            return "root"
        return resolved.relative_to(self.root).as_posix()

    def _within_root(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.root)
        except ValueError:
            return False
        return True

    def _to_drive_file(self, path: Path) -> DriveFile:
        stat = path.stat()
        is_dir = path.is_dir()
        return DriveFile(
            id=self._to_id(path),
            name=path.name,
            mime_type=FOLDER_MIME
            if is_dir
            else (mimetypes.guess_type(path.name)[0] or "application/octet-stream"),
            parents=[self._to_id(path.parent)],
            size_bytes=None if is_dir else stat.st_size,
            md5_checksum=None if is_dir else self._md5(path),
            created_time=datetime.fromtimestamp(stat.st_ctime),
            modified_time=datetime.fromtimestamp(stat.st_mtime),
            web_view_link=path.as_uri(),
        )

    @staticmethod
    def _md5(path: Path) -> str | None:
        # Large media files would be slow to hash in full; the first megabyte
        # plus the size is enough to deduplicate re-uploads.
        try:
            h = hashlib.md5()
            with path.open("rb") as fh:
                h.update(fh.read(1024 * 1024))
            h.update(str(path.stat().st_size).encode())
            return h.hexdigest()
        except OSError:
            return None

    def _record(self, action: str, **fields: str) -> None:
        self.audit.append({"action": action, "dry_run": str(self.dry_run), **fields})

    # ── Reading ──────────────────────────────────────────────────────────────

    def get_file(self, file_id: str) -> DriveFile:
        path = self._to_path(file_id)
        if not path.exists():
            raise FileNotFoundError(f"no such file: {file_id}")
        return self._to_drive_file(path)

    def list_children(self, folder_id: str, *, folders_only: bool = False) -> list[DriveFile]:
        path = self._to_path(folder_id)
        if not path.is_dir():
            return []
        out = [
            self._to_drive_file(child)
            for child in sorted(path.iterdir())
            if not (folders_only and not child.is_dir())
        ]
        return out

    def find_child_by_name(self, folder_id: str, name: str) -> DriveFile | None:
        candidate = self._to_path(folder_id) / name
        return self._to_drive_file(candidate) if candidate.exists() else None

    def download(self, file_id: str, dest: Path) -> Path:
        src = self._to_path(file_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return dest

    def path_of(self, file_id: str) -> str:
        return "/" + self._to_id(self._to_path(file_id)).lstrip("/")

    # ── Writing ──────────────────────────────────────────────────────────────

    def rename(self, file_id: str, new_name: str) -> DriveFile:
        path = self._to_path(file_id)
        target = path.parent / new_name
        self._record("rename", file_id=file_id, before=path.name, after=new_name)
        if self.dry_run:
            return self._to_drive_file(path)
        if target.exists() and target != path:
            raise DriveWriteRefused(f"cannot rename onto existing file: {new_name}")
        path.rename(target)
        return self._to_drive_file(target)

    def move(
        self, file_id: str, new_parent_id: str, old_parent_id: str | None = None
    ) -> DriveFile:
        path = self._to_path(file_id)
        new_parent = self._to_path(new_parent_id)
        target = new_parent / path.name
        self._record(
            "move", file_id=file_id, before=self.path_of(file_id), after=self._to_id(target)
        )
        if self.dry_run:
            return self._to_drive_file(path)
        new_parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise DriveWriteRefused(f"destination already occupied: {self._to_id(target)}")
        shutil.move(str(path), str(target))
        return self._to_drive_file(target)

    def upload(
        self, local_path: Path, name: str, parent_id: str, mime_type: str | None = None
    ) -> DriveFile:
        parent = self._to_path(parent_id)
        target = parent / name
        self._record("upload", file_id=self._to_id(target), after=name)
        if self.dry_run:
            return DriveFile(
                id=self._to_id(target),
                name=name,
                mime_type=mime_type or "application/octet-stream",
                parents=[parent_id],
            )
        parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, target)
        return self._to_drive_file(target)

    def create_folder(self, name: str, parent_id: str) -> DriveFile:
        target = self._to_path(parent_id) / name
        self._record("create_folder", file_id=self._to_id(target), after=name)
        if self.dry_run:
            return DriveFile(id=self._to_id(target), name=name, mime_type=FOLDER_MIME,
                             parents=[parent_id])
        target.mkdir(parents=True, exist_ok=True)
        return self._to_drive_file(target)

    # ── Change detection ─────────────────────────────────────────────────────

    def get_start_page_token(self) -> str:
        return "0"

    def list_changes(self, page_token: str) -> tuple[list[DriveChange], str]:
        """Report files modified since the token's timestamp.

        A simplification of Drive's changes feed that is sufficient for local
        rehearsal — enough to exercise the reconciler's logic.
        """
        since = float(page_token or 0)
        changes: list[DriveChange] = []
        newest = since
        for path in self.root.rglob("*"):
            if path.is_dir():
                continue
            mtime = path.stat().st_mtime
            if mtime > since:
                changes.append(
                    DriveChange(file_id=self._to_id(path), removed=False,
                                file=self._to_drive_file(path))
                )
                newest = max(newest, mtime)
        return changes, str(newest)

    # ── Introspection ────────────────────────────────────────────────────────

    def is_descendant_of(self, file_id: str, ancestor_id: str) -> bool:
        try:
            child = self._to_path(file_id).resolve()
            ancestor = self._to_path(ancestor_id).resolve()
        except DriveWriteRefused:
            return False
        return ancestor == child or ancestor in child.parents

    # ── Helpers for tests and the dry run ────────────────────────────────────

    def tree(self, folder_id: str = "root", *, indent: int = 0) -> str:
        """Render the tree, for dry-run before/after output."""
        lines: list[str] = []
        for child in self.list_children(folder_id):
            marker = "/" if child.is_folder else ""
            lines.append("  " * indent + f"{child.name}{marker}")
            if child.is_folder:
                lines.append(self.tree(child.id, indent=indent + 1))
        return "\n".join(line for line in lines if line)
