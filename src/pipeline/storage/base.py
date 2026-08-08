"""The storage port.

Everything downstream talks to this interface, never to Google's client
directly. Two implementations exist: `GoogleDriveAdapter` for the real thing
and `LocalDriveAdapter` backed by a directory tree.

The local adapter is not just a test double — it is what lets the entire
pipeline be built, demonstrated and rehearsed before any credentials are
handed over, and it makes a safe place to watch what the system would do to a
mirrored copy of your real folder structure. See ADR-014.

Note what is *absent*: there is no `delete`. The pipeline moves and renames.
That is enforced by the interface, not by convention.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


class DriveWriteRefused(RuntimeError):
    """A write was blocked by a safety guard.

    Raised when an operation would touch something outside the configured root
    folder, or when dry-run mode is active. Distinct from an API error so
    callers can tell "I was not allowed to" from "it did not work".
    """


@dataclass
class DriveFile:
    id: str
    name: str
    mime_type: str
    parents: list[str] = field(default_factory=list)
    size_bytes: int | None = None
    md5_checksum: str | None = None
    created_time: datetime | None = None
    modified_time: datetime | None = None
    web_view_link: str | None = None

    @property
    def is_folder(self) -> bool:
        return self.mime_type == "application/vnd.google-apps.folder"

    @property
    def extension(self) -> str:
        return Path(self.name).suffix.lower()

    @property
    def stem(self) -> str:
        return Path(self.name).stem


@dataclass
class DriveChange:
    """One entry from the changes feed.

    Drive push notifications say only "something happened" — you must call
    changes.list from a saved page token to find out what. See
    docs/ARCHITECTURE.md § 3.2.
    """

    file_id: str
    removed: bool
    file: DriveFile | None = None


class DrivePort(ABC):
    """Storage operations the pipeline needs. Nothing more."""

    # ── Reading ──────────────────────────────────────────────────────────────

    @abstractmethod
    def get_file(self, file_id: str) -> DriveFile: ...

    @abstractmethod
    def list_children(self, folder_id: str, *, folders_only: bool = False) -> list[DriveFile]: ...

    @abstractmethod
    def find_child_by_name(self, folder_id: str, name: str) -> DriveFile | None:
        """Exact-name lookup among a folder's direct children."""

    @abstractmethod
    def download(self, file_id: str, dest: Path) -> Path: ...

    @abstractmethod
    def path_of(self, file_id: str) -> str:
        """Human-readable path, for logs and reports."""

    # ── Writing ──────────────────────────────────────────────────────────────
    # Rename and move are metadata-only operations. A 2 GB lecture is
    # repositioned in the tree with a few hundred bytes of traffic, and its
    # file ID is preserved so existing share links keep working. See ADR-003.

    @abstractmethod
    def rename(self, file_id: str, new_name: str) -> DriveFile: ...

    @abstractmethod
    def move(
        self, file_id: str, new_parent_id: str, old_parent_id: str | None = None
    ) -> DriveFile: ...

    @abstractmethod
    def upload(
        self, local_path: Path, name: str, parent_id: str, mime_type: str | None = None
    ) -> DriveFile: ...

    @abstractmethod
    def create_folder(self, name: str, parent_id: str) -> DriveFile: ...

    # ── Change detection ─────────────────────────────────────────────────────

    @abstractmethod
    def get_start_page_token(self) -> str: ...

    @abstractmethod
    def list_changes(self, page_token: str) -> tuple[list[DriveChange], str]:
        """Return changes since `page_token` and the next token to persist."""

    # ── Introspection ────────────────────────────────────────────────────────

    @abstractmethod
    def is_descendant_of(self, file_id: str, ancestor_id: str) -> bool:
        """Whether `file_id` sits under `ancestor_id`.

        The confinement check. Every mutation calls this against the configured
        root before touching anything.
        """
