"""Google Drive implementation of DrivePort.

The API-call shapes are complete and correct. Authentication is the only part
that cannot be finished without knowing whether the target account is Google
Workspace or personal Gmail — see docs/SETUP.md § 0, which explains why that
distinction changes the design rather than just the credentials.

Three safety properties are enforced here rather than left to callers:

  1. Every mutation is confined to the configured root folder.
  2. There is no delete operation, at all.
  3. Every mutation is audited with before/after state, so it can be reversed.

Requires the `cloud` extra: pip install -e ".[cloud]"
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pipeline.config import Secrets
from pipeline.storage.base import DriveChange, DriveFile, DrivePort, DriveWriteRefused

log = logging.getLogger(__name__)

FOLDER_MIME = "application/vnd.google-apps.folder"

#: Full Drive scope is required, not optional. `drive.file` grants access only
#: to files the application itself created, which by definition excludes an
#: existing folder tree. See ADR-009 for the mitigations that follow from this.
SCOPES = ["https://www.googleapis.com/auth/drive"]

#: Requested on every read so Shared Drives are visible. Omitting these is the
#: most common reason a Shared Drive appears empty through the API.
SHARED_DRIVE_PARAMS: dict[str, Any] = {
    "supportsAllDrives": True,
    "includeItemsFromAllDrives": True,
}

FILE_FIELDS = (
    "id,name,mimeType,parents,size,md5Checksum,createdTime,modifiedTime,webViewLink"
)

CHANGE_FIELDS = (
    "nextPageToken, newStartPageToken, "
    f"changes(fileId, removed, file({FILE_FIELDS}))"
)


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _to_drive_file(payload: dict[str, Any]) -> DriveFile:
    return DriveFile(
        id=payload["id"],
        name=payload.get("name", ""),
        mime_type=payload.get("mimeType", ""),
        parents=payload.get("parents", []) or [],
        size_bytes=int(payload["size"]) if payload.get("size") else None,
        md5_checksum=payload.get("md5Checksum"),
        created_time=_parse_time(payload.get("createdTime")),
        modified_time=_parse_time(payload.get("modifiedTime")),
        web_view_link=payload.get("webViewLink"),
    )


def build_credentials(secrets: Secrets):
    """Construct credentials for the configured auth mode.

    The three modes exist because of a Google constraint worth stating plainly:
    a service account has no Drive storage quota of its own, so it cannot own
    files in a personal My Drive. Uploads fail with a storage-quota error even
    though the human's Drive has plenty of space. Hence Shared Drives, or
    impersonation, or user OAuth. See docs/SETUP.md § 0.
    """
    mode = secrets.google_auth_mode

    if mode == "service_account":
        from google.oauth2 import service_account

        if not secrets.google_service_account_file:
            raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_FILE is not set")
        return service_account.Credentials.from_service_account_file(
            str(secrets.google_service_account_file), scopes=SCOPES
        )

    if mode == "impersonation":
        from google.oauth2 import service_account

        if not secrets.google_service_account_file or not secrets.google_impersonate_subject:
            raise RuntimeError(
                "impersonation mode needs GOOGLE_SERVICE_ACCOUNT_FILE and "
                "GOOGLE_IMPERSONATE_SUBJECT"
            )
        base = service_account.Credentials.from_service_account_file(
            str(secrets.google_service_account_file), scopes=SCOPES
        )
        # Domain-wide delegation: act as a real user, so files are owned by
        # that user and the quota problem does not arise.
        return base.with_subject(secrets.google_impersonate_subject)

    if mode == "oauth_user":
        from google.oauth2.credentials import Credentials

        if not secrets.google_oauth_refresh_token:
            raise RuntimeError("oauth_user mode needs GOOGLE_OAUTH_REFRESH_TOKEN")
        return Credentials(
            token=None,
            refresh_token=secrets.google_oauth_refresh_token,
            client_id=secrets.google_oauth_client_id,
            client_secret=secrets.google_oauth_client_secret,
            token_uri="https://oauth2.googleapis.com/token",
            scopes=SCOPES,
        )

    raise RuntimeError(
        f"auth mode {mode!r} does not use Google credentials. "
        "Use LocalDriveAdapter for local mode."
    )


class GoogleDriveAdapter(DrivePort):
    def __init__(
        self,
        secrets: Secrets,
        *,
        root_folder_id: str,
        shared_drive_id: str | None = None,
        dry_run: bool = False,
        confine_to_root: bool = True,
    ) -> None:
        from googleapiclient.discovery import build

        self.secrets = secrets
        self.root_folder_id = root_folder_id
        self.shared_drive_id = shared_drive_id
        self.dry_run = dry_run
        self.confine_to_root = confine_to_root
        self.audit: list[dict[str, Any]] = []

        self._service = build(
            "drive", "v3", credentials=build_credentials(secrets), cache_discovery=False
        )
        #: file_id -> parent chain, so confinement checks do not re-walk the
        #: tree on every mutation.
        self._ancestry_cache: dict[str, list[str]] = {}

    # ── Guards ───────────────────────────────────────────────────────────────

    def _guard(self, action: str, file_id: str, **detail: Any) -> None:
        """Refuse anything outside the root, and honour dry-run mode."""
        if self.confine_to_root and not self.is_descendant_of(file_id, self.root_folder_id):
            raise DriveWriteRefused(
                f"refusing to {action} {file_id}: outside configured root "
                f"{self.root_folder_id}"
            )
        self.audit.append(
            {"action": action, "file_id": file_id, "dry_run": self.dry_run, **detail}
        )
        if self.dry_run:
            log.info("dry-run: would %s %s %s", action, file_id, detail)

    # ── Reading ──────────────────────────────────────────────────────────────

    def get_file(self, file_id: str) -> DriveFile:
        payload = (
            self._service.files()
            .get(fileId=file_id, fields=FILE_FIELDS, supportsAllDrives=True)
            .execute()
        )
        return _to_drive_file(payload)

    def list_children(self, folder_id: str, *, folders_only: bool = False) -> list[DriveFile]:
        query = f"'{folder_id}' in parents and trashed = false"
        if folders_only:
            query += f" and mimeType = '{FOLDER_MIME}'"

        out: list[DriveFile] = []
        page_token: str | None = None
        while True:
            params: dict[str, Any] = {
                "q": query,
                "fields": f"nextPageToken, files({FILE_FIELDS})",
                "pageSize": 1000,
                **SHARED_DRIVE_PARAMS,
            }
            if self.shared_drive_id:
                params.update(driveId=self.shared_drive_id, corpora="drive")
            if page_token:
                params["pageToken"] = page_token
            response = self._service.files().list(**params).execute()
            out.extend(_to_drive_file(f) for f in response.get("files", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                return out

    def find_child_by_name(self, folder_id: str, name: str) -> DriveFile | None:
        # Escaping matters: folder names here contain apostrophes and commas.
        escaped = name.replace("\\", "\\\\").replace("'", "\\'")
        params: dict[str, Any] = {
            "q": f"'{folder_id}' in parents and name = '{escaped}' and trashed = false",
            "fields": f"files({FILE_FIELDS})",
            "pageSize": 1,
            **SHARED_DRIVE_PARAMS,
        }
        if self.shared_drive_id:
            params.update(driveId=self.shared_drive_id, corpora="drive")
        files = self._service.files().list(**params).execute().get("files", [])
        return _to_drive_file(files[0]) if files else None

    def download(self, file_id: str, dest: Path) -> Path:
        from googleapiclient.http import MediaIoBaseDownload

        dest.parent.mkdir(parents=True, exist_ok=True)
        request = self._service.files().get_media(fileId=file_id, supportsAllDrives=True)
        with dest.open("wb") as fh:
            downloader = MediaIoBaseDownload(fh, request, chunksize=8 * 1024 * 1024)
            done = False
            while not done:
                _status, done = downloader.next_chunk()
        return dest

    def path_of(self, file_id: str) -> str:
        parts: list[str] = []
        current: str | None = file_id
        seen: set[str] = set()
        while current and current not in seen:
            seen.add(current)
            f = self.get_file(current)
            parts.append(f.name)
            if current == self.root_folder_id:
                break
            current = f.parents[0] if f.parents else None
        return "/" + "/".join(reversed(parts))

    # ── Writing ──────────────────────────────────────────────────────────────

    def rename(self, file_id: str, new_name: str) -> DriveFile:
        before = self.get_file(file_id)
        self._guard("rename", file_id, before=before.name, after=new_name)
        if self.dry_run:
            return before
        payload = (
            self._service.files()
            .update(fileId=file_id, body={"name": new_name}, fields=FILE_FIELDS,
                    supportsAllDrives=True)
            .execute()
        )
        return _to_drive_file(payload)

    def move(
        self, file_id: str, new_parent_id: str, old_parent_id: str | None = None
    ) -> DriveFile:
        """Reparent. Metadata-only — the file content is never transferred."""
        before = self.get_file(file_id)
        old_parent_id = old_parent_id or (before.parents[0] if before.parents else None)
        self._guard("move", file_id, before=old_parent_id, after=new_parent_id)
        if self.dry_run:
            return before
        payload = (
            self._service.files()
            .update(
                fileId=file_id,
                addParents=new_parent_id,
                removeParents=old_parent_id,
                fields=FILE_FIELDS,
                supportsAllDrives=True,
            )
            .execute()
        )
        # The parent changed, so any cached ancestry for this file is stale.
        self._ancestry_cache.pop(file_id, None)
        return _to_drive_file(payload)

    def upload(
        self, local_path: Path, name: str, parent_id: str, mime_type: str | None = None
    ) -> DriveFile:
        from googleapiclient.http import MediaFileUpload

        self._guard("upload", parent_id, after=name)
        if self.dry_run:
            return DriveFile(id="dry-run", name=name,
                             mime_type=mime_type or "application/octet-stream",
                             parents=[parent_id])
        media = MediaFileUpload(
            str(local_path),
            mimetype=mime_type,
            # Resumable, so a dropped connection resumes rather than restarts.
            resumable=local_path.stat().st_size > 5 * 1024 * 1024,
        )
        payload = (
            self._service.files()
            .create(
                body={"name": name, "parents": [parent_id]},
                media_body=media,
                fields=FILE_FIELDS,
                supportsAllDrives=True,
            )
            .execute()
        )
        return _to_drive_file(payload)

    def create_folder(self, name: str, parent_id: str) -> DriveFile:
        self._guard("create_folder", parent_id, after=name)
        if self.dry_run:
            return DriveFile(id="dry-run", name=name, mime_type=FOLDER_MIME,
                             parents=[parent_id])
        payload = (
            self._service.files()
            .create(
                body={"name": name, "mimeType": FOLDER_MIME, "parents": [parent_id]},
                fields=FILE_FIELDS,
                supportsAllDrives=True,
            )
            .execute()
        )
        return _to_drive_file(payload)

    # ── Change detection ─────────────────────────────────────────────────────

    def get_start_page_token(self) -> str:
        params: dict[str, Any] = {"supportsAllDrives": True}
        if self.shared_drive_id:
            params["driveId"] = self.shared_drive_id
        return self._service.changes().getStartPageToken(**params).execute()["startPageToken"]

    def list_changes(self, page_token: str) -> tuple[list[DriveChange], str]:
        """Walk the changes feed from `page_token`.

        This is the part people miss: a Drive push notification carries no
        information about *what* changed. It is a doorbell. You still have to
        call this to find out. See docs/ARCHITECTURE.md § 3.2.
        """
        changes: list[DriveChange] = []
        token = page_token
        while True:
            params: dict[str, Any] = {
                "pageToken": token,
                "fields": CHANGE_FIELDS,
                **SHARED_DRIVE_PARAMS,
            }
            if self.shared_drive_id:
                params.update(driveId=self.shared_drive_id)
            response = self._service.changes().list(**params).execute()
            for change in response.get("changes", []):
                changes.append(
                    DriveChange(
                        file_id=change["fileId"],
                        removed=change.get("removed", False),
                        file=_to_drive_file(change["file"]) if change.get("file") else None,
                    )
                )
            if "newStartPageToken" in response:
                return changes, response["newStartPageToken"]
            token = response["nextPageToken"]

    def watch_changes(self, channel_id: str, webhook_url: str, token: str, page_token: str) -> dict:
        """Register a push channel on the changes feed.

        Channels expire — commonly within a day. The scheduler renews them well
        before expiry; a lapsed channel is the single most common cause of
        "it worked for a week and then stopped".
        """
        body = {
            "id": channel_id,
            "type": "web_hook",
            "address": webhook_url,
            "token": token,
        }
        params: dict[str, Any] = {"pageToken": page_token, "body": body,
                                  "supportsAllDrives": True}
        if self.shared_drive_id:
            params["driveId"] = self.shared_drive_id
        return self._service.changes().watch(**params).execute()

    def stop_watch(self, channel_id: str, resource_id: str) -> None:
        self._service.channels().stop(
            body={"id": channel_id, "resourceId": resource_id}
        ).execute()

    # ── Introspection ────────────────────────────────────────────────────────

    def is_descendant_of(self, file_id: str, ancestor_id: str) -> bool:
        if file_id == ancestor_id:
            return True
        chain = self._ancestry_cache.get(file_id)
        if chain is None:
            chain = []
            current: str | None = file_id
            seen: set[str] = set()
            while current and current not in seen:
                seen.add(current)
                try:
                    f = self.get_file(current)
                except Exception:  # noqa: BLE001 - a missing parent means "not a descendant"
                    break
                current = f.parents[0] if f.parents else None
                if current:
                    chain.append(current)
            self._ancestry_cache[file_id] = chain
        return ancestor_id in chain

    # NOTE: there is deliberately no delete method. The pipeline moves and
    # renames; it never removes. See ADR-009.
