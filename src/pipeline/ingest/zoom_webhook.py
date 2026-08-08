"""Zoom webhook verification and payload parsing.

Two things every Zoom integration must get right, and both are implemented
here rather than left to the caller:

1. **URL validation.** Zoom proves you own the endpoint by sending an
   ``endpoint.url_validation`` event with a ``plainToken``; you must reply with
   an HMAC-SHA256 of it keyed by your webhook secret token.

2. **Request signing.** Every subsequent delivery carries ``x-zm-signature``
   and ``x-zm-request-timestamp``. Both must be checked — the signature for
   authenticity and the timestamp for freshness, since a valid-but-old
   signature is replayable.

An unverified webhook endpoint lets anyone enqueue arbitrary work.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from datetime import datetime
from typing import Any

log = logging.getLogger(__name__)

#: Reject deliveries older than this. Zoom retries within seconds, so a
#: five-minute window is generous while still bounding replay.
MAX_CLOCK_SKEW_S = 300


class ZoomWebhookVerifier:
    def __init__(self, secret_token: str) -> None:
        if not secret_token:
            raise ValueError("Zoom webhook secret token is required")
        self.secret_token = secret_token

    def verify(self, body: bytes, signature: str, timestamp: str) -> bool:
        """Constant-time check of signature and timestamp."""
        if not signature or not timestamp:
            return False
        try:
            age = time.time() - int(timestamp)
        except ValueError:
            return False
        if abs(age) > MAX_CLOCK_SKEW_S:
            log.warning("rejecting Zoom webhook: timestamp %ss out of window", int(age))
            return False

        message = f"v0:{timestamp}:{body.decode('utf-8')}"
        digest = hmac.new(
            self.secret_token.encode(), message.encode(), hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(f"v0={digest}", signature)

    def url_validation_response(self, plain_token: str) -> dict[str, str]:
        """Answer Zoom's endpoint-ownership challenge."""
        encrypted = hmac.new(
            self.secret_token.encode(), plain_token.encode(), hashlib.sha256
        ).hexdigest()
        return {"plainToken": plain_token, "encryptedToken": encrypted}


# ── Payload parsing ───────────────────────────────────────────────────────────


class ZoomRecordingFile:
    """One file from a recording — Zoom produces several per meeting."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.id: str = payload.get("id", "")
        self.file_type: str = payload.get("file_type", "")
        self.file_extension: str = payload.get("file_extension", "").lower()
        self.file_size: int = payload.get("file_size", 0)
        self.download_url: str = payload.get("download_url", "")
        self.recording_type: str = payload.get("recording_type", "")

    @property
    def is_audio_only(self) -> bool:
        return self.file_type == "M4A"

    @property
    def is_video(self) -> bool:
        return self.file_type == "MP4"


class ZoomRecording:
    def __init__(self, payload: dict[str, Any]) -> None:
        obj = payload.get("payload", {}).get("object", {})
        self.uuid: str = obj.get("uuid", "")
        self.meeting_id: str = str(obj.get("id", ""))
        self.topic: str = obj.get("topic", "")
        self.host_email: str = obj.get("host_email", "")
        self.start_time: str = obj.get("start_time", "")
        self.duration_minutes: int = obj.get("duration", 0)
        self.files = [ZoomRecordingFile(f) for f in obj.get("recording_files", [])]
        #: Short-lived JWT for downloading. Present on the webhook payload only.
        self.download_token: str = payload.get("download_token", "")

    @property
    def recorded_at(self) -> datetime:
        if not self.start_time:
            return datetime.utcnow()
        return datetime.fromisoformat(self.start_time.replace("Z", "+00:00"))

    @property
    def duration_s(self) -> float:
        return self.duration_minutes * 60.0

    def best_file_for_transcription(
        self, *, prefer_audio_only: bool = True
    ) -> ZoomRecordingFile | None:
        """Pick the smallest file that contains the audio.

        Zoom produces an audio-only M4A alongside the MP4 at roughly a tenth
        the size. Transcription needs only audio, and Drive repositioning needs
        no content at all, so downloading the video is almost always waste.
        See ADR-003.
        """
        if prefer_audio_only:
            for f in self.files:
                if f.is_audio_only and f.download_url:
                    return f
        for f in self.files:
            if f.is_video and f.download_url:
                return f
        return self.files[0] if self.files else None


def parse_recording_completed(payload: dict[str, Any]) -> ZoomRecording:
    event = payload.get("event", "")
    if event != "recording.completed":
        raise ValueError(f"expected recording.completed, got {event!r}")
    return ZoomRecording(payload)
