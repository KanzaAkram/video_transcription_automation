"""Zoom webhook verification.

An unverified webhook endpoint lets anyone enqueue work, so these tests guard
the two checks that must both hold: signature authenticity and timestamp
freshness. A valid-but-old signature is replayable.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time

import pytest

from pipeline.ingest import ZoomWebhookVerifier, parse_recording_completed

SECRET = "test-secret-token"


def _sign(body: bytes, timestamp: str, secret: str = SECRET) -> str:
    message = f"v0:{timestamp}:{body.decode()}"
    digest = hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()
    return f"v0={digest}"


def test_valid_signature_accepted():
    verifier = ZoomWebhookVerifier(SECRET)
    body = b'{"event":"recording.completed"}'
    ts = str(int(time.time()))
    assert verifier.verify(body, _sign(body, ts), ts)


def test_wrong_secret_rejected():
    verifier = ZoomWebhookVerifier(SECRET)
    body = b'{"event":"recording.completed"}'
    ts = str(int(time.time()))
    assert not verifier.verify(body, _sign(body, ts, "wrong-secret"), ts)


def test_tampered_body_rejected():
    verifier = ZoomWebhookVerifier(SECRET)
    ts = str(int(time.time()))
    signature = _sign(b'{"event":"recording.completed"}', ts)
    assert not verifier.verify(b'{"event":"something.else"}', signature, ts)


def test_stale_timestamp_rejected():
    """Replay protection: a correctly signed request from an hour ago is
    still a replay."""
    verifier = ZoomWebhookVerifier(SECRET)
    body = b'{"event":"recording.completed"}'
    ts = str(int(time.time()) - 3600)
    assert not verifier.verify(body, _sign(body, ts), ts)


@pytest.mark.parametrize("signature,timestamp", [("", "123"), ("v0=abc", ""), ("", "")])
def test_missing_headers_rejected(signature, timestamp):
    assert not ZoomWebhookVerifier(SECRET).verify(b"{}", signature, timestamp)


def test_empty_secret_refused_at_construction():
    with pytest.raises(ValueError):
        ZoomWebhookVerifier("")


def test_url_validation_challenge():
    """Zoom proves endpoint ownership with an HMAC of a plaintext token."""
    verifier = ZoomWebhookVerifier(SECRET)
    response = verifier.url_validation_response("abc123")
    expected = hmac.new(SECRET.encode(), b"abc123", hashlib.sha256).hexdigest()
    assert response == {"plainToken": "abc123", "encryptedToken": expected}


# ── Payload parsing ───────────────────────────────────────────────────────────

PAYLOAD = {
    "event": "recording.completed",
    "download_token": "eyJ0oken",
    "payload": {
        "object": {
            "uuid": "abc==",
            "id": 123456789,
            "topic": "Chemistry Class",
            "host_email": "teacher@school.edu",
            "start_time": "2026-08-10T14:02:00Z",
            "duration": 62,
            "recording_files": [
                {
                    "id": "f1",
                    "file_type": "MP4",
                    "file_extension": "MP4",
                    "file_size": 1_200_000_000,
                    "download_url": "https://zoom.us/rec/f1",
                },
                {
                    "id": "f2",
                    "file_type": "M4A",
                    "file_extension": "M4A",
                    "file_size": 95_000_000,
                    "download_url": "https://zoom.us/rec/f2",
                },
            ],
        }
    },
}


def test_metadata_is_extracted():
    recording = parse_recording_completed(PAYLOAD)
    assert recording.uuid == "abc=="
    assert recording.topic == "Chemistry Class"
    assert recording.host_email == "teacher@school.edu"
    assert recording.duration_s == 3720
    assert recording.recorded_at.year == 2026


def test_audio_only_file_preferred_for_transcription():
    """The M4A is roughly a tenth the size of the MP4 and contains everything
    transcription needs. See ADR-003."""
    recording = parse_recording_completed(PAYLOAD)
    chosen = recording.best_file_for_transcription()
    assert chosen.file_type == "M4A"
    assert chosen.file_size < 100_000_000


def test_falls_back_to_video_when_no_audio_file():
    payload = json.loads(json.dumps(PAYLOAD))
    payload["payload"]["object"]["recording_files"] = [
        payload["payload"]["object"]["recording_files"][0]
    ]
    chosen = parse_recording_completed(payload).best_file_for_transcription()
    assert chosen.file_type == "MP4"


def test_wrong_event_type_rejected():
    with pytest.raises(ValueError, match="recording.completed"):
        parse_recording_completed({"event": "meeting.started"})
