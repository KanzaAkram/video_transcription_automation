"""Webhook receiver and admin API.

This service does the least work of anything in the system, on purpose. It
verifies, deduplicates, enqueues, and returns — nothing slow, nothing that
touches a GPU, nothing that waits on Drive beyond a dedupe lookup.

The reason is specific: webhook senders retry aggressively on slow responses.
A handler that transcribes before replying turns one lecture into five
duplicate jobs and a Zoom app marked unhealthy.

Requires the `api` extra: pip install -e ".[api]"
"""

from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Header, Request, Response, status

from pipeline.config import load_secrets, load_settings
from pipeline.ingest import ZoomWebhookVerifier, parse_recording_completed

log = logging.getLogger(__name__)

app = FastAPI(title="Lecture Pipeline", version="0.1.0")

secrets = load_secrets()
settings = load_settings(secrets.settings_file)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "env": secrets.pipeline_env,
        "subjects": [s.id for s in settings.subjects],
        "dry_run": settings.safety.dry_run,
    }


@app.post("/webhooks/zoom")
async def zoom_webhook(
    request: Request,
    x_zm_signature: str = Header(default=""),
    x_zm_request_timestamp: str = Header(default=""),
) -> Response | dict:
    """Receive Zoom events.

    Handles the one-time URL validation challenge and `recording.completed`.
    Everything else is acknowledged and ignored — returning an error for an
    event we didn't subscribe to just makes Zoom retry it.
    """
    body = await request.body()
    payload = await request.json()

    if not secrets.zoom_webhook_secret_token:
        log.error("ZOOM_WEBHOOK_SECRET_TOKEN is not configured; refusing webhook")
        return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

    verifier = ZoomWebhookVerifier(secrets.zoom_webhook_secret_token)

    # Zoom's endpoint-ownership challenge. Answered before signature checking,
    # because this is the request that establishes the trust relationship.
    if payload.get("event") == "endpoint.url_validation":
        plain = payload.get("payload", {}).get("plainToken", "")
        return verifier.url_validation_response(plain)

    if not verifier.verify(body, x_zm_signature, x_zm_request_timestamp):
        log.warning("rejected Zoom webhook with bad signature")
        return Response(status_code=status.HTTP_401_UNAUTHORIZED)

    if payload.get("event") != "recording.completed":
        return {"status": "ignored", "event": payload.get("event")}

    recording = parse_recording_completed(payload)

    # Ignore very short recordings — usually an accidental start/stop.
    if recording.duration_s < settings.ingest.min_duration_seconds:
        log.info("ignoring short recording %s (%.0fs)", recording.uuid, recording.duration_s)
        return {"status": "ignored", "reason": "below minimum duration"}

    # TODO(credentials): enqueue onto the io queue once the job store is wired
    # up. Deduplication is on the Zoom meeting UUID, which is stable across
    # webhook retries and reconciler re-detections alike.
    job_id = str(uuid.uuid4())
    log.info(
        "accepted recording %s (%s, %.0f min) -> job %s",
        recording.uuid, recording.topic, recording.duration_minutes, job_id,
    )
    return {"status": "accepted", "job_id": job_id, "dedupe_key": f"zoom:{recording.uuid}"}


@app.post("/webhooks/drive")
async def drive_webhook(
    x_goog_channel_id: str = Header(default=""),
    x_goog_channel_token: str = Header(default=""),
    x_goog_resource_state: str = Header(default=""),
) -> Response:
    """Receive a Drive change notification.

    Deliberately does not read a body: a Drive notification carries no
    information about *what* changed. It is a doorbell. The actual work is to
    enqueue a `changes.list` walk from the saved page token — see
    docs/ARCHITECTURE.md § 3.2.
    """
    if x_goog_resource_state == "sync":
        # The handshake sent when a channel is first registered.
        return Response(status_code=status.HTTP_200_OK)

    # TODO(credentials): validate the channel token against watch_channels and
    # enqueue a changes sweep.
    log.info(
        "Drive change notification on channel %s (%s)",
        x_goog_channel_id,
        x_goog_resource_state,
    )
    return Response(status_code=status.HTTP_200_OK)
