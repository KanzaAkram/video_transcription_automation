#!/usr/bin/env python
"""Step 1 of connecting the real accounts: check config without calling anything.

Validates that .env parses, settings.yaml is well-formed, taxonomies load, and
every required identifier is present. Makes no network calls, so it is safe to
run before any credentials exist and it localises problems to configuration
rather than to authentication.

    python scripts/verify_config.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline.config import load_secrets, load_settings  # noqa: E402
from pipeline.syllabus import load_taxonomy  # noqa: E402

OK, WARN, FAIL = "  ok  ", " warn ", " FAIL "


def main() -> int:
    problems = 0
    warnings = 0

    def report(status: str, message: str) -> None:
        nonlocal problems, warnings
        if status == FAIL:
            problems += 1
        elif status == WARN:
            warnings += 1
        print(f"[{status}] {message}")

    print("Configuration check\n")

    # ── Secrets ──────────────────────────────────────────────────────────────
    try:
        secrets = load_secrets()
        report(OK, f"loaded .env (env={secrets.pipeline_env})")
    except Exception as exc:  # noqa: BLE001
        report(FAIL, f"could not load .env: {exc}")
        return 1

    # ── Settings ─────────────────────────────────────────────────────────────
    settings_path = Path(secrets.settings_file)
    if not settings_path.exists():
        report(
            WARN,
            f"{settings_path} not found — using built-in defaults. "
            "Copy config/settings.example.yaml to create it.",
        )
        settings_path = REPO_ROOT / "config" / "settings.example.yaml"
    try:
        settings = load_settings(settings_path)
        report(OK, f"loaded {settings_path}")
    except Exception as exc:  # noqa: BLE001
        report(FAIL, f"could not load settings: {exc}")
        return 1

    # ── Subjects and taxonomies ──────────────────────────────────────────────
    if not settings.subjects:
        report(FAIL, "no subjects configured")
    for subject in settings.subjects:
        try:
            taxonomy = load_taxonomy(subject.taxonomy_file)
        except Exception as exc:  # noqa: BLE001
            report(FAIL, f"{subject.id}: taxonomy failed to load: {exc}")
            continue

        nodes = taxonomy.all_nodes()
        placeable = taxonomy.placeable_nodes()
        unverified = [n for n in nodes if not n.verified]
        report(
            OK,
            f"{subject.id}: {len(nodes)} nodes, {len(placeable)} with a Drive folder",
        )
        if unverified:
            report(
                WARN,
                f"{subject.id}: {len(unverified)}/{len(nodes)} nodes unverified — "
                "check learning outcomes against the syllabus PDF before production",
            )
        if not subject.drive_root_folder_id:
            report(WARN, f"{subject.id}: drive_root_folder_id is not set")

    # ── Google ───────────────────────────────────────────────────────────────
    if secrets.google_auth_mode == "local":
        report(WARN, "GOOGLE_AUTH_MODE=local — no real Drive will be touched")
    else:
        try:
            secrets.require_google()
            report(OK, f"Google configuration present (mode={secrets.google_auth_mode})")
        except RuntimeError as exc:
            report(FAIL, str(exc))
        if secrets.google_auth_mode == "service_account" and not secrets.google_shared_drive_id:
            report(
                WARN,
                "service_account mode without GOOGLE_SHARED_DRIVE_ID. Service "
                "accounts have no storage quota and cannot own files in a "
                "personal My Drive — see docs/SETUP.md § 0.",
            )

    # ── Zoom ─────────────────────────────────────────────────────────────────
    if settings.ingest.zoom_webhook_enabled:
        missing = [
            name
            for name, value in {
                "ZOOM_ACCOUNT_ID": secrets.zoom_account_id,
                "ZOOM_CLIENT_ID": secrets.zoom_client_id,
                "ZOOM_CLIENT_SECRET": secrets.zoom_client_secret,
                "ZOOM_WEBHOOK_SECRET_TOKEN": secrets.zoom_webhook_secret_token,
            }.items()
            if not value
        ]
        if missing:
            report(WARN, f"Zoom enabled but missing: {', '.join(missing)}")
        else:
            report(OK, "Zoom credentials present")

    # ── Classification ───────────────────────────────────────────────────────
    if secrets.anthropic_api_key:
        report(OK, f"Anthropic key present (model={settings.classify.model})")
    else:
        report(
            WARN,
            "ANTHROPIC_API_KEY not set — classification falls back to the "
            "keyword baseline, which routes far more lectures to review",
        )

    # ── Local tooling ────────────────────────────────────────────────────────
    if shutil.which("ffmpeg"):
        report(OK, "ffmpeg found on PATH")
    elif settings.asr.provider in ("faster_whisper", "whisperx"):
        report(FAIL, "ffmpeg not on PATH but a self-hosted ASR provider is configured")
    else:
        report(WARN, "ffmpeg not on PATH")

    if sys.version_info >= (3, 14) and settings.asr.provider == "faster_whisper":
        report(
            WARN,
            f"Python {sys.version_info.major}.{sys.version_info.minor} may not have "
            "ctranslate2/torch wheels yet; the ASR worker needs 3.11-3.13",
        )

    # ── Safety ───────────────────────────────────────────────────────────────
    if settings.safety.dry_run:
        report(OK, "dry_run is ON — no Drive changes will be made")
    else:
        report(WARN, "dry_run is OFF — Drive changes will be made for real")

    print(f"\n{problems} problem(s), {warnings} warning(s)")
    if problems:
        print("Fix the problems above before continuing. See docs/SETUP.md.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
