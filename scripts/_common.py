"""Shared helpers for the CLI scripts."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline.config import Secrets, Settings, load_settings  # noqa: E402

EXAMPLE_SETTINGS = REPO_ROOT / "config" / "settings.example.yaml"


def load_settings_with_fallback(secrets: Secrets) -> tuple[Settings, Path, bool]:
    """Load settings.yaml, falling back to the committed example.

    A fresh clone has no settings.yaml, and the scripts should still be
    runnable — otherwise the first thing anyone does is hit a confusing empty
    configuration. Returns the settings, the path used, and whether it fell
    back, so callers can say so out loud rather than silently using defaults.
    """
    path = Path(secrets.settings_file)
    if path.exists():
        return load_settings(path), path, False
    return load_settings(EXAMPLE_SETTINGS), EXAMPLE_SETTINGS, True


def select_subjects(settings: Settings, subject_id: str | None):
    """Pick the requested subject, or all of them, with a useful error."""
    if not settings.subjects:
        raise SystemExit(
            "No subjects configured. Copy config/settings.example.yaml to "
            "config/settings.yaml and add at least one subject."
        )
    if subject_id is None:
        return settings.subjects
    matches = [s for s in settings.subjects if s.id == subject_id]
    if not matches:
        known = ", ".join(s.id for s in settings.subjects)
        raise SystemExit(f"No subject {subject_id!r}. Configured subjects: {known}")
    return matches
