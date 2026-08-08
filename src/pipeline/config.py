"""Typed configuration.

Two sources, deliberately separated:

  * ``.env``               secrets and per-deployment identifiers
  * ``config/settings.yaml``  behaviour — thresholds, templates, policy

The split means the YAML is safe to commit and review, and secrets never end
up in version control.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Secrets and deployment identifiers (.env) ─────────────────────────────────


class Secrets(BaseSettings):
    """Everything sensitive or deployment-specific. Never committed."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    pipeline_env: Literal["dev", "staging", "prod"] = "dev"
    log_level: str = "INFO"
    settings_file: Path = Path("config/settings.yaml")

    # Google Drive. See docs/SETUP.md § 0 for why the mode matters so much.
    google_auth_mode: Literal["service_account", "impersonation", "oauth_user", "local"] = "local"
    google_service_account_file: Path | None = None
    google_impersonate_subject: str | None = None
    google_oauth_client_id: str | None = None
    google_oauth_client_secret: str | None = None
    google_oauth_refresh_token: str | None = None

    google_drive_root_folder_id: str | None = None
    google_drive_inbox_folder_id: str | None = None
    google_drive_review_folder_id: str | None = None
    google_shared_drive_id: str | None = None

    # Zoom Server-to-Server OAuth
    zoom_account_id: str | None = None
    zoom_client_id: str | None = None
    zoom_client_secret: str | None = None
    zoom_webhook_secret_token: str | None = None

    anthropic_api_key: str | None = None

    database_url: str = "sqlite+aiosqlite:///data/pipeline.db"
    redis_url: str = "redis://localhost:6379/0"

    notify_channel: Literal["none", "email", "slack"] = "none"
    slack_webhook_url: str | None = None
    notify_email_to: str | None = None
    smtp_url: str | None = None

    public_base_url: str = "http://localhost:8080"

    def require_google(self) -> None:
        """Fail fast with an actionable message rather than a 401 later."""
        if self.google_auth_mode == "local":
            return
        missing = [
            name
            for name, value in {
                "GOOGLE_DRIVE_ROOT_FOLDER_ID": self.google_drive_root_folder_id,
                "GOOGLE_DRIVE_INBOX_FOLDER_ID": self.google_drive_inbox_folder_id,
            }.items()
            if not value
        ]
        if self.google_auth_mode == "service_account" and not self.google_service_account_file:
            missing.append("GOOGLE_SERVICE_ACCOUNT_FILE")
        if self.google_auth_mode == "impersonation" and not self.google_impersonate_subject:
            missing.append("GOOGLE_IMPERSONATE_SUBJECT")
        if self.google_auth_mode == "oauth_user" and not self.google_oauth_refresh_token:
            missing.append("GOOGLE_OAUTH_REFRESH_TOKEN")
        if missing:
            raise RuntimeError(
                "Missing Google configuration: "
                + ", ".join(missing)
                + ". See docs/SETUP.md."
            )


# ── Behaviour (settings.yaml) ─────────────────────────────────────────────────


class RetrievalSettings(BaseModel):
    lexical_weight: float = 0.55
    semantic_weight: float = 0.45


class PlacementSettings(BaseModel):
    strategy: Literal["deepest_existing", "exact_node", "chapter_only"] = "deepest_existing"
    allow_create_folders: bool = False
    create_secondary_shortcuts: bool = False


class SubjectSettings(BaseModel):
    id: str
    name: str
    syllabus_code: str | None = None
    taxonomy_file: Path
    drive_root_folder_id: str | None = None
    recordings_subfolder: str | None = None
    languages: list[str] = Field(default_factory=lambda: ["auto"])
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    placement: PlacementSettings = Field(default_factory=PlacementSettings)


class ASRSettings(BaseModel):
    provider: Literal["faster_whisper", "whisperx", "api", "local_stub"] = "faster_whisper"
    model: str = "large-v3"
    device: Literal["cuda", "cpu", "auto"] = "auto"
    compute_type: str = "float16"
    beam_size: int = 5
    # Essential on long lectures: leaving this on causes Whisper to loop,
    # emitting the same sentence for minutes. See docs/OPERATIONS.md.
    condition_on_previous_text: bool = False
    vad_filter: bool = True
    vad_min_silence_ms: int = 500
    prime_from_syllabus: bool = True
    max_hotwords: int = 60
    word_timestamps: bool = False
    diarize: bool = False


class ClassifySettings(BaseModel):
    model: str = "claude-opus-5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    window_seconds: int = 240
    window_overlap_seconds: int = 30
    top_k_candidates: int = 8
    review_threshold: float = 0.85
    coverage_floor: float = 0.25
    secondary_coverage_min: float = 0.15
    use_corrections_as_examples: bool = True
    max_correction_examples: int = 5
    embedding_model: str = "BAAI/bge-m3"


class NamingSettings(BaseModel):
    # Reproduces the existing convention: "Aug, 5 - Atomic Structure.mp4"
    template: str = "{month_abbr}, {day} - {topic}{part_suffix}{ext}"
    max_length: int = 120
    strip_characters: str = r'/\:*?"<>|'
    collision_strategy: Literal["part_suffix", "counter", "timestamp"] = "part_suffix"
    # Must not contain a character in `strip_characters` — "?" is illegal in
    # Windows filenames and in Drive, so a "?" marker produces a rename that
    # fails on every sync client.
    low_confidence_marker: str = " (review)"

    @model_validator(mode="after")
    def _marker_is_filesystem_safe(self) -> NamingSettings:
        illegal = {ch for ch in self.low_confidence_marker if ch in self.strip_characters}
        if illegal:
            raise ValueError(
                f"low_confidence_marker contains characters that are illegal in "
                f"filenames: {''.join(sorted(illegal))!r}. Every rename using the "
                "marker would fail."
            )
        return self


class TranscriptSettings(BaseModel):
    formats: list[Literal["md", "srt", "vtt", "txt", "json"]] = Field(
        default_factory=lambda: ["md", "srt"]
    )
    location: Literal["alongside", "subfolder"] = "alongside"
    subfolder_name: str = "Transcripts"
    write_sidecar: bool = True
    sidecar_folder: str = ".lecture-meta"
    include_timestamps: bool = True
    timestamp_heading_interval_s: int = 300


class IngestSettings(BaseModel):
    zoom_webhook_enabled: bool = True
    drive_watch_enabled: bool = True
    prefer_audio_only: bool = True
    reconcile_interval_minutes: int = 10
    watch_renew_interval_minutes: int = 30
    stuck_job_timeout_minutes: int = 90
    video_extensions: list[str] = Field(default_factory=lambda: [".mp4", ".m4v", ".mov", ".mkv"])
    audio_extensions: list[str] = Field(default_factory=lambda: [".m4a", ".mp3", ".wav"])
    min_duration_seconds: int = 300


class SafetySettings(BaseModel):
    """Guardrails on a component that holds write access to your file tree."""

    dry_run: bool = False
    confine_to_root: bool = True
    allow_delete: bool = False
    audit_log: bool = True
    max_files_per_run: int = 50


class RuntimeSettings(BaseModel):
    media_cache_dir: Path = Path("media_cache")
    keep_audio_after_transcription: bool = False
    max_retries_per_step: int = 3
    retry_backoff_seconds: int = 30
    io_concurrency: int = 8
    gpu_concurrency: int = 1


class NotificationSettings(BaseModel):
    on_needs_review: bool = True
    on_failure: bool = True
    on_success: bool = False
    daily_summary: bool = True


class Settings(BaseModel):
    subjects: list[SubjectSettings] = Field(default_factory=list)
    asr: ASRSettings = Field(default_factory=ASRSettings)
    classify: ClassifySettings = Field(default_factory=ClassifySettings)
    naming: NamingSettings = Field(default_factory=NamingSettings)
    transcripts: TranscriptSettings = Field(default_factory=TranscriptSettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    safety: SafetySettings = Field(default_factory=SafetySettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    notifications: NotificationSettings = Field(default_factory=NotificationSettings)

    def subject(self, subject_id: str) -> SubjectSettings:
        for s in self.subjects:
            if s.id == subject_id:
                return s
        known = ", ".join(s.id for s in self.subjects) or "(none configured)"
        raise KeyError(f"unknown subject {subject_id!r}; configured subjects: {known}")


# ── Loading ───────────────────────────────────────────────────────────────────

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _expand_env(value: Any) -> Any:
    """Expand ``${VAR}`` references in YAML values from the environment.

    Lets settings.yaml reference deployment-specific IDs (folder IDs, mostly)
    without hardcoding them, keeping the file committable.
    """
    if isinstance(value, str):
        return _ENV_VAR_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_settings(path: Path | str | None = None) -> Settings:
    """Load settings.yaml, falling back to defaults if it does not exist.

    Missing config is not an error: the defaults are a working configuration,
    which is what lets the dry run work on a fresh clone.
    """
    if path is None:
        path = Secrets().settings_file
    path = Path(path)
    if not path.exists():
        return Settings()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Settings.model_validate(_expand_env(raw))


def load_secrets() -> Secrets:
    return Secrets()
