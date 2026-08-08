"""ASR provider interface and registry.

Everything downstream depends only on `Transcript`, so the engine is a config
change rather than a rewrite. That matters here because the right engine
depends on answers we do not have yet: self-hosted vs hosted turns on volume
and whether a GPU exists, and multilingual vs English-only turns on whether
lectures code-switch. See ADR-004.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pipeline.config import ASRSettings
from pipeline.models import Transcript, TranscriptionHints


class ASRProvider(ABC):
    name: str

    @abstractmethod
    def transcribe(self, audio_path: Path, hints: TranscriptionHints) -> Transcript: ...

    @abstractmethod
    def is_available(self) -> tuple[bool, str]:
        """Whether this provider can run here, and why not if it cannot.

        Checked before a job is dispatched so a missing GPU or a missing
        ffmpeg surfaces as a clear message rather than a stack trace inside a
        worker.
        """


def get_provider(settings: ASRSettings) -> ASRProvider:
    """Construct the configured provider.

    Imports are deferred so that the dry run does not require torch to be
    installed just to build a stub transcript.
    """
    if settings.provider == "faster_whisper":
        from pipeline.asr.faster_whisper_provider import FasterWhisperProvider

        return FasterWhisperProvider(settings)

    if settings.provider == "local_stub":
        from pipeline.asr.stub import StubProvider

        return StubProvider(settings)

    if settings.provider in ("whisperx", "api"):
        raise NotImplementedError(
            f"ASR provider {settings.provider!r} is planned but not implemented. "
            "whisperx adds diarization (Phase 3); `api` covers hosted providers "
            "and is the recommended path if there is no GPU. See ADR-004."
        )

    raise ValueError(f"unknown ASR provider: {settings.provider!r}")
