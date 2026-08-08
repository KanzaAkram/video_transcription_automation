"""A stub ASR provider that reads a fixture transcript instead of doing work.

Exists so the dry run and the test suite can exercise every stage after
transcription without a GPU, without ffmpeg, and without waiting minutes for a
real transcription. It is never used in production — `is_available` is honest
about what it is.
"""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.asr.base import ASRProvider
from pipeline.config import ASRSettings
from pipeline.models import Transcript, TranscriptionHints, TranscriptSegment


class StubProvider(ASRProvider):
    name = "local_stub"

    def __init__(self, settings: ASRSettings, fixture: Path | None = None) -> None:
        self.settings = settings
        self.fixture = fixture

    def is_available(self) -> tuple[bool, str]:
        return True, "stub provider — produces fixture output, not real transcription"

    def transcribe(self, audio_path: Path, hints: TranscriptionHints) -> Transcript:
        """Return the fixture transcript.

        Looks for a sibling `.transcript.json` next to the audio, then the
        explicitly configured fixture, then falls back to a minimal placeholder
        so the pipeline still completes.
        """
        candidates = [
            audio_path.with_suffix(".transcript.json"),
            self.fixture,
        ]
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return self._load(Path(candidate))

        return Transcript(
            segments=[
                TranscriptSegment(start_s=0.0, end_s=5.0, text="(stub transcript)")
            ],
            language="en",
            duration_s=5.0,
            model="stub",
            provider=self.name,
            runtime_s=0.0,
        )

    @staticmethod
    def _load(path: Path) -> Transcript:
        data = json.loads(path.read_text(encoding="utf-8"))
        segments = [TranscriptSegment.model_validate(s) for s in data["segments"]]
        return Transcript(
            segments=segments,
            language=data.get("language", "en"),
            duration_s=data.get("duration_s", segments[-1].end_s if segments else 0.0),
            model=data.get("model", "stub"),
            provider="local_stub",
            runtime_s=0.0,
        )
