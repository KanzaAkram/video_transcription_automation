"""faster-whisper: the recommended transcription engine.

A CTranslate2 reimplementation of Whisper — same weights and accuracy as the
reference implementation, substantially faster and lighter on VRAM.

Three settings here matter more than the model choice, and all three are
lecture-specific rather than general defaults:

  * ``condition_on_previous_text=False`` — the default (True) makes Whisper
    loop on long audio, emitting the same sentence for minutes.
  * ``vad_filter=True`` — drops the "can everyone hear me" preamble and stops
    the model hallucinating text over silence.
  * ``initial_prompt`` / ``hotwords`` primed from the syllabus — "stoichiometry"
    and "halogenoalkane" are exactly what generic ASR mangles, and exactly what
    the classifier then depends on.

Requires the `asr` extra and ffmpeg on PATH: pip install -e ".[asr]"
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from pipeline.asr.base import ASRProvider
from pipeline.config import ASRSettings
from pipeline.models import Transcript, TranscriptionHints, TranscriptSegment

log = logging.getLogger(__name__)


class FasterWhisperProvider(ASRProvider):
    name = "faster_whisper"

    def __init__(self, settings: ASRSettings) -> None:
        self.settings = settings
        self._model = None  # loaded lazily; the weights are large

    # ── Availability ─────────────────────────────────────────────────────────

    def is_available(self) -> tuple[bool, str]:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return False, 'faster-whisper not installed; pip install -e ".[asr]"'
        if self.settings.device == "cuda":
            try:
                import ctranslate2

                if ctranslate2.get_cuda_device_count() == 0:
                    return False, "device is 'cuda' but no CUDA device was found"
            except Exception as exc:  # noqa: BLE001
                return False, f"CUDA check failed: {exc}"
        return True, ""

    # ── Model ────────────────────────────────────────────────────────────────

    def _resolve_device(self) -> tuple[str, str]:
        """Pick device and compute type, degrading gracefully to CPU."""
        device, compute = self.settings.device, self.settings.compute_type
        if device == "auto":
            try:
                import ctranslate2

                device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:  # noqa: BLE001
                device = "cpu"
        if device == "cpu" and compute in ("float16", "int8_float16"):
            # float16 is not usable on CPU; int8 is the sensible equivalent.
            log.info("compute_type %s is GPU-only; using int8 on CPU", compute)
            compute = "int8"
        return device, compute

    def _load(self):
        if self._model is not None:
            return self._model
        from faster_whisper import WhisperModel

        device, compute = self._resolve_device()
        log.info(
            "loading %s on %s (%s) — first run downloads the weights",
            self.settings.model, device, compute,
        )
        self._model = WhisperModel(self.settings.model, device=device, compute_type=compute)
        return self._model

    # ── Transcription ────────────────────────────────────────────────────────

    def transcribe(self, audio_path: Path, hints: TranscriptionHints) -> Transcript:
        model = self._load()
        started = time.monotonic()

        kwargs: dict = {
            "beam_size": self.settings.beam_size,
            # See module docstring — this default must be False for lectures.
            "condition_on_previous_text": self.settings.condition_on_previous_text,
            "vad_filter": self.settings.vad_filter,
            "word_timestamps": self.settings.word_timestamps,
        }
        if self.settings.vad_filter:
            kwargs["vad_parameters"] = {"min_silence_duration_ms": self.settings.vad_min_silence_ms}
        # "auto" means let the model detect; anything else pins it.
        if hints.language and hints.language != "auto":
            kwargs["language"] = hints.language
        if hints.initial_prompt:
            kwargs["initial_prompt"] = hints.initial_prompt
        if hints.hotwords:
            kwargs["hotwords"] = " ".join(hints.hotwords[: self.settings.max_hotwords])

        segments_iter, info = model.transcribe(str(audio_path), **kwargs)

        segments = [
            TranscriptSegment(
                start_s=s.start,
                end_s=s.end,
                text=s.text.strip(),
                no_speech_prob=getattr(s, "no_speech_prob", None),
            )
            for s in segments_iter
            if s.text and s.text.strip()
        ]

        runtime = time.monotonic() - started
        duration = getattr(info, "duration", 0.0) or 0.0
        if duration:
            log.info(
                "transcribed %.1f min of audio in %.1f min (%.1fx real time)",
                duration / 60, runtime / 60, duration / runtime if runtime else 0,
            )

        return Transcript(
            segments=segments,
            language=getattr(info, "language", hints.language or "unknown"),
            duration_s=duration,
            model=self.settings.model,
            provider=self.name,
            runtime_s=runtime,
        )
