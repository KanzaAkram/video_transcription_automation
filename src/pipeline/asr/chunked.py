"""Transcribing a lecture without loading all of it into memory at once.

faster-whisper computes the log-mel spectrogram for the **entire** file before
decoding a single word. That is fine for a three-minute clip and fatal for a
long lecture: a 99.8-minute IGCSE Maths recording asked for a single 1.23 GiB
complex array and died with

    numpy._core._exceptions._ArrayMemoryError:
    Unable to allocate 1.23 GiB for an array with shape (1, 411600, 201)

The allocation scales linearly with duration, so this is not a "buy more RAM"
problem - it is a ceiling on how long a lecture the pipeline can accept, and
the ceiling moves with whatever else the machine happens to be doing. On a
server processing a term's recordings unattended, that is a job that fails on
the longest and most valuable file, intermittently, for reasons that look like
a bug in the ASR library.

So the audio is decoded once and fed to Whisper in fixed slices, with each
slice's timestamps shifted back into whole-lecture time. Peak memory becomes a
function of the slice length rather than of the lecture length, and a
three-hour recording costs the same as a twenty-minute one.

**What this costs.** A word straddling a slice boundary can be clipped, and
Whisper loses the (already disabled) cross-slice context. At the default
twenty minutes that is four boundaries in a long lecture, against roughly four
thousand words - and classification scores four-minute windows of keywords, so
a damaged word at one boundary cannot move a result. Overlapping the slices
would avoid even that, at the price of duplicated text that would then be
counted twice by exactly the keyword-frequency logic the transcript feeds. A
clipped word is the cheaper error.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000

#: Slice length. Twenty minutes keeps the spectrogram allocation near 250 MB,
#: which fits alongside the model on a machine with little free memory, while
#: keeping boundaries rare enough not to matter.
DEFAULT_CHUNK_MINUTES = 20


@dataclass
class TranscriptionInfo:
    """The subset of faster-whisper's info the caller actually uses."""

    duration: float
    language: str


def transcribe_long(
    model,
    audio,
    *,
    language: str,
    task: str = "transcribe",
    initial_prompt: str | None = None,
    beam_size: int = 5,
    vad_filter: bool = True,
    vad_parameters: dict | None = None,
    chunk_minutes: int = DEFAULT_CHUNK_MINUTES,
) -> tuple[list[dict], TranscriptionInfo]:
    """Transcribe `audio` in slices, returning whole-lecture segment rows.

    `audio` is a decoded float32 mono array at 16 kHz, not a path: decoding
    once and slicing is what keeps the peak allocation bounded.
    """
    total_samples = len(audio)
    duration_s = total_samples / SAMPLE_RATE
    chunk_samples = chunk_minutes * 60 * SAMPLE_RATE

    rows: list[dict] = []
    chunks = max(1, -(-total_samples // chunk_samples))  # ceiling division
    if chunks > 1:
        log.info(
            "  long recording (%.0f min): transcribing in %d slices of %d min",
            duration_s / 60,
            chunks,
            chunk_minutes,
        )

    for index in range(chunks):
        start_sample = index * chunk_samples
        clip = audio[start_sample : start_sample + chunk_samples]
        offset_s = start_sample / SAMPLE_RATE

        segments, _ = model.transcribe(
            clip,
            beam_size=beam_size,
            language=language,
            task=task,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters or {"min_silence_duration_ms": 700},
            # Priming with syllabus vocabulary. These are the terms generic ASR
            # mangles and the terms retrieval then depends on, so the error
            # would cost twice. Every slice is primed, not just the first -
            # each one is an independent decode.
            initial_prompt=initial_prompt,
            # Without this one hallucinated line conditions the next, and a
            # lecture can spiral into repeated text.
            condition_on_previous_text=False,
        )
        for segment in segments:
            rows.append(
                {
                    "start": round(segment.start + offset_s, 2),
                    "end": round(segment.end + offset_s, 2),
                    "text": segment.text.strip(),
                    "no_speech_prob": round(
                        getattr(segment, "no_speech_prob", 0.0) or 0.0, 3
                    ),
                }
            )
        if chunks > 1:
            log.info(
                "    slice %d/%d done (%d segments so far)", index + 1, chunks, len(rows)
            )

    return rows, TranscriptionInfo(duration=duration_s, language=language)
