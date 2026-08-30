"""Rename the recordings in one folder from what is said in them.

    # see what it would do
    python scripts/rename_folder.py --folder "D:/chap 12-Respiration" \
        --syllabus config/syllabi/a2-biology-9700.yaml --scope bio.topic_12

    # do it
    python scripts/rename_folder.py --folder "D:/chap 12-Respiration" \
        --syllabus config/syllabi/a2-biology-9700.yaml --scope bio.topic_12 --apply

Two stages: Whisper transcribes each recording, then ChatGPT reads the whole
transcript against the whole syllabus and returns the chapter and the one to
three subtopics the lecture actually taught. The filename is then built from
the syllabus' own titles for those ids — the model returns ids, never names,
so nothing invented can reach a filename. Set `OPENAI_API_KEY` in `.env`.

Deliberately narrower than the full pipeline: this **only renames**. Nothing
moves, no folder is created, no transcript is published. It exists for the case
where the recordings are already sitting in the right chapter folder and the
only thing wrong with them is that they are called "class 4 (9 aug).mp4".

Dry run is the default. `--apply` writes an undo file next to the plan, so a
bad batch is one command to reverse.

--scope
-------
Restricts candidates to one subtree. When you already know the folder is
chapter 12, letting the classifier consider photosynthesis and gene therapy
adds nothing but ways to be wrong: a lecture on the electron transport chain
in respiration shares most of its vocabulary with the one in photosynthesis.
Scoping is the single largest accuracy win available here and costs nothing.

--level
-------
Which layer of the taxonomy the name comes off:

    chapter   depth 0 - one Drive folder
    subtopic  depth 1 - one syllabus subtopic
    lesson    the leaves below depth 1, where a syllabus has that layer

The default is **adaptive**: the deepest of these that the loaded syllabus
actually has. It has to be, because the right answer is a property of the
subject and picking wrong is silent. A 9701 or 0580 subtopic is one lecture, so
its short label is the filename. A 9700 subtopic is not: topic 12 is two
subtopics ("Energy", "Respiration") over six recordings, so naming from
subtopics gives six files two names between them — and one of those names is
the name of the folder they are already sitting in. That is exactly what the
hand-curated lesson layer exists to avoid, and a fixed default of `subtopic`
would throw it away without saying so.

So: `lesson` where a lesson layer exists, `subtopic` where it does not. Which
one was chosen, and why, is logged on every run. An explicit `--level`
overrides it, and `lesson` still falls back to `subtopic` where a syllabus has
no lesson layer.

--classifier
------------
Who decides. `openai` is the default: the transcript and the syllabus subtree
go to ChatGPT in one call, and what comes back is a chapter id and one to three
subtopic ids drawn from an enumeration of that subtree — a reply naming
something the syllabus does not contain is not a legal reply, and every id is
re-checked against the taxonomy afterwards anyway.

`keyword` is the earlier behaviour and needs no API key: it counts syllabus
vocabulary per four-minute window and files the recording where the counting
points. That is why it is no longer the default. Counting cannot distinguish a
topic the teacher taught for fifty minutes from one they named twice, so a
lecture that explains glycolysis without ever saying "glycolysis" is filed
under whichever neighbouring chapter happened to share its words, and a lecture
that opens on a recap is named after the recap. Both failures produce a
confident-looking filename. It stays for offline runs, for measuring the model
against, and for a folder where an API call is not wanted.

`anthropic` is the same idea through Claude, over a retrieved shortlist rather
than the whole syllabus.

--language
----------
`en` by default, and that default is a measurement rather than a shrug. These
lectures code-switch Urdu and English. Handing "detect it yourself" to
faster-whisper - which is what `language=None` means - is the one option that
does not work at all: it calls the audio Hindi at 0.94, emits Devanagari that
matches no English syllabus keyword, and runs seventeen times slower because an
uncertain language sends every window through the temperature-fallback retries.
So `None` is never passed to the decoder; `transcribe()` refuses it outright.

`--language auto` does not mean that. It means: detect the language once for
the folder, and then *choose* between the two decodes that produce usable
English - transcribe-as-English and translate-to-English - by scoring both on
how much of this subject's syllabus vocabulary they recover
(`pipeline.asr.language.choose_decode`). The chosen decode, the detected
language and the two scores are logged, cached in `.transcripts/language.json`
so re-runs stay instant, and recorded in `rename_plan.json`.

`en` is the default because on 48 minutes of this teaching there is nothing to
find. Transcribing as English recovers 5.7% more syllabus vocabulary than
translating in Chemistry and 7.3% less in Maths - opposite answers, both far
smaller than the probe's own scatter, and both reversed when you count the
technical terms a human would check, where transcription leads in both
subjects. Translation writes tidier sentences and loses content inside them:
"one twelfth the mass of carbon 12" came back as "112th the mass of
carbon-12". `auto` is what you pass for a teacher or a subject that has not
been measured. See docs/RENAMING.md.

--transcribe-only
-----------------
Stops after transcription and prints what each file produced. No classifier is
built and no API key is read, so the ASR half can be verified on its own before
any naming decision depends on it.

Transcripts are cached per (file, model), so re-running to try a different
scope or classifier is instant and does not re-transcribe.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.asr.chunked import transcribe_long  # noqa: E402
from pipeline.asr.language import DecodeChoice, choose_decode  # noqa: E402
from pipeline.classify.retriever import HybridRetriever  # noqa: E402
from pipeline.classify.service import build_classifier  # noqa: E402
from pipeline.config import (  # noqa: E402
    ClassifySettings,
    NamingSettings,
    RetrievalSettings,
    load_secrets,
)
from pipeline.models import (  # noqa: E402
    Classification,
    Recording,
    Source,
    Transcript,
    TranscriptSegment,
)
from pipeline.naming import build_name, sanitise  # noqa: E402
from pipeline.publish.renderers import render_markdown  # noqa: E402
from pipeline.syllabus import Taxonomy, load_taxonomy, normalise_folder_name  # noqa: E402

log = logging.getLogger("rename")

VIDEO_SUFFIXES = {".mp4", ".m4a", ".mov", ".mkv", ".webm", ".mp3", ".wav"}

_MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1
    )
}
#: "class 3 (6 aug).mp4", "Lecture 2 - 14 Sept", "2025-08-06 class"
_DATE_IN_NAME = re.compile(
    r"(?:(?P<day>\d{1,2})\s*(?P<mon>jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)"
    r"|(?P<mon2>jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\w*\s*,?\s*(?P<day2>\d{1,2}))",
    re.IGNORECASE,
)
_ISO_IN_NAME = re.compile(r"(?P<y>20\d{2})[-_](?P<m>\d{2})[-_](?P<d>\d{2})")


def recorded_at_for(path: Path) -> tuple[datetime, str]:
    """Date for the filename, and where it came from.

    The name is trusted over the mtime: a Drive download rewrites mtime to the
    download time, which would date every lecture in the batch to today.
    """
    stem = path.stem
    iso = _ISO_IN_NAME.search(stem)
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    if iso:
        return (
            datetime(int(iso["y"]), int(iso["m"]), int(iso["d"])),
            "filename",
        )
    match = _DATE_IN_NAME.search(stem)
    if match:
        day = match["day"] or match["day2"]
        mon = (match["mon"] or match["mon2"]).lower()[:3]
        # The year is the one thing a "6 aug" filename cannot tell you, so it
        # comes from the mtime — wrong only if a folder spans a new year.
        return datetime(mtime.year, _MONTHS[mon], int(day)), "filename"
    return mtime, "mtime"


# ── Transcription ─────────────────────────────────────────────────────────────


def content_key(path: Path) -> str:
    """Short stable id for a media file, independent of its name.

    Keyed on content rather than filename precisely because this script
    *renames* files: a name-keyed cache is invalidated by the tool's own
    output, so the second run re-transcribes everything it just did.

    Hashing the head and tail plus the size is enough — two lectures never
    collide, and reading a whole 60 MB video to compute a key would cost more
    than it saves.
    """
    size = path.stat().st_size
    digest = hashlib.blake2b(str(size).encode(), digest_size=8)
    with path.open("rb") as handle:
        digest.update(handle.read(1 << 20))
        if size > (2 << 20):
            handle.seek(-(1 << 20), 2)
            digest.update(handle.read(1 << 20))
    return digest.hexdigest()


class UnreadableMedia(Exception):
    """A media file that cannot be decoded, as opposed to a bug in this script.

    Raised so the batch can skip one bad recording and keep going. A folder is
    a term's worth of lectures; letting a single unfinalised Zoom file abort
    the run means the other twenty are not processed either, and the operator
    sees a stack trace rather than a list of what succeeded.
    """


def _decode_audio(path: Path, attempts: int = 3):
    """Decode to 16 kHz mono, retrying a failure that is really memory pressure.

    PyAV raised `InvalidDataError: Invalid data found when processing input` on
    a file whose CRC matched its zip entry byte for byte and whose header was a
    valid `ftypmp42`. The file was fine; the machine had 1.5 GB free at the
    time. A transient allocation failure that surfaces as a corruption error is
    exactly the kind of thing that gets a good recording written off as broken,
    so it is retried rather than believed the first time.
    """
    from faster_whisper import decode_audio

    for attempt in range(1, attempts + 1):
        try:
            return decode_audio(str(path), sampling_rate=16000)
        except Exception as error:  # noqa: BLE001 - PyAV raises several types
            if attempt == attempts:
                raise UnreadableMedia(
                    f"{path.name}: {type(error).__name__}: {error}"
                ) from error
            log.warning(
                "  decode failed (%s: %s); retrying %d/%d", type(error).__name__, error,
                attempt + 1, attempts,
            )
            gc.collect()
    raise AssertionError("unreachable")


def load_model(model_holder: dict, model_name: str):
    """The one WhisperModel this run uses, loaded on first need.

    Shared by the language probe and by transcription, because a second model
    would be a second 250 MB resident on a machine that has 500 MB free.
    """
    if "model" not in model_holder:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise SystemExit(
                'faster-whisper is not installed. pip install -e ".[asr]"'
            ) from None
        log.info("  loading %s ...", model_name)
        model_holder["model"] = WhisperModel(
            model_name, device="cpu", compute_type="int8", cpu_threads=10
        )
    return model_holder["model"]


def transcribe(
    path: Path,
    cache_dir: Path,
    model_name: str,
    taxonomy: Taxonomy,
    *,
    model_holder: dict,
    language: str = "en",
    task: str = "transcribe",
) -> Transcript:
    """Transcribe with faster-whisper, caching one JSON per (content, model, language).

    `language` is part of the cache key because it changes the output entirely,
    not marginally: the same lecture decoded as Hindi and as English are two
    different transcripts, and silently reusing the wrong one would be invisible.

    `language` is required to be a real ISO code. faster-whisper spells
    auto-detect as `language=None`, and that is the one setting measured to be
    both unusable and seventeen times slower on this audio - see
    `pipeline.asr.language`. `--language auto` resolves to a concrete code
    *before* it gets here; None never reaches the decoder, so the guard below
    is a wall rather than a preference.
    """
    if not language:
        raise ValueError(
            "transcribe() needs an explicit language. None means auto-detect to "
            "faster-whisper, which on this audio produced Devanagari at 0.3x "
            "realtime. Resolve '--language auto' with resolve_decode() first."
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{content_key(path)}.{model_name}.{language}-{task}.json"
    legacy = cache_dir / f"{path.stem}.{model_name}.json"
    if not cached.exists() and legacy.exists():
        legacy.rename(cached)

    if cached.exists():
        payload = json.loads(cached.read_text(encoding="utf-8"))
        log.info("  transcript: cached (%s)", cached.name)
    else:
        model = load_model(model_holder, model_name)
        log.info("  transcribing %s (not cached) ...", path.name)
        started = datetime.now()
        # Decoded once here rather than inside faster-whisper, because
        # transcribe_long slices the array: passing a path would make the
        # library compute one spectrogram for the whole lecture, which is the
        # 1.23 GiB allocation that killed the 99.8-minute Maths recording.
        audio = _decode_audio(path)
        rows, info = transcribe_long(
            model,
            audio,
            # Pinning the language is not a nicety here. These lectures are
            # taught in mixed Urdu and English, which Whisper auto-detects as
            # Hindi at 0.85 - and then decodes into Devanagari, so every
            # English term retrieval depends on disappears. It is also
            # catastrophically slow: measured on a 3-minute clip of the 11 Aug
            # lecture, auto-detect ran at 0.3x realtime against 5.2x with
            # language="en", because an uncertain language sends every window
            # through Whisper's temperature-fallback retries. Same audio, same
            # model: 17x faster, and usable instead of unusable.
            language=language,
            task=task,
            initial_prompt=taxonomy.initial_prompt(40),
        )
        payload = {
            "source": path.name,
            "model": model_name,
            "duration_s": round(info.duration, 1),
            "language": info.language,
            "task": task,
            "segments": rows,
            "runtime_s": round((datetime.now() - started).total_seconds(), 1),
        }
        cached.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    return Transcript(
        segments=[
            TranscriptSegment(
                start_s=r["start"],
                end_s=r["end"],
                text=r["text"],
                no_speech_prob=r.get("no_speech_prob"),
            )
            for r in payload["segments"]
        ],
        language=payload.get("language", "en"),
        duration_s=payload["duration_s"],
        model=payload["model"],
        provider="faster-whisper",
        runtime_s=payload.get("runtime_s"),
    )


def report_transcript(transcript: Transcript) -> None:
    """What --transcribe-only prints: enough to judge the ASR run alone.

    The realtime factor is the number that decides whether a batch is worth
    starting: at 0.4x an hour of lecture costs twenty-four minutes, at 3x it
    costs three hours and the model is too big for this machine.
    """
    duration = transcript.duration_s or 0.0
    runtime = transcript.runtime_s
    factor = f"{runtime / duration:.2f}x" if runtime and duration else "n/a"
    log.info("  duration  %.1f min (%.0f s)", duration / 60, duration)
    log.info("  segments  %d", len(transcript.segments))
    log.info("  runtime   %s", f"{runtime:.0f} s" if runtime else "n/a")
    log.info("  realtime  %s   (runtime / audio duration; lower is faster)", factor)
    log.info("  language  %s", transcript.language)
    text = transcript.text
    log.info("  text      %s%s", text[:200], "..." if len(text) > 200 else "")


# ── Language ──────────────────────────────────────────────────────────────────

#: Where a folder's language decision is remembered, inside the transcript
#: cache so it travels with the transcripts it explains and is thrown away by
#: the same `rm -r .transcripts` that invalidates them.
DECISION_FILE = "language.json"


def _read_decision(cache_dir: Path, subject_id: str) -> DecodeChoice | None:
    """The probe's answer from a previous run of this folder, if there is one.

    Persisted because re-running is the normal way this script is used - try a
    different scope, a different level, a different classifier - and every one
    of those re-runs is otherwise instant off the transcript cache. A probe
    that reloads the model and re-decodes six minutes of audio on every re-run
    would take the cheapest operation in the tool and make it the most
    expensive, which is exactly how a good default stops being used.

    Keyed by subject, matching `choose_decode`'s own memoisation: the score is
    a count of *that subject's* vocabulary, so Chemistry's answer is not
    evidence about Maths.
    """
    path = cache_dir / DECISION_FILE
    if not path.exists():
        return None
    try:
        stored = json.loads(path.read_text(encoding="utf-8")).get(subject_id)
    except (OSError, ValueError):
        return None
    if not stored:
        return None
    return DecodeChoice(
        language=stored["language"],
        task=stored["task"],
        detected=stored["detected"],
        detected_probability=stored["detected_probability"],
        reason=stored["reason"],
        scores=stored.get("scores", {}),
    )


def _write_decision(cache_dir: Path, subject_id: str, choice: DecodeChoice) -> None:
    path = cache_dir / DECISION_FILE
    stored: dict = {}
    if path.exists():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            stored = {}
    stored[subject_id] = {
        "language": choice.language,
        "task": choice.task,
        "detected": choice.detected,
        "detected_probability": round(choice.detected_probability, 3),
        "reason": choice.reason,
        "scores": {k: round(v, 2) for k, v in choice.scores.items()},
        "decided_at": datetime.now().isoformat(timespec="seconds"),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stored, indent=1, ensure_ascii=False), encoding="utf-8")


def probe_order(files: list[Path]) -> list[Path]:
    """The recordings the language probe will try, cheapest first.

    The probe samples three minutes from the middle of one file, and every file
    in the folder is the same teacher teaching the same subject, so any of them
    answers the question equally well. Smallest first because reaching the
    middle means decoding the whole file, and that decode is the probe's real
    cost: on a machine with 500 MB free, probing the 100-minute lecture instead
    of the 45-minute one is a 380 MB allocation that buys nothing.

    A list rather than a single pick, because "smallest" and "undecodable" are
    not independent - an interrupted upload produces a *short* file, so the
    cheapest candidate is disproportionately likely to be the broken one. In
    the Maths folder it is exactly that: the smallest file has no moov atom,
    and a probe that insisted on it would abort a run the rest of which is
    fine, before a single lecture had been looked at.
    """
    return sorted(files, key=lambda p: p.stat().st_size)


def resolve_decode(
    language_flag: str,
    files: list[Path],
    taxonomy: Taxonomy,
    cache_dir: Path,
    *,
    model_holder: dict,
    model_name: str,
    reprobe: bool = False,
) -> DecodeChoice:
    """Turn `--language` into a concrete (language, task) for the whole folder.

    This is the function that exists so `None` never reaches faster-whisper.
    "auto" here does **not** mean "let the decoder detect it" - that is the one
    option measured to be both unusable and seventeen times slower on this
    audio. It means "detect the language here, then choose between the two
    decodes that produce usable English", which is `choose_decode`.

    Decided once per folder rather than once per file. The lectures in a folder
    are one teacher and one subject, so a per-file probe would pay for the same
    answer twenty times, and - worse - could return different answers for
    consecutive lectures of one course, which is a difference no operator
    reading the plan could explain.
    """
    flag = (language_flag or "").strip().lower()
    if not flag:
        raise SystemExit("--language cannot be empty. Pass an ISO code, or 'auto'.")

    if flag != "auto":
        # Logged even though nothing was decided, because "which language was
        # this decoded as" is the question a transcript full of Devanagari or
        # of clumsy translated prose makes someone ask, and the answer should
        # be in the run they already have rather than in a re-run.
        log.info(
            "language: %s (forced by --language, no probe). Pass --language auto to "
            "detect it and choose between transcribing as English and translating.",
            flag,
        )
        return DecodeChoice(
            language=flag,
            task="transcribe",
            detected="(not probed)",
            detected_probability=0.0,
            reason=f"--language {flag}, no probe",
        )

    if not reprobe and (remembered := _read_decision(cache_dir, taxonomy.subject_id)):
        log.info(
            "language: auto -> %s / %s   (%s; remembered from %s, pass --reprobe to redo)",
            remembered.language, remembered.task, remembered.reason,
            cache_dir / DECISION_FILE,
        )
        _log_scores(remembered)
        return remembered

    model = load_model(model_holder, model_name)
    choice = None
    for sample in probe_order(files):
        log.info("language: auto - probing %s ...", sample.name)
        try:
            audio = _decode_audio(sample)
        except UnreadableMedia as error:
            log.warning("  cannot decode it (%s); probing the next recording", error)
            continue
        try:
            choice = choose_decode(model, audio, taxonomy)
        finally:
            del audio
            gc.collect()
        break
    if choice is None:
        raise SystemExit(
            "--language auto: none of the recordings in this folder could be decoded, "
            "so the language could not be probed. Pass an explicit --language, or "
            "re-download the folder."
        )

    log.info("language: auto -> %s", choice.summary)
    _log_scores(choice)
    _write_decision(cache_dir, taxonomy.subject_id, choice)
    log.info("  decision cached in %s", cache_dir / DECISION_FILE)
    return choice


def _log_scores(choice: DecodeChoice) -> None:
    """Print the evidence, not just the verdict.

    A default that varies with the audio is only safe if every run says which
    way it went and on what grounds - the same reason `--level` prints its
    reason. Two numbers a percent apart look very different from two numbers
    that are a factor of two apart, and only one of those is worth acting on.
    """
    for label, score in sorted(choice.scores.items(), key=lambda kv: -kv[1]):
        log.info("  probe %-16s %5.1f syllabus terms per 1000 words", label, score)


# ── Scoping ───────────────────────────────────────────────────────────────────


def infer_scope(taxonomy: Taxonomy, folder: Path) -> str | None:
    """Match a folder name to a syllabus node, so --scope is usually unnecessary.

    Exact match on `drive_folder_name` first, then the normalised form, which
    strips the leading ordinal and the "(Chapter N)" suffix. Returns None
    rather than guessing when nothing matches — silently scoping to the wrong
    chapter would be far worse than not scoping at all.
    """
    name = folder.name
    for node in taxonomy.all_nodes():
        if node.drive_folder_name == name:
            return node.id
    target = normalise_folder_name(name)
    matches = [
        node.id
        for node in taxonomy.all_nodes()
        if node.drive_folder_name and normalise_folder_name(node.drive_folder_name) == target
    ]
    return matches[0] if len(matches) == 1 else None


def scoped_taxonomy(taxonomy: Taxonomy, scope: str) -> Taxonomy:
    """A copy of `taxonomy` holding only `scope` and its descendants."""
    root = taxonomy.get(scope)
    if root is None:
        available = ", ".join(n.id for n in taxonomy.all_nodes() if n.depth == 0)
        raise SystemExit(f"unknown --scope {scope!r}. Top-level nodes: {available}")
    subset = taxonomy.model_copy(update={"nodes": [root], "pending_nodes": []}, deep=True)
    index: dict = {}

    def walk(nodes, parent):
        for node in nodes:
            node.parent_id = parent.id if parent else None
            node.depth = (parent.depth + 1) if parent else 0
            node.path = [*parent.path, parent.title] if parent else []
            index[node.id] = node
            walk(node.children, node)

    walk(subset.nodes, None)
    subset._index = index
    return subset


def nodes_at_level(taxonomy: Taxonomy, level: str) -> set[str]:
    """The node ids the classifier is allowed to name a file after.

    Filtering candidates *before* classifying rather than after is what makes
    the reported confidence mean anything: it is confidence in the choice
    actually being made, not in a choice that was then overruled.

    'lesson' falls back to depth 1 when a taxonomy has no lesson layer. Most
    syllabi do not have one — it is hand-curated teaching structure, not
    Cambridge text — and returning an empty set there would name every file
    "Unclassified" for a reason no console line would explain.

    A depth-0 chapter with no subtopics stays selectable at 'subtopic' level
    for the same reason: something has to be namable.
    """
    nodes = taxonomy.all_nodes()
    if level == "chapter":
        return {n.id for n in nodes if n.depth == 0}
    subtopics = {n.id for n in nodes if n.depth == 1}
    childless_chapters = {n.id for n in nodes if n.depth == 0 and not n.children}
    if level == "subtopic":
        return subtopics | childless_chapters
    lessons = {n.id for n in nodes if n.depth > 1 and not n.children}
    return lessons or (subtopics | childless_chapters)


def default_level(taxonomy: Taxonomy) -> tuple[str, str]:
    """The level to name from when `--level` was not given, and why.

    Adaptive rather than a fixed string, because which layer is the teaching
    unit is a property of the syllabus, and getting it wrong produces plausible
    filenames that are all the same. 9701 and 0580 subtopics *are* the teaching
    unit — one lecture, one subtopic — so `subtopic` distinguishes their
    recordings. A2 Biology 9700 carries a hand-curated lesson layer precisely
    because its subtopics do not: every biology subtopic has between two and
    ten lessons under it, and chapter 12 is two subtopics over six recordings.
    Defaulting to `subtopic` there would name six files "... - Respiration.mp4"
    inside a folder already called "chap 12-Respiration", differing only by
    date.

    So the rule is: go as deep as this taxonomy goes. The reason is returned
    alongside the level rather than kept private — a default that varies by
    input is only safe if every run says which way it went and on what grounds.
    """
    nodes = taxonomy.all_nodes()
    subtopics = sum(1 for n in nodes if n.depth == 1)
    lessons = sum(1 for n in nodes if n.depth > 1 and not n.children)
    if lessons:
        return "lesson", (
            f"this syllabus has a lesson layer - {lessons} lesson nodes under "
            f"{subtopics} subtopics - and the lesson is what one recording covers"
        )
    return "subtopic", (
        f"this syllabus has no lesson layer, so its {subtopics} subtopics are "
        f"the finest layer available and are the teaching unit"
    )


def chapter_report(taxonomy: Taxonomy, classification: Classification) -> dict:
    """The chapter decision, flattened for the console and for the JSON plan.

    Both consumers want the same three things — which chapter, how much of the
    lecture it holds, and what came second — and a near-tie between the top two
    is the signature of a misfiling, which is invisible in the filename.

    A recording that was not placed has no winner, so it gets no runner-up
    either. Filtering the ranking by a chapter id of None removes nothing, so
    the heaviest chapter used to be reported as the runner-up to a chapter that
    was never chosen: "chapter (none)   runner-up: Chemical Bonding 62%" reads
    as second place to something, in exactly the case where a human is being
    asked to supply the first place. The ranking is still the useful thing to
    show there, so it is shown under its honest name instead.
    """
    weights = classification.chapter_weights
    chapter_id = classification.chapter_node_id
    ranked = sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))

    def describe(node_id: str, weight: float) -> str:
        node = taxonomy.get(node_id)
        return f"{node.title if node else node_id} {weight:.0%}"

    if chapter_id is None:
        return {
            "id": None,
            "title": None,
            "weight": None,
            "runner_up": None,
            "top_candidates": [describe(i, w) for i, w in ranked[:3]] or None,
        }

    node = taxonomy.get(chapter_id)
    runner_up = next(((i, w) for i, w in ranked if i != chapter_id), None)
    return {
        "id": chapter_id,
        "title": node.title if node else chapter_id,
        "weight": round(weights[chapter_id], 3) if chapter_id in weights else None,
        "runner_up": describe(*runner_up) if runner_up else None,
        "top_candidates": None,
    }


def discover_media(folder: Path) -> list[Path]:
    """Media in the folder, including inside chapter subfolders.

    Re-running over an already-sorted folder has to keep working: once
    --sort-into-chapters has moved recordings into "1. Atomic Structure
    (Chapter 1)", a root-only scan finds nothing and the run silently does
    nothing at all. That is the shape of an unattended job that appears to
    succeed while processing zero files, so one level of descent is part of
    discovery rather than an option.

    One level, not arbitrary depth: chapter folders are the structure this tool
    creates, and recursing further would sweep in unrelated media a user parked
    nearby.
    """
    found = [p for p in folder.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES]
    for child in folder.iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue
        found += [p for p in child.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES]
    return found


def write_transcript_files(
    folder: Path,
    rows: list[dict],
    transcripts: dict[str, Transcript],
    teacher: str | None,
    subject_id: str,
    *,
    location: str,
    subfolder_name: str,
    placed: dict[str, Path],
) -> int:
    """Write one readable .md per recording.

    `location` mirrors `transcripts.location` in settings: "alongside" puts the
    transcript next to its video - which means inside the chapter folder once
    sorting has moved it, so the two travel together - and "subfolder" collects
    them all under one directory instead.

    Named after the *new* filename so transcript and video sort adjacently. The
    JSON in .transcripts/ is a machine cache keyed by content hash; this is the
    copy a human reads.
    """
    written = 0
    for row in rows:
        transcript = transcripts.get(row["from"])
        if transcript is None:
            continue
        recording = Recording(
            id=Path(row["from"]).stem,
            source=Source.MANUAL,
            source_id=row["from"],
            subject_id=subject_id,
            original_filename=row["from"],
            recorded_at=datetime.fromisoformat(row["recorded_at"]),
            duration_s=transcript.duration_s,
            teacher=teacher,
        )
        title = Path(row["to"]).stem
        # `placed` records where sorting actually put the file; a recording held
        # back for review is still in the folder root, and its transcript
        # belongs beside it rather than in a chapter it was not filed into.
        home = placed.get(row["to"], folder)
        # "subfolder" is relative to where the recording lives, not to the folder
        # root: a transcript belongs with its chapter, and collecting every
        # subject's transcripts into one root directory separates them from the
        # recordings they describe.
        destination = home if location == "alongside" else home / subfolder_name
        destination.mkdir(parents=True, exist_ok=True)
        (destination / f"{title}.md").write_text(
            render_markdown(recording, transcript, title=title), encoding="utf-8"
        )
        written += 1
    log.info("wrote %d transcript(s) (%s)", written, location)
    return written


def sort_into_chapters(
    folder: Path,
    rows: list[dict],
    *,
    include_reviewed: bool,
    strip: str,
    placed: dict[str, Path],
    current: dict[str, Path],
) -> list[dict]:
    """Move each renamed recording into its chapter's folder.

    Everything is filed, including recordings that came in below the
    confidence threshold, unless `include_reviewed` says otherwise.

    That is the opposite of the original rule, and the reason is that the two
    decisions are not equally uncertain. Confidence is reported for the
    *subtopic* — the name — while the chapter is a much easier call and the one
    the model has been most consistent about: the 10 August maths lecture moved
    between three different subtopic names across runs and never once left
    Number. Holding a file out of its chapter because its name is uncertain
    withholds the reliable half of the answer on account of the unreliable
    half, and leaves the recording in a folder root where nothing is going to
    find it either.

    A recording with no chapter at all is still left alone; there is nowhere to
    put it. `--hold-review` restores the old behaviour.
    """
    moves: list[dict] = []
    for row in rows:
        if row["low_confidence"] and not include_reviewed:
            log.info("  kept in place (review): %s", row["to"])
            continue
        chapter = row.get("chapter_title")
        if not chapter:
            log.info("  kept in place (no chapter): %s", row["to"])
            continue
        # The Drive folder name when the overlay records one, else the chapter
        # title - never a name invented from the subtopic.
        destination = folder / sanitise(row.get("chapter_folder") or chapter, strip)
        destination.mkdir(exist_ok=True)
        source = current.get(row["to"], folder / row["to"])
        if source.parent == destination:
            log.info("  already in %s/: %s", destination.name, row["to"])
            continue
        if not source.exists():
            log.warning("  missing, cannot sort: %s", row["to"])
            continue
        target = destination / row["to"]
        if target.exists():
            log.warning("  already present, not overwritten: %s", target)
            continue
        source.rename(target)
        placed[row["to"]] = destination
        moves.append({"file": row["to"], "into": destination.name})
        log.info("  %s  ->  %s/", row["to"], destination.name)
    return moves


def _report_unreadable(unreadable: list[str], total: int) -> int:
    """Summarise skipped files, and fail the run only if nothing worked.

    Exit 0 with a warning when some files succeeded: the operator wants the
    good results filed and a named list of what to re-download, not an
    all-or-nothing failure.
    """
    if not unreadable:
        return 0
    log.warning("")
    log.warning(
        "%d of %d file(s) could not be decoded and were skipped:",
        len(unreadable), total,
    )
    for name in unreadable:
        log.warning("  %s", name)
    log.warning(
        "A file with no 'moov' atom is an unfinalised recording - the container index is "
        "written last, so an interrupted upload leaves the media with no index. Re-download "
        "that file on its own rather than as part of a folder zip; if it is still unreadable, "
        "the copy in Drive is itself incomplete."
    )
    return 0 if len(unreadable) < total else 1


def build_parser() -> argparse.ArgumentParser:
    """The command line, built separately so its defaults can be tested.

    `--language` in particular: 'en' there is a measurement, not a
    preference, and flipping it is a one-character change with folder-wide
    consequences that no test would otherwise notice.
    """
    parser = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    parser.add_argument("--folder", required=True, type=Path)
    parser.add_argument("--syllabus", required=True, type=Path)
    parser.add_argument("--scope", help="Restrict candidates to this node id and its children")
    parser.add_argument("--model", default="small", help="faster-whisper model (default: small)")
    parser.add_argument(
        "--language",
        default="en",
        help="How to decode the audio. An ISO code ('en', 'ur', 'hi') forces "
        "that language with no probe. 'auto' detects the language once per "
        "folder and then CHOOSES between transcribing as English and "
        "translating to English, scoring both on how much of this subject's "
        "syllabus vocabulary they recover - it never hands 'detect it "
        "yourself' to the decoder, which on this audio produced Devanagari at "
        "0.3x realtime against 5.2x for 'en'. Default 'en': measured on 48 "
        "minutes of these lectures the two decodes recover the same syllabus "
        "vocabulary, so the free option wins. Use 'auto' for a teacher or a "
        "subject this has not been measured on (default: en)",
    )
    parser.add_argument(
        "--reprobe",
        action="store_true",
        help="With --language auto, re-run the language probe even though this "
        "folder already has a remembered decision. The decision is cached in "
        ".transcripts/language.json so ordinary re-runs stay instant; pass this "
        "after changing teacher, syllabus or model.",
    )
    parser.add_argument(
        "--cache", type=Path, help="Transcript cache dir (default: <folder>/.transcripts)"
    )
    parser.add_argument("--teacher", default=None)
    parser.add_argument(
        "--level",
        choices=["chapter", "subtopic", "lesson"],
        default=None,
        help="Which layer of the taxonomy names the file. 'chapter' is a Drive "
        "folder (depth 0), 'subtopic' is one syllabus subtopic (depth 1), "
        "'lesson' is the hand-curated teaching layer below that, falling back "
        "to subtopic where a syllabus has no lesson layer. The default is "
        "adaptive - the deepest layer the loaded syllabus actually has, so "
        "'lesson' for A2 Biology 9700 and 'subtopic' for AS Chemistry 9701 and "
        "IGCSE Maths 0580 - and the level chosen is logged with its reason. "
        "Pass this flag to override it.",
    )
    parser.add_argument(
        "--classifier",
        choices=["openai", "keyword", "anthropic"],
        default="openai",
        help="Who decides which chapter and subtopics a recording is named "
        "for. 'openai' sends the whole transcript and the whole (scoped) "
        "syllabus to ChatGPT, which reads the lecture and returns syllabus "
        "node ids; needs OPENAI_API_KEY. 'keyword' is retrieval only and "
        "needs no API key - it counts syllabus vocabulary per window, which "
        "cannot tell a topic that was taught from one that was named, and is "
        "kept as an offline baseline rather than as a recommendation. "
        "'anthropic' is the earlier Claude path, over a retrieved shortlist; "
        "needs ANTHROPIC_API_KEY (default: openai)",
    )
    parser.add_argument(
        "--ai-model",
        default=None,
        help="Override the ChatGPT model, e.g. gpt-5.1 or gpt-5-mini. The "
        f"default is {ClassifySettings().openai_model}, which every account "
        "with API access has; newer siblings are better and not universally "
        "available. Ignored unless --classifier openai.",
    )
    parser.add_argument(
        "--transcribe-only",
        action="store_true",
        help="Stop after transcription and print what each file produced. No "
        "classifier is constructed and no API key is read, so this verifies "
        "the ASR half on its own before any naming decision depends on it.",
    )
    parser.add_argument(
        "--transcripts",
        action="store_true",
        help="Write a readable .md transcript per recording into a Transcripts/ "
        "subfolder, named to match the renamed video. Until now transcripts "
        "only existed as JSON in the hidden .transcripts/ cache.",
    )
    parser.add_argument(
        "--transcripts-location",
        choices=["alongside", "subfolder"],
        default="subfolder",
        help="Where --transcripts writes. 'subfolder' puts them in a "
        "Transcripts/ directory inside the recording's own chapter folder, so "
        "they travel with the chapter but stay out of the video listing; "
        "'alongside' puts each transcript directly next to its recording. "
        "Mirrors transcripts.location in settings.yaml (default: subfolder)",
    )
    parser.add_argument(
        "--sort-into-chapters",
        action="store_true",
        help="After renaming, move each recording into a subfolder for the "
        "chapter it was classified into, creating it if needed. Files still "
        "marked '(review)' are LEFT IN PLACE - a misfiled recording is harder "
        "to find than a misnamed one, so an uncertain classification does not "
        "get a folder like any other. Pass --hold-review to keep them out.",
    )
    parser.add_argument(
        "--hold-review",
        action="store_true",
        help="With --sort-into-chapters, leave low-confidence recordings in the "
        "folder root instead of filing them, so a human sees them before they "
        "disappear into a chapter. Off by default: the chapter is the part the "
        "classifier is most reliable about, and an unfiled recording in the "
        "root is not more findable than a filed one.",
    )
    parser.add_argument(
        "--review-marker",
        action="store_true",
        help="Append ' (review)' to the name of any recording that came in "
        "below the confidence threshold. Off by default - the flag is still "
        "recorded in rename_plan.json under 'low_confidence', which is where "
        "it can be read without it being carried in the filename forever.",
    )
    parser.add_argument(
        "--apply", action="store_true", help="Actually rename. Default is a dry run."
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    logging.getLogger("pipeline.syllabus.taxonomy").setLevel(logging.ERROR)

    if not args.folder.is_dir():
        raise SystemExit(f"not a folder: {args.folder}")

    taxonomy = load_taxonomy(args.syllabus)
    # Kept before scoping. The language probe scores *subject* vocabulary, and
    # the question it answers - "does this teacher's audio decode better as
    # English or as translation" - is a property of the recording, not of the
    # chapter this folder happens to be. Probing a scoped taxonomy would judge
    # a three-minute clip on the handful of keywords in one chapter, which is
    # too little signal to decide anything, and would file the answer under a
    # subject id that does not describe how it was reached.
    full_taxonomy = taxonomy
    scope = args.scope or infer_scope(taxonomy, args.folder)
    if scope:
        if not args.scope:
            log.info("scope inferred from the folder name: %s", scope)
        taxonomy = scoped_taxonomy(taxonomy, scope)
    else:
        log.info(
            "no scope: considering the whole syllabus. Pass --scope <node id> if "
            "you know which chapter this folder is - it is the single largest "
            "accuracy win available here."
        )

    # Resolved against the scoped taxonomy, so the level and the candidate set
    # are decided from the same tree.
    if args.level:
        level = args.level
        log.info("level: %s (from --level)", level)
    else:
        level, reason = default_level(taxonomy)
        log.info("level: %s (default) - %s. Pass --level to override.", level, reason)

    allowed = nodes_at_level(taxonomy, level)

    classify_settings = ClassifySettings(
        **({"openai_model": args.ai_model} if args.ai_model else {})
    )
    # No review marker by default. The confidence flag still exists and is
    # still written to rename_plan.json; what changed is that it stops being
    # carried in the filename, where it is permanent, visible to students, and
    # says nothing about the recording - only about how sure a classifier was
    # on the day it ran. Pass --review-marker to put it back.
    naming = NamingSettings(
        low_confidence_marker=" (review)" if args.review_marker else ""
    )
    cache_dir = args.cache or (args.folder / ".transcripts")

    files = discover_media(args.folder)
    # Names are unique within a folder tree, so this is the map back to the
    # real path once a recording is no longer in the folder root.
    by_name = {p.name: p for p in files}
    if not files:
        raise SystemExit(f"no media files in {args.folder}")
    # Chronological, not alphabetical. Collision handling numbers by iteration
    # order, so two lectures on one topic become "Part 1"/"Part 2" in the order
    # they were taught. Sorting by name would also put "Aug, 11" before
    # "Aug, 2" on any re-run over already-renamed files.
    files.sort(key=lambda p: (recorded_at_for(p)[0], p.name))

    model_holder: dict = {}
    unreadable: list[str] = []
    transcripts: dict[str, Transcript] = {}

    # One decision for the whole folder, made here so that a concrete language
    # - never None - is what reaches faster-whisper.
    decode = resolve_decode(
        args.language, files, full_taxonomy, cache_dir,
        model_holder=model_holder, model_name=args.model, reprobe=args.reprobe,
    )
    if decode.task == "translate":
        log.info(
            "  decoding with task=translate: the transcript will be English prose, "
            "not a verbatim record of what was said."
        )
    log.info("")

    if args.transcribe_only:
        log.info("%d file(s) in %s", len(files), args.folder)
        log.info("transcribe-only: no classifier, no API key, nothing renamed.\n")
        for path in files:
            log.info("%s", path.name)
            try:
                transcript = transcribe(
                    path, cache_dir, args.model, taxonomy, model_holder=model_holder,
                    language=decode.language, task=decode.task,
                )
            except UnreadableMedia as error:
                log.error("  SKIPPED - %s", error)
                unreadable.append(path.name)
                log.info("")
                continue
            report_transcript(transcript)
            log.info("")
        return _report_unreadable(unreadable, len(files))

    # Constructed only past this point: --transcribe-only must not need an API
    # key, and every model-backed classifier resolves credentials as soon as it
    # is asked to classify.
    retriever = HybridRetriever(taxonomy, classify_settings, RetrievalSettings())
    # .env is read into a Secrets object rather than into os.environ, so a key
    # that lives only in .env is invisible to a client that reads the
    # environment for itself. Passed explicitly for that reason; a key already
    # exported in the environment still works, because None means "look it up".
    secrets = load_secrets()
    api_key = (
        secrets.openai_api_key if args.classifier == "openai" else secrets.anthropic_api_key
    )
    # Checked here, before the first lecture is transcribed. Discovering a
    # missing key after forty minutes of Whisper is a bad way to find out, and
    # the transcript cache means the retry is instant either way — but the
    # operator should not have to know that to trust the tool.
    if args.classifier == "openai" and not (api_key or os.environ.get("OPENAI_API_KEY")):
        raise SystemExit(
            "--classifier openai needs an OpenAI API key. Put OPENAI_API_KEY=sk-... "
            "in .env at the repo root, or export it in this shell. Pass "
            "--classifier keyword to run offline on keyword matching instead."
        )
    if args.classifier == "anthropic" and not (api_key or os.environ.get("ANTHROPIC_API_KEY")):
        raise SystemExit(
            "--classifier anthropic needs ANTHROPIC_API_KEY in .env or in this shell."
        )
    classifier = build_classifier(
        taxonomy,
        classify_settings,
        naming,
        provider=args.classifier,
        api_key=api_key,
        # The naming level, handed to the classifier rather than applied to its
        # output. Asking ChatGPT to choose among the lessons and then silently
        # substituting their subtopic would report a confidence in a decision
        # that was not the one taken.
        allowed_ids=allowed,
    )

    log.info("%d file(s) in %s", len(files), args.folder)
    log.info("syllabus: %s  |  %d candidate nodes at level '%s'  |  classifier: %s%s\n",
             taxonomy.subject_name, len(allowed), level, classifier.name,
             f" ({classify_settings.openai_model})" if args.classifier == "openai" else "")

    taken: set[str] = set()
    plan: list[dict] = []

    for path in files:
        log.info("%s", path.name)
        recorded_at, date_source = recorded_at_for(path)
        try:
            transcript = transcribe(
                path, cache_dir, args.model, taxonomy, model_holder=model_holder,
                language=decode.language, task=decode.task,
            )
        except UnreadableMedia as error:
            log.error("  SKIPPED - %s", error)
            unreadable.append(path.name)
            log.info("")
            continue

        recording = Recording(
            id=path.stem,
            source=Source.MANUAL,
            source_id=path.name,
            subject_id=taxonomy.subject_id,
            original_filename=path.name,
            recorded_at=recorded_at,
            duration_s=transcript.duration_s,
            teacher=args.teacher,
        )

        transcripts[path.name] = transcript

        candidates = retriever.retrieve(transcript)
        # Drop candidates deeper than the requested naming level *before*
        # classifying, so confidence reflects the choice actually being made.
        candidates.candidates = [c for c in candidates.candidates if c.node.id in allowed]
        try:
            classification = classifier.classify(recording, transcript, candidates)
        except Exception as error:  # noqa: BLE001 - one bad call, not a bad batch
            # Same reasoning as UnreadableMedia above: a folder is a term of
            # lectures, and a single failed API call should not cost the
            # nineteen after it. The credential and model checks that would
            # fail on *every* file already ran before transcription started, so
            # what reaches here is the transient kind - a malformed reply, a
            # timeout that outlasted the SDK's retries.
            #
            # The recording is not skipped: it gets a classification that
            # abstains, which routes it to review with the reason attached. A
            # file the tool could not think about must not be quietly left
            # looking like one it considered and had no opinion on.
            log.error("  classifier failed: %s: %s", type(error).__name__, error)
            classification = Classification(
                abstain=True,
                abstain_reason=f"{classifier.name} classifier failed: "
                f"{type(error).__name__}: {error}",
                method=classifier.name,
            )

        # Delegate to the model rather than restating the rule. This block used
        # to reimplement it and dropped the coverage floor, which is the gate
        # that catches the dangerous case: high confidence on a topic that
        # occupied almost none of the lecture. A 56-minute course-orientation
        # class was filed unattended at 0.92 with coverage 0.125 because of it -
        # the plan recorded that 87% of the lecture matched nothing, and filed
        # it anyway. Two copies of a safety rule is one copy of a safety rule.
        low = classification.needs_review(
            classify_settings.review_threshold, classify_settings.coverage_floor
        )
        name_plan = build_name(
            recording, classification, taxonomy, naming,
            existing_names=taken, is_low_confidence=low,
        )
        taken.add(name_plan.filename)

        top = candidates.top(3)
        conf = classification.primary.confidence if classification.primary else 0.0
        chapter = chapter_report(taxonomy, classification)
        log.info("  date      %s  (from %s)", recorded_at.strftime("%b %-d, %Y")
                 if sys.platform != "win32" else recorded_at.strftime("%b %#d, %Y"), date_source)
        log.info("  duration  %.0f min, %d segments", transcript.duration_s / 60,
                 len(transcript.segments))
        # The chapter decision is the one that can misfile a recording, and it
        # is invisible in the filename, so it is printed rather than left to
        # whoever thinks to open rename_plan.json.
        if chapter["id"] is None:
            log.info("  chapter   (none - not placed)%s",
                     f"   top candidates: {'  |  '.join(chapter['top_candidates'])}"
                     if chapter["top_candidates"] else "")
        else:
            log.info("  chapter   %s %s%s", chapter["title"],
                     f"{chapter['weight']:.0%}" if chapter["weight"] is not None else "",
                     f"   runner-up: {chapter['runner_up']}" if chapter["runner_up"] else "")
        # Named "keywords", not "ranking", because on the AI path it is not the
        # ranking anything was decided by - it is the keyword evidence, printed
        # so the two can be seen to disagree. That disagreement is the whole
        # reason the AI classifier exists, and hiding it would make the tool
        # harder to trust rather than easier.
        log.info("  keywords  %s", "  |  ".join(
            f"{c.node.label} {c.coverage:.0%}" for c in top) or "nothing matched")
        if classification.topic_node_ids:
            # Exactly the subtopics that end up in the filename, in order, with
            # the coverage each was credited with. `secondary` also carries
            # reporting-only matches on the keyword path, so it is the wrong
            # list to print here.
            covered = {
                m.node_id: m.coverage
                for m in [classification.primary, *classification.secondary]
                if m is not None
            }
            log.info("  subtopics %s", "  |  ".join(
                f"{node.label} {covered.get(node_id, 0.0):.0%}"
                for node_id in classification.topic_node_ids
                if (node := taxonomy.get(node_id)) is not None
            ))
        if classification.reasoning:
            log.info("  why       %s", classification.reasoning.replace("\n", " ")[:300])
        if classification.abstain_reason:
            log.info("  abstained %s", classification.abstain_reason)
        # ASCII only: the Windows console is cp1252 and mangles arrows/dashes.
        log.info("  ->  %s   (confidence %.2f%s)", name_plan.filename, conf,
                 ", REVIEW" if low else "")
        log.info("")

        plan.append(
            {
                "from": path.name,
                "to": name_plan.filename,
                "confidence": round(conf, 3),
                "low_confidence": low,
                "node_id": classification.primary.node_id if classification.primary else None,
                "coverage": round(classification.primary.coverage, 3)
                if classification.primary
                else 0.0,
                "chapter_node_id": classification.chapter_node_id,
                "chapter_title": chapter["title"],
                # The folder recorded in the overlay, when there is one. Sorting
                # must reproduce the tree that already exists in Drive, not
                # invent a parallel one named after the chapter title.
                "chapter_folder": (
                    node.drive_folder_name
                    if (node := taxonomy.get(classification.chapter_node_id or ""))
                    else None
                ),
                "chapter_weight": chapter["weight"],
                "chapter_runner_up": chapter["runner_up"],
                # Only ever populated when no chapter was chosen: the ranking
                # a reviewer needs, without implying a decision was made.
                "chapter_top_candidates": chapter["top_candidates"],
                "topic_node_ids": classification.topic_node_ids,
                "recorded_at": recorded_at.isoformat(),
                "date_source": date_source,
                # How the audio was decoded. In the plan because it is the one
                # upstream choice that can invalidate every number below it: a
                # transcript decoded the wrong way scores badly against every
                # candidate, and the ranking that results looks like a
                # classifier problem rather than an ASR one.
                "decode": {
                    "language": decode.language,
                    "task": decode.task,
                    "detected": decode.detected,
                    "reason": decode.reason,
                },
                "ranking": [
                    {"node": c.node.label, "id": c.node.id, "coverage": round(c.coverage, 3)}
                    for c in top
                ],
                "evidence": [
                    {"start_s": e.start_s, "quote": e.quote}
                    for e in (classification.primary.evidence if classification.primary else [])
                ],
                "reasoning": classification.reasoning,
            }
        )

    # Consecutive lectures landing on the same lesson is worth saying out loud.
    # It is either a genuine two-parter or — more often with the keyword
    # classifier — the second lecture opening with a recap of the first and the
    # recap outweighing the new material. The tool cannot tell those apart
    # without reading the transcript, so it reports the pattern rather than
    # guessing at it.
    # strict=False is correct here: the two slices are deliberately unequal.
    for earlier, later in zip(plan, plan[1:], strict=False):
        if earlier["node_id"] and earlier["node_id"] == later["node_id"]:
            log.info(
                "note: %r and %r both classified as %r. Either a genuine two-parter, "
                "or the later one opens with a recap that outweighed its new "
                "material - check it before applying.",
                earlier["from"], later["from"], later["ranking"][0]["node"],
            )

    plan_path = args.folder / "rename_plan.json"
    plan_path.write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
    log.info("plan written to %s", plan_path)

    flagged = [row for row in plan if row["low_confidence"]]
    if flagged:
        if naming.low_confidence_marker:
            log.info(
                "%d of %d below the review threshold - marked '%s' in the new name",
                len(flagged), len(plan), naming.low_confidence_marker.strip(),
            )
        else:
            # Said out loud even though nothing in the folder will show it.
            # A confidence flag that is recorded and never surfaced is the same
            # as one that was not recorded.
            log.info(
                "%d of %d below the review threshold - filed anyway, unmarked. "
                "They are the rows with \"low_confidence\": true in %s:",
                len(flagged), len(plan), plan_path.name,
            )
            for row in flagged:
                log.info("    %.2f  %s", row["confidence"], row["to"])

    if not args.apply:
        log.info("\nDry run. Nothing renamed. Re-run with --apply to perform these renames.")
        return 0

    # Rename via a temporary name first: a plan can legitimately swap two
    # names, and a direct rename would clobber one of them.
    undo: list[dict] = []
    for row in plan:
        src = by_name[row["from"]]
        if row["from"] == row["to"]:
            continue
        staging = src.with_name(f".renaming-{row['from']}")
        src.rename(staging)
        undo.append(
            {"staging": str(staging), "to": row["to"], "from": row["from"]}
        )
    for step in undo:
        staged = Path(step["staging"])
        staged.rename(staged.with_name(step["to"]))

    undo_path = args.folder / "rename_undo.json"
    # Merge rather than overwrite. A second --apply run over an
    # already-renamed folder legitimately renames nothing, and a naive write
    # would replace a real undo mapping with an empty one - destroying the only
    # record of the original filenames precisely when the operator is most
    # likely to want it back.
    previous: list[dict] = []
    if undo_path.exists():
        stored = json.loads(undo_path.read_text(encoding="utf-8"))
        previous = stored.get("renames", []) if isinstance(stored, dict) else stored
    fresh = [{"from": step["to"], "to": step["from"]} for step in undo]
    known = {row["from"] for row in fresh}
    merged = fresh + [row for row in previous if row["from"] not in known]
    undo_path.write_text(json.dumps({"renames": merged}, indent=1), encoding="utf-8")
    log.info("renamed %d file(s). Undo plan: %s", len(undo), undo_path)

    # Where each file currently lives, so an "alongside" transcript follows it -
    # including files already sitting in a chapter folder from an earlier run.
    placed: dict[str, Path] = {
        row["to"]: by_name[row["from"]].parent for row in plan if row["from"] in by_name
    }
    moves: list[dict] = []
    if args.sort_into_chapters:
        log.info("")
        log.info("sorting into chapter folders:")
        moves = sort_into_chapters(
            args.folder, plan, include_reviewed=not args.hold_review,
            strip=naming.strip_characters, placed=placed,
            current={row['to']: placed[row['to']] / row['to'] for row in plan},
        )

    # After sorting, so an "alongside" transcript lands in the chapter folder
    # its recording was just moved into rather than in the folder root.
    if args.transcripts:
        write_transcript_files(
            args.folder, plan, transcripts, args.teacher, taxonomy.subject_id,
            location=args.transcripts_location,
            subfolder_name="Transcripts",
            placed=placed,
        )

    if args.sort_into_chapters:
        if moves:
            undo_path.write_text(
                json.dumps({"renames": merged, "moves": moves}, indent=1),
                encoding="utf-8",
            )
        log.info("moved %d file(s); undo records both steps in %s", len(moves), undo_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
