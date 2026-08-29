"""Choosing how to decode a lecture that is not taught in one language.

These lectures are taught in mixed Urdu and English: the explanation is Urdu,
the technical vocabulary is English. That is normal for the classrooms this
pipeline serves, and it breaks the obvious approach in a way worth spelling
out, because the failure is silent and expensive.

**Letting Whisper detect the language is the one option that does not work.**
Measured on a 3-minute clip of the 11 Aug AS Chemistry lecture, `small`, int8
on CPU:

    auto-detect  ->  'hi' at 0.85  ->  Devanagari output   0.3x realtime
    language=en  ->  clean English, terms intact           5.2x realtime
    translate    ->  English, slightly clumsier            5.5x realtime

Two separate problems, both fatal:

1. **The output is unusable.** Retrieval scores English syllabus keywords. A
   transcript in Devanagari matches none of them, so every recording lands in
   review and a transcription failure is indistinguishable from a classifier
   one.

2. **It is seventeen times slower.** An uncertain language sends every window
   through Whisper's temperature-fallback retries - up to six decodes where
   one would do. A 50-minute lecture goes from ten minutes to nearly three
   hours, which is the difference between a week of lectures transcribing over
   lunch and not being able to run this at all.

So the decision is made here rather than delegated to the decoder. Whisper's
own detection is still used - it is a good detector, and knowing the audio is
*not* English is exactly the signal needed - but what happens next is chosen
rather than inherited.

**How the choice is made.** For non-English audio there are two credible
decodes, and which one wins is a property of the speaker, not something that
can be settled once in a docstring:

    transcribe as English   keeps English terms verbatim, renders the Urdu
                            phonetically in Latin script
    translate to English    turns the Urdu into English prose, sometimes at
                            the cost of mangling a technical term

Rather than guess, both are run on one sample and scored on **what the
transcript is actually for**: how much syllabus vocabulary they recover. The
transcript exists to be matched against the taxonomy, so the decode that
recovers more of the taxonomy is better by definition, and the metric cannot
drift away from the goal it stands in for.

The bake-off costs about two minutes and is memoised per subject and detected
language, so a folder of lectures by one teacher pays for it once, not once
per file.

**What the bake-off actually found, on 48 minutes of this teaching.** Four
12-minute clips, two subjects, both decodes each, `small` int8 on CPU:

    AS Chemistry   transcribe 19.7   translate 18.7   terms/1000   en +5.7%
    IGCSE Maths    transcribe 22.8   translate 24.4   terms/1000   translate +7.3%

The two subjects disagree, which is exactly why the memoisation is keyed on
subject. But neither difference is worth acting on, for two reasons. It is
smaller than the probe's own scatter - see MIN_RELATIVE_MARGIN - and it points
the other way from the thing a human would check: counting subject-specific
technical terms across the same clips, transcription leads in *both* subjects
(Chemistry 74 to 69, Maths 88 to 79). Translation writes tidier sentences and
loses content inside them. On the 16 Aug Chemistry clip it rendered "one u is
basically one twelfth the mass of carbon 12" as "1u is basically 112th the
mass of carbon-12" - a cleaner sentence with the fact removed, and `one twelfth`
is a retrieval keyword while `112th` is nothing.

What translation does buy is the removal of untranslated Urdu: on that same
clip, four romanised Urdu function words became zero. Across the seven shipped
transcripts there are 23 such tokens in 34,553 words - 0.07%, in four
sentences, none carrying syllabus vocabulary. That is the size of the problem
translation solves, against the size of the problem it causes.

So the caller's default is `--language en`, and this module exists for the
teacher it has not met yet.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np

from pipeline.syllabus import Taxonomy

log = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-']*")

#: Below this, Whisper is not confident enough for its answer to be worth
#: acting on, and English is the safer assumption: Latin script is at least
#: matchable against the syllabus, whereas a wrong non-English guess is not.
MIN_DETECTION_CONFIDENCE = 0.5

#: Sample length for the bake-off. Long enough to contain real teaching rather
#: than the roll call, short enough that the probe is not itself the job.
SAMPLE_SECONDS = 180

#: Where in the lecture the sample is taken from. Openings are admin - "can
#: you guys hear me", homework, lesson planning - and score badly for every
#: candidate, so the sample starts well past them.
#:
#: Three points rather than one, at no extra cost: the same SAMPLE_SECONDS of
#: audio, spread across the lecture instead of taken from one place. That is
#: not a refinement, it is the difference between a statistic and a coin flip.
#: Running the one-point probe at five offsets through the 11 Aug Chemistry
#: lecture gave margins of 11.5%, 15.5%, 1.7%, 3.0% and 5.2% - and the winner
#: changed. Through the 12 Aug Maths lecture it gave 1.6%, 1.8% and 17.8%,
#: also with the winner changing. A single window is measuring which three
#: minutes it landed in: worked examples are dense in syllabus vocabulary, a
#: digression about the exam timetable has none, and a lecture contains both.
SAMPLE_AT_FRACTIONS = (0.25, 0.50, 0.75)

SAMPLE_RATE = 16000

#: How many 30-second windows of the sample the language detector looks at.
#: faster-whisper defaults to **one**, and one is not enough on code-switched
#: audio. Detection run on single windows of these two lectures returned `en`
#: 0.40, `en` 0.65, `hi` 0.76, `hi` 0.87, `hi` 0.94 and `hi` 0.99 - the same
#: teachers, the same model, different thirty seconds. A lecture that spends
#: two minutes on a worked example in English and the next two explaining it
#: in Urdu has no single-window answer, and this decision gates everything
#: below it.
DETECTION_WINDOWS = SAMPLE_SECONDS // 30

#: Passed to faster-whisper as `language_detection_threshold`, where it means
#: "stop looking once a window is this sure". Set to 1.0 to disable that early
#: exit entirely, so all DETECTION_WINDOWS are classified and the language is
#: the majority across them rather than whichever window came first.
#:
#: Not a confidence threshold - MIN_DETECTION_CONFIDENCE is that, and it is
#: applied to the *result*. This is only about how much evidence gets
#: collected before the vote, and collecting all of it costs six encoder
#: passes over 30 seconds each: fractions of a second against the minutes the
#: decision commits the whole folder to.
DETECT_OVER_WHOLE_SAMPLE = 1.0

#: Relative margin the winning decode must clear for the probe to be believed.
#: Below it the tie is broken for `transcribe`, which keeps English technical
#: terms verbatim instead of round-tripping them through a paraphrase.
#:
#: This was 0.10, set from a 2% difference on one 3-minute clip - a number too
#: small to calibrate a threshold with. Re-measured on 48 minutes of real
#: teaching (four 12-minute clips, two subjects), the difference between the
#: two decodes is:
#:
#:     AS Chemistry   transcribe 19.7  translate 18.7  terms/1000  en +5.7%
#:     IGCSE Maths    transcribe 22.8  translate 24.4  terms/1000  translate +7.3%
#:
#: - opposite answers by subject, and both under 10%. So the *true* effect on
#: this audio lives entirely inside the old threshold, and anything that
#: crossed it was noise by construction. The noise is large: running the probe
#: at five points through the 11 Aug Chemistry lecture produced margins of
#: 11.5%, 15.5%, 1.7%, 3.0% and 5.2% with the winner changing, and spreading
#: the sample over three points (see SAMPLE_AT_FRACTIONS) cut the disagreement
#: without removing it - 4.6%, 10.3% and 20.1% for three choices of where to
#: sample the same lecture.
#:
#: 0.25 is therefore set *above the probe's own scatter* rather than at the
#: size of a difference worth having. It is a guard, not a decision boundary:
#: its job is to catch a decode that is dramatically better - a teacher whose
#: Urdu defeats English transcription outright - and to refuse to arbitrate
#: anything smaller, because at this sample size the probe genuinely cannot.
MIN_RELATIVE_MARGIN = 0.25


@dataclass
class DecodeChoice:
    """How to decode one recording, and the evidence for it."""

    #: Passed to faster-whisper as `language`. Never None: auto-detect at
    #: decode time is the option this module exists to avoid.
    language: str
    #: "transcribe" or "translate".
    task: str
    detected: str
    detected_probability: float
    reason: str
    #: candidate label -> syllabus-term density, when a bake-off was run.
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        return (
            f"detected {self.detected} {self.detected_probability:.2f} -> "
            f"{self.task} as {self.language} ({self.reason})"
        )


def syllabus_terms(taxonomy: Taxonomy) -> list[list[str]]:
    """Syllabus vocabulary as token lists, for the density metric.

    Subject vocabulary plus every node's keywords and aliases - the same text
    retrieval scores against, so the probe optimises for the real objective
    rather than for a proxy of it.
    """
    seen: set[str] = {term.lower() for term in taxonomy.subject_vocabulary}
    for node in taxonomy.all_nodes():
        seen.update(term.lower() for term in (*node.keywords, *node.aliases))
    return [tokens for term in seen if (tokens := _TOKEN.findall(term))]


def syllabus_density(text: str, terms: list[list[str]]) -> float:
    """How many syllabus terms the text contains, per thousand words.

    Normalised by length because the candidate decodes do not produce the same
    word count - translation is terser - and a raw count would simply reward
    whichever one rambled more.

    A multi-word term counts only when every token is present, matching how
    `classify.retriever` scores keywords. The probe therefore measures the
    same thing the classifier will.
    """
    tokens = set(_TOKEN.findall(text.lower()))
    words = max(len(text.split()), 1)
    hits = sum(1 for term in terms if all(token in tokens for token in term))
    return 1000.0 * hits / words


def _cache_key(taxonomy: Taxonomy, detected: str) -> str:
    """Memoisation key for the bake-off.

    Keyed on the subject as well as the language, because the score is a count
    of *that subject's* vocabulary. A decision reached for Chemistry says
    nothing about Maths even when both detect the same language, and reusing
    it would silently apply one subject's evidence to another.
    """
    return f"{taxonomy.subject_id}:{detected}"


def _sample(audio, duration_s: float):
    """SAMPLE_SECONDS of audio, drawn from several points in the lecture.

    Concatenated into one clip rather than decoded separately, so both
    candidates see exactly the same audio including the same two joins, and the
    probe stays two decodes rather than six.
    """
    each = int(SAMPLE_SECONDS / len(SAMPLE_AT_FRACTIONS) * SAMPLE_RATE)
    windows = []
    for fraction in SAMPLE_AT_FRACTIONS:
        start = int(duration_s * fraction * SAMPLE_RATE)
        window = audio[start : start + each]
        if len(window):
            windows.append(window)
    # A recording shorter than the sample has nothing to spread over; it is
    # already its own sample.
    return np.concatenate(windows) if windows else audio


def _decode(model, clip, *, language: str, task: str) -> str:
    segments, _ = model.transcribe(
        clip,
        beam_size=5,
        language=language,
        task=task,
        condition_on_previous_text=False,
    )
    return " ".join(segment.text.strip() for segment in segments)


def choose_decode(
    model,
    audio,
    taxonomy: Taxonomy,
    *,
    cache: dict[str, DecodeChoice] | None = None,
) -> DecodeChoice:
    """Detect the language, then decide how to decode it.

    `cache` memoises the expensive half by detected language, so a folder of
    lectures by one teacher runs the bake-off once rather than once per file.
    """
    duration_s = len(audio) / SAMPLE_RATE
    clip = _sample(audio, duration_s)
    detected, probability, _ = model.detect_language(
        audio=clip,
        language_detection_threshold=DETECT_OVER_WHOLE_SAMPLE,
        language_detection_segments=DETECTION_WINDOWS,
    )

    if detected == "en":
        return DecodeChoice(
            language="en",
            task="transcribe",
            detected=detected,
            detected_probability=probability,
            reason="already English",
        )

    if probability < MIN_DETECTION_CONFIDENCE:
        # An unconfident non-English guess is worse than no guess: acting on it
        # risks non-Latin output for audio that was mostly English anyway.
        return DecodeChoice(
            language="en",
            task="transcribe",
            detected=detected,
            detected_probability=probability,
            reason=f"detection below {MIN_DETECTION_CONFIDENCE:.2f}, assuming English",
        )

    cache = cache if cache is not None else {}
    key = _cache_key(taxonomy, detected)
    if key in cache:
        prior = cache[key]
        return DecodeChoice(
            language=prior.language,
            task=prior.task,
            detected=detected,
            detected_probability=probability,
            reason=f"{prior.reason}, decided earlier for {detected}",
            scores=prior.scores,
        )

    # Code-switched audio. Run both credible decodes on one sample and keep
    # whichever recovers more syllabus vocabulary.
    terms = syllabus_terms(taxonomy)
    candidates = {
        "transcribe-en": ("en", "transcribe"),
        f"translate-{detected}": (detected, "translate"),
    }
    scores: dict[str, float] = {}
    for label, (language, task) in candidates.items():
        text = _decode(model, clip, language=language, task=task)
        scores[label] = syllabus_density(text, terms)
        # debug, not info: the scores are part of the decision, and the
        # decision's presentation belongs to whoever is reporting it - the
        # CLI prints them once, under the verdict they explain, rather than
        # twice in two formats.
        log.debug("probe %-16s %5.1f syllabus terms per 1000 words", label, scores[label])

    best = max(scores, key=lambda label: scores[label])
    top, runner_up = sorted(scores.values(), reverse=True)
    margin = (top - runner_up) / top if top else 0.0

    if top == 0.0:
        # Neither decode found any syllabus vocabulary, so the metric has
        # nothing to say. English keeps the output in Latin script, which at
        # least stays legible and matchable.
        best, reason = "transcribe-en", "probe found no syllabus terms, defaulting to English"
    elif margin < MIN_RELATIVE_MARGIN:
        # The two decodes are within noise of each other. Reporting a winner
        # here would dress a coin flip up as a measurement.
        best = "transcribe-en"
        reason = f"probe inconclusive ({margin:.0%} margin), defaulting to English"
    else:
        reason = f"probe chose {best} by {margin:.0%}"
    language, task = candidates[best]

    choice = DecodeChoice(
        language=language,
        task=task,
        detected=detected,
        detected_probability=probability,
        reason=reason,
        scores=scores,
    )
    cache[_cache_key(taxonomy, detected)] = choice
    return choice
