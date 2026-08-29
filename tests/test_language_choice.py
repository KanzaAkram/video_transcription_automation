"""How a code-switched lecture is decoded, and how that choice reaches Whisper.

The decision these tests pin down is not cosmetic. Letting Whisper auto-detect
these lectures produced Devanagari at 0.3x realtime; forcing English produced
usable text at 5.2x. So the rule "never decode at the detected language when
that language is not English" is load-bearing, and a regression would be
invisible until a whole term of recordings had been filed from unusable
transcripts.

Two halves. The first tests `choose_decode` - which decode wins, and on what
evidence. The second tests the command line, because for a while the first half
was entirely unreachable from it: `rename_folder.py` mapped `--language auto`
to None and handed that to faster-whisper, which is precisely the setting this
module was written to avoid. A correct algorithm nobody calls is not a
safeguard, and nothing in the first half could tell.

The ASR model is stubbed throughout: what is under test is the choice, not
Whisper.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from pipeline.asr.chunked import TranscriptionInfo
from pipeline.asr.language import (
    MIN_RELATIVE_MARGIN,
    SAMPLE_RATE,
    SAMPLE_SECONDS,
    DecodeChoice,
    choose_decode,
    syllabus_density,
)


class StubModel:
    """Returns a fixed detection and a canned transcript per (language, task)."""

    def __init__(self, detected: str, probability: float, texts: dict[tuple[str, str], str]):
        self._detected = detected
        self._probability = probability
        self._texts = texts
        self.decodes: list[tuple[str, str]] = []
        self.detections: list[dict] = []

    def detect_language(self, audio=None, **kwargs):
        self.detections.append({"samples": 0 if audio is None else len(audio), **kwargs})
        return self._detected, self._probability, []

    def transcribe(self, _clip, *, language: str, task: str, **__):
        self.decodes.append((language, task))

        class _Segment:
            text = self._texts.get((language, task), "")

        return [_Segment()], None


def _audio(minutes: float = 50.0):
    return np.zeros(int(minutes * 60 * 16000), dtype="float32")


def test_english_audio_is_never_probed(taxonomy):
    """No bake-off when the audio is already English - it would cost a minute
    to answer a question nobody asked."""
    model = StubModel("en", 0.98, {})
    choice = choose_decode(model, _audio(), taxonomy)

    assert (choice.language, choice.task) == ("en", "transcribe")
    assert model.decodes == []


def test_non_english_is_never_decoded_in_the_detected_language(taxonomy):
    """The whole point of the module. Decoding Urdu-English as Hindi produced
    Devanagari, which matches no syllabus keyword at all."""
    model = StubModel("hi", 0.94, {("en", "transcribe"): "isotopes and mass spectrometry"})
    choice = choose_decode(model, _audio(), taxonomy)

    assert choice.language == "en"
    assert choice.detected == "hi"


def test_detection_reads_the_whole_sample_not_one_window(taxonomy):
    """faster-whisper looks at a single 30-second window by default, and one
    window does not describe a code-switched lecture. The 10 Aug Maths
    recording detects `en` at 0.40 from the window at 34:55 and `hi` at 0.87
    from the window at 17:47 - the same lecture, the same teacher. Everything
    below this line is gated on the answer, so it must not be decided by which
    half-minute the sampler happened to land in."""
    model = StubModel("hi", 0.94, {("en", "transcribe"): "isotopes"})
    choose_decode(model, _audio(), taxonomy)

    (detection,) = model.detections
    assert detection["language_detection_segments"] > 1
    assert detection["language_detection_segments"] * 30 <= detection["samples"] / SAMPLE_RATE
    # faster-whisper stops at the first window that beats this, so anything
    # reachable turns the vote back into "whichever window came first".
    assert detection["language_detection_threshold"] > 1.0 - 1e-9


def test_unconfident_detection_falls_back_to_english(taxonomy):
    """A weak non-English guess is worse than no guess: acting on it risks
    non-Latin output for audio that was mostly English anyway."""
    model = StubModel("hi", 0.31, {})
    choice = choose_decode(model, _audio(), taxonomy)

    assert (choice.language, choice.task) == ("en", "transcribe")
    assert "below" in choice.reason
    assert model.decodes == []


def test_the_sample_is_drawn_from_across_the_lecture(taxonomy):
    """One window measures which three minutes it landed in, not which decode
    is better. Run at five offsets through the 11 Aug Chemistry lecture the
    one-window probe reported margins of 11.5%, 15.5%, 1.7%, 3.0% and 5.2% -
    with the winner changing. Worked examples are dense in syllabus vocabulary
    and a digression about the exam timetable has none, and a lecture is both.

    Marking each window of the audio and checking which marks survive is the
    only way to assert this: the probe hands one concatenated array to the
    model, so the seams are invisible by the time anything else can look.
    """
    minutes = 50
    # Every frame carries the minute it belongs to, so the clip can be asked
    # where it came from.
    audio = np.repeat(np.arange(minutes, dtype="float32"), 60 * SAMPLE_RATE)
    model = StubModel("hi", 0.94, {("en", "transcribe"): "isotopes"})
    seen: list = []
    original = model.transcribe

    def watch(clip, **kwargs):
        seen.append(clip)
        return original(clip, **kwargs)

    model.transcribe = watch
    choose_decode(model, audio, taxonomy)

    clip = seen[0]
    assert len(clip) == SAMPLE_SECONDS * SAMPLE_RATE
    covered = sorted({int(minute) for minute in clip[::SAMPLE_RATE]})
    assert max(covered) - min(covered) > 20, (
        f"the sample spans only minutes {covered}, which is one place in the lecture"
    )


def test_translation_wins_only_when_it_clears_the_margin(taxonomy):
    """Translation has to earn it. It round-trips technical terms through a
    paraphrase, so a marginal win is not a reason to prefer it."""
    rich = "isotopes relative atomic mass carbon-12 abundance nucleon number"
    model = StubModel(
        "hi",
        0.94,
        {("en", "transcribe"): "one two three", ("hi", "translate"): rich},
    )
    choice = choose_decode(model, _audio(), taxonomy)

    assert (choice.language, choice.task) == ("hi", "translate")


def test_a_noise_sized_margin_does_not_decide(taxonomy):
    """Measured on the real 11 Aug lecture the two candidates scored 12.9 and
    13.1 terms per thousand words - a 2% gap. Reporting a winner there dresses
    a coin flip up as a measurement.

    The gap is built out of *length*, not vocabulary: both decodes recover the
    same syllabus terms and translation is the terser of the two, so it wins by
    one word in fifteen. Constructing the gap the other way - giving one decode
    an extra keyword - would tie this test to which terms the overlays happen
    to carry today, and it broke exactly that way once already when `isotopes`
    moved out of chem.1_2.
    """
    terms = "so the relative atomic mass of chlorine works out as thirty five point five"
    model = StubModel(
        "hi",
        0.94,
        {("en", "transcribe"): terms + " okay", ("hi", "translate"): terms},
    )
    choice = choose_decode(model, _audio(), taxonomy)

    top, runner_up = sorted(choice.scores.values(), reverse=True)
    assert (top - runner_up) / top < MIN_RELATIVE_MARGIN
    assert (choice.language, choice.task) == ("en", "transcribe")
    assert "inconclusive" in choice.reason


def test_no_syllabus_signal_falls_back_to_english(taxonomy):
    """Neither decode found anything to measure, so the metric is silent.
    English at least keeps the output in a script that can be matched."""
    model = StubModel(
        "hi", 0.94, {("en", "transcribe"): "hello there", ("hi", "translate"): "good morning"}
    )
    choice = choose_decode(model, _audio(), taxonomy)

    assert (choice.language, choice.task) == ("en", "transcribe")
    assert "no syllabus terms" in choice.reason


def test_probe_is_memoised_per_subject_not_per_language(taxonomy):
    """The score counts *that subject's* vocabulary, so a decision reached for
    Chemistry is not evidence about Maths even when both detect Hindi."""
    model = StubModel("hi", 0.94, {("en", "transcribe"): "isotopes"})
    cache: dict = {}

    choose_decode(model, _audio(), taxonomy, cache=cache)
    first = len(model.decodes)
    choose_decode(model, _audio(), taxonomy, cache=cache)

    assert len(model.decodes) == first, "same subject should reuse the probe"

    other = taxonomy.model_copy(update={"subject_id": "ig_maths"}, deep=True)
    choose_decode(model, _audio(), other, cache=cache)

    assert len(model.decodes) > first, "a different subject must be probed again"


@pytest.mark.parametrize(
    ("text", "expected"),
    [("relative atomic mass", True), ("relative mass of the atom", False)],
)
def test_density_requires_every_token_of_a_phrase(text, expected):
    """Matches how classify.retriever scores keywords, so the probe measures
    the same thing the classifier will."""
    assert (syllabus_density(text, [["relative", "atomic", "mass"]]) > 0) is expected


# ── The CLI ───────────────────────────────────────────────────────────────────
#
# `choose_decode` was written and tested and then not called. `rename_folder.py`
# passed `--language` straight to faster-whisper and mapped "auto" to None,
# which is the single setting this module exists to prevent, so the
# auto-detection the tests above pin down was never reached from the command
# line. What follows tests the wire rather than the algorithm: that a decision
# is made, that it arrives at the decoder, and that None never does.


def _script():
    """Import rename_folder.py, which is a script rather than a package module."""
    spec = importlib.util.spec_from_file_location(
        "rename_folder", Path(__file__).resolve().parents[1] / "scripts" / "rename_folder.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Registered before exec so dataclasses in the module can resolve their
    # annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rename = _script()


@pytest.fixture
def media(tmp_path):
    """Three plausible recordings of different sizes. Never actually decoded."""
    made = []
    for name, size in {"long.mp4": 9000, "short.mp4": 1000, "mid.mp4": 4000}.items():
        path = tmp_path / name
        path.write_bytes(b"\0" * size)
        made.append(path)
    return made


class RecordingProbe:
    """Stands in for choose_decode, remembering whether it was called at all."""

    def __init__(self, choice=None):
        self.choice = choice
        self.calls = 0

    def __call__(self, model, audio, taxonomy, **_):
        self.calls += 1
        return self.choice


def _choice(language="en", task="transcribe", **kwargs):
    return DecodeChoice(
        language=language,
        task=task,
        detected=kwargs.pop("detected", "hi"),
        detected_probability=kwargs.pop("detected_probability", 0.94),
        reason=kwargs.pop("reason", "probe inconclusive"),
        scores=kwargs.pop("scores", {}),
    )


def _forbid_asr(monkeypatch):
    """Make any attempt to load a model or decode audio a loud failure."""

    def boom(*_, **__):
        raise AssertionError("the ASR model must not be touched here")

    monkeypatch.setattr(rename, "load_model", boom)
    monkeypatch.setattr(rename, "_decode_audio", boom)


def _stub_asr(monkeypatch):
    monkeypatch.setattr(rename, "load_model", lambda holder, name: "model")
    monkeypatch.setattr(rename, "_decode_audio", lambda path, attempts=3: np.zeros(16000))


def test_an_explicit_language_bypasses_the_probe(tmp_path, media, taxonomy, monkeypatch):
    """`--language en` is the measured default and has to stay free. Probing to
    confirm what the operator already said would cost a model load and several
    minutes of decoding per folder for no information."""
    _forbid_asr(monkeypatch)
    probe = RecordingProbe()
    monkeypatch.setattr(rename, "choose_decode", probe)

    choice = rename.resolve_decode(
        "en", media, taxonomy, tmp_path / ".transcripts",
        model_holder={}, model_name="small",
    )

    assert (choice.language, choice.task) == ("en", "transcribe")
    assert probe.calls == 0
    assert not (tmp_path / ".transcripts" / rename.DECISION_FILE).exists()


def test_an_explicit_non_english_language_is_honoured(tmp_path, media, taxonomy, monkeypatch):
    """The override has to be a real override. A class genuinely taught in Urdu
    is the case `--language ur` exists for, and quietly substituting English
    would be worse than not offering the flag."""
    _forbid_asr(monkeypatch)
    monkeypatch.setattr(rename, "choose_decode", RecordingProbe())

    choice = rename.resolve_decode(
        "UR", media, taxonomy, tmp_path / ".transcripts",
        model_holder={}, model_name="small",
    )

    assert (choice.language, choice.task) == ("ur", "transcribe")


def test_auto_routes_through_choose_decode(tmp_path, media, taxonomy, monkeypatch):
    """The wire. 'auto' means run the bake-off in `choose_decode`; it does not
    mean hand 'detect it yourself' to the decoder."""
    _stub_asr(monkeypatch)
    probe = RecordingProbe(_choice("hi", "translate", reason="probe chose translate-hi by 31%"))
    monkeypatch.setattr(rename, "choose_decode", probe)

    choice = rename.resolve_decode(
        "auto", media, taxonomy, tmp_path / ".transcripts",
        model_holder={}, model_name="small",
    )

    assert probe.calls == 1
    assert (choice.language, choice.task) == ("hi", "translate")


def test_auto_never_resolves_to_none(tmp_path, media, taxonomy, monkeypatch):
    """`language=None` is what auto-detect means to faster-whisper: Devanagari
    at 0.3x realtime. Whatever the probe decides, a code is what leaves here."""
    _stub_asr(monkeypatch)
    monkeypatch.setattr(rename, "choose_decode", RecordingProbe(_choice()))

    choice = rename.resolve_decode(
        "auto", media, taxonomy, tmp_path / ".transcripts",
        model_holder={}, model_name="small",
    )

    assert isinstance(choice.language, str) and choice.language


def test_the_decoder_is_never_offered_none(tmp_path, taxonomy):
    """A wall rather than a preference: even a caller that skipped
    `resolve_decode` entirely cannot get an auto-detecting decode out of this
    script. The failure has to be an exception, because the alternative is a
    folder of Devanagari transcripts that took seventeen times as long."""
    media = tmp_path / "lecture.mp4"
    media.write_bytes(b"\0" * 4096)

    with pytest.raises(ValueError, match="explicit language"):
        rename.transcribe(
            media, tmp_path / ".transcripts", "small", taxonomy,
            model_holder={}, language=None,
        )


def test_the_chosen_decode_reaches_the_decoder(tmp_path, taxonomy, monkeypatch):
    """The other end of the wire, and the shape of the bug this fixes: a
    decision that is computed and logged but not passed on is invisible,
    because the console prints the choice either way."""
    media = tmp_path / "lecture.mp4"
    media.write_bytes(b"\0" * 4096)
    _stub_asr(monkeypatch)
    seen: dict = {}

    def fake_transcribe_long(model, audio, **kwargs):
        seen.update(kwargs)
        return [], TranscriptionInfo(duration=60.0, language=kwargs["language"])

    monkeypatch.setattr(rename, "transcribe_long", fake_transcribe_long)

    rename.transcribe(
        media, tmp_path / ".transcripts", "small", taxonomy,
        model_holder={}, language="hi", task="translate",
    )

    assert (seen["language"], seen["task"]) == ("hi", "translate")


def test_a_translated_transcript_is_cached_apart_from_an_english_one(
    tmp_path, taxonomy, monkeypatch
):
    """Two ways of decoding one lecture are two different transcripts. Sharing
    a cache entry would silently serve whichever was produced first, and the
    console would report the language that was *asked for*."""
    media = tmp_path / "lecture.mp4"
    media.write_bytes(b"\0" * 4096)
    cache = tmp_path / ".transcripts"
    _stub_asr(monkeypatch)
    monkeypatch.setattr(
        rename, "transcribe_long",
        lambda model, audio, **kw: ([], TranscriptionInfo(duration=60.0, language=kw["language"])),
    )

    rename.transcribe(media, cache, "small", taxonomy, model_holder={},
                      language="en", task="transcribe")
    rename.transcribe(media, cache, "small", taxonomy, model_holder={},
                      language="hi", task="translate")

    names = sorted(p.name.split(".", 1)[1] for p in cache.glob("*.json"))
    assert names == ["small.en-transcribe.json", "small.hi-translate.json"]


def test_the_probe_is_not_repeated_on_every_re_run(tmp_path, media, taxonomy, monkeypatch):
    """Re-running is the normal way this script is used - a different scope, a
    different level, a different classifier - and every re-run is instant off
    the transcript cache. A probe that reloaded the model each time would turn
    the cheapest operation in the tool into the most expensive one."""
    _stub_asr(monkeypatch)
    probe = RecordingProbe(_choice("hi", "translate", scores={"translate-hi": 13.0}))
    monkeypatch.setattr(rename, "choose_decode", probe)
    cache = tmp_path / ".transcripts"

    first = rename.resolve_decode("auto", media, taxonomy, cache,
                                  model_holder={}, model_name="small")
    second = rename.resolve_decode("auto", media, taxonomy, cache,
                                   model_holder={}, model_name="small")

    assert probe.calls == 1, "the remembered decision should be reused"
    assert (second.language, second.task) == (first.language, first.task)
    assert second.scores == first.scores, "the evidence is remembered, not just the verdict"

    rename.resolve_decode("auto", media, taxonomy, cache,
                          model_holder={}, model_name="small", reprobe=True)
    assert probe.calls == 2, "--reprobe has to actually re-probe"


def test_a_remembered_decision_is_per_subject(tmp_path, media, taxonomy, monkeypatch):
    """The same reason `choose_decode` memoises per subject: the score counts
    *that subject's* vocabulary, so Chemistry's answer is not evidence about
    Maths even when both detect Hindi."""
    _stub_asr(monkeypatch)
    probe = RecordingProbe(_choice())
    monkeypatch.setattr(rename, "choose_decode", probe)
    cache = tmp_path / ".transcripts"

    rename.resolve_decode("auto", media, taxonomy, cache,
                          model_holder={}, model_name="small")
    other = taxonomy.model_copy(update={"subject_id": "igcse_maths"}, deep=True)
    rename.resolve_decode("auto", media, other, cache,
                          model_holder={}, model_name="small")

    assert probe.calls == 2


def test_an_unreadable_decision_file_costs_a_probe_not_the_run(
    tmp_path, media, taxonomy, monkeypatch
):
    """This file lives in a cache directory operators are told they can delete
    freely, so a half-written one is a normal event rather than a corruption."""
    _stub_asr(monkeypatch)
    probe = RecordingProbe(_choice())
    monkeypatch.setattr(rename, "choose_decode", probe)
    cache = tmp_path / ".transcripts"
    cache.mkdir()
    (cache / rename.DECISION_FILE).write_text("{not json", encoding="utf-8")

    choice = rename.resolve_decode("auto", media, taxonomy, cache,
                                   model_holder={}, model_name="small")

    assert probe.calls == 1
    assert choice.language == "en"


def test_the_probe_listens_to_the_smallest_recording(media):
    """Reaching the middle of a file means decoding all of it, and that decode
    is the probe's whole cost. Every recording in the folder is the same
    teacher and subject, so the cheapest one answers the question as well as
    the 100-minute one - which on this machine is a 380 MB allocation."""
    assert [p.name for p in rename.probe_order(media)] == ["short.mp4", "mid.mp4", "long.mp4"]


def test_an_undecodable_probe_file_does_not_abort_the_run(
    tmp_path, media, taxonomy, monkeypatch
):
    """"Smallest" and "broken" are correlated, not independent: an interrupted
    upload leaves a *short* file with no moov atom, so the cheapest candidate is
    disproportionately likely to be the unreadable one. It is in the real Maths
    folder. Insisting on it would abort a run whose other four recordings are
    fine, before a single lecture had been looked at."""
    monkeypatch.setattr(rename, "load_model", lambda holder, name: "model")
    tried: list[str] = []

    def decode(path, attempts=3):
        tried.append(path.name)
        if path.name == "short.mp4":
            raise rename.UnreadableMedia("short.mp4: no moov atom")
        return np.zeros(16000)

    monkeypatch.setattr(rename, "_decode_audio", decode)
    monkeypatch.setattr(rename, "choose_decode", RecordingProbe(_choice()))

    choice = rename.resolve_decode("auto", media, taxonomy, tmp_path / ".transcripts",
                                   model_holder={}, model_name="small")

    assert tried == ["short.mp4", "mid.mp4"]
    assert choice.language == "en"


def test_a_folder_of_undecodable_media_fails_loudly(tmp_path, media, taxonomy, monkeypatch):
    """The one case where there is nothing to fall back to. Silently assuming
    English here would hide a folder that needs re-downloading behind a
    plausible-looking default."""
    monkeypatch.setattr(rename, "load_model", lambda holder, name: "model")
    monkeypatch.setattr(rename, "_decode_audio", lambda path, attempts=3: (_ for _ in ()).throw(
        rename.UnreadableMedia(f"{path.name}: no moov atom")))
    monkeypatch.setattr(rename, "choose_decode", RecordingProbe(_choice()))

    with pytest.raises(SystemExit, match="none of the recordings"):
        rename.resolve_decode("auto", media, taxonomy, tmp_path / ".transcripts",
                              model_holder={}, model_name="small")


def test_an_empty_language_is_refused_rather_than_defaulted(tmp_path, media, taxonomy):
    """`--language ""` used to become None two lines later. Refusing is the
    only safe reading - it is indistinguishable from a shell quoting mistake,
    and the fallback it used to reach is the unusable one."""
    with pytest.raises(SystemExit):
        rename.resolve_decode("  ", media, taxonomy, tmp_path / ".transcripts",
                              model_holder={}, model_name="small")


def test_the_default_language_is_english():
    """Guarded because it is a measurement, not a preference, and because
    changing it is a one-word edit with folder-wide consequences. On 48 minutes
    of this teaching, transcribing as English and translating recover the same
    syllabus vocabulary to within a few percent - so the option that costs no
    probe and keeps technical terms verbatim wins."""
    parsed = rename.build_parser().parse_args(
        ["--folder", ".", "--syllabus", "s.yaml"]
    )
    assert parsed.language == "en"
    assert parsed.reprobe is False
    assert rename.build_parser().parse_args(
        ["--folder", ".", "--syllabus", "s.yaml", "--language", "auto"]
    ).language == "auto"
