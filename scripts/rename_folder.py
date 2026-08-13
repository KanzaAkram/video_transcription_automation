"""Rename the recordings in one folder from what is said in them.

    # see what it would do
    python scripts/rename_folder.py --folder "D:/chap 12-Respiration" \
        --syllabus config/syllabi/a2-biology-9700.yaml --scope bio.topic_12

    # do it
    python scripts/rename_folder.py --folder "D:/chap 12-Respiration" \
        --syllabus config/syllabi/a2-biology-9700.yaml --scope bio.topic_12 --apply

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

Transcripts are cached per (file, model), so re-running to try a different
scope or classifier is instant and does not re-transcribe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.classify.retriever import HybridRetriever  # noqa: E402
from pipeline.classify.service import KeywordClassifier, LLMClassifier  # noqa: E402
from pipeline.config import ClassifySettings, NamingSettings, RetrievalSettings  # noqa: E402
from pipeline.models import (  # noqa: E402
    Recording,
    Source,
    Transcript,
    TranscriptSegment,
)
from pipeline.naming import build_name  # noqa: E402
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


def transcribe(
    path: Path,
    cache_dir: Path,
    model_name: str,
    taxonomy: Taxonomy,
    *,
    model_holder: dict,
) -> Transcript:
    """Transcribe with faster-whisper, caching one JSON per (content, model)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{content_key(path)}.{model_name}.json"
    legacy = cache_dir / f"{path.stem}.{model_name}.json"
    if not cached.exists() and legacy.exists():
        legacy.rename(cached)

    if cached.exists():
        payload = json.loads(cached.read_text(encoding="utf-8"))
        log.info("  transcript: cached (%s)", cached.name)
    else:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise SystemExit(
                'faster-whisper is not installed. pip install -e ".[asr]"'
            ) from None

        if "model" not in model_holder:
            log.info("  loading %s ...", model_name)
            model_holder["model"] = WhisperModel(
                model_name, device="cpu", compute_type="int8", cpu_threads=10
            )
        model = model_holder["model"]

        log.info("  transcribing %s (not cached) ...", path.name)
        started = datetime.now()
        segments, info = model.transcribe(
            str(path),
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 700},
            # Priming with syllabus vocabulary. These are the terms generic ASR
            # mangles and the terms retrieval then depends on, so the error
            # would cost twice.
            initial_prompt=taxonomy.initial_prompt(40),
            # Without this one hallucinated line conditions the next, and a
            # lecture can spiral into repeated text.
            condition_on_previous_text=False,
        )
        rows = [
            {
                "start": round(s.start, 2),
                "end": round(s.end, 2),
                "text": s.text.strip(),
                "no_speech_prob": round(getattr(s, "no_speech_prob", 0.0) or 0.0, 3),
            }
            for s in segments
        ]
        payload = {
            "source": path.name,
            "model": model_name,
            "duration_s": round(info.duration, 1),
            "language": info.language,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--folder", required=True, type=Path)
    parser.add_argument("--syllabus", required=True, type=Path)
    parser.add_argument("--scope", help="Restrict candidates to this node id and its children")
    parser.add_argument("--model", default="small", help="faster-whisper model (default: small)")
    parser.add_argument(
        "--cache", type=Path, help="Transcript cache dir (default: <folder>/.transcripts)"
    )
    parser.add_argument("--teacher", default=None)
    parser.add_argument(
        "--level",
        choices=["lesson", "subtopic", "topic"],
        default="lesson",
        help="Granularity of the name. 'lesson' is what makes six files in one "
        "chapter get six different names (default: lesson)",
    )
    parser.add_argument(
        "--classifier",
        choices=["keyword", "llm"],
        default="keyword",
        help="'keyword' is retrieval only and needs no API key. 'llm' adds the "
        "Claude call that reads the transcript and picks among the retrieved "
        "candidates; needs ANTHROPIC_API_KEY (default: keyword)",
    )
    parser.add_argument(
        "--apply", action="store_true", help="Actually rename. Default is a dry run."
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    logging.getLogger("pipeline.syllabus.taxonomy").setLevel(logging.ERROR)

    if not args.folder.is_dir():
        raise SystemExit(f"not a folder: {args.folder}")

    taxonomy = load_taxonomy(args.syllabus)
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

    # The name comes off whichever level the caller asked for, so the same
    # taxonomy serves "one file per lesson" and "one file per subtopic".
    #
    # 'lesson' means leaves, not "depth <= 2". A broad lecture otherwise wins
    # on its *parent* — and the parent of a lesson is the subtopic, whose title
    # is usually the folder name. "Aug, 6 - Respiration.mp4" sitting inside
    # "chap 12-Respiration" says nothing that the folder did not already say.
    # Forcing the choice down to a lesson makes the classifier commit; the
    # coverage figure and the (review) marker are what express its doubt.
    if args.level == "lesson":
        allowed = {n.id for n in taxonomy.all_nodes() if not n.children}
    elif args.level == "subtopic":
        allowed = {n.id for n in taxonomy.all_nodes() if n.depth == 1 or not n.children}
    else:
        allowed = {n.id for n in taxonomy.all_nodes() if n.depth == 0}

    classify_settings = ClassifySettings()
    retriever = HybridRetriever(taxonomy, classify_settings, RetrievalSettings())
    classifier = (
        LLMClassifier(taxonomy, classify_settings)
        if args.classifier == "llm"
        else KeywordClassifier(taxonomy)
    )
    naming = NamingSettings()
    cache_dir = args.cache or (args.folder / ".transcripts")

    files = [p for p in args.folder.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES]
    if not files:
        raise SystemExit(f"no media files in {args.folder}")
    # Chronological, not alphabetical. Collision handling numbers by iteration
    # order, so two lectures on one topic become "Part 1"/"Part 2" in the order
    # they were taught. Sorting by name would also put "Aug, 11" before
    # "Aug, 2" on any re-run over already-renamed files.
    files.sort(key=lambda p: (recorded_at_for(p)[0], p.name))

    log.info("%d file(s) in %s", len(files), args.folder)
    log.info("syllabus: %s  |  %d candidate nodes at level '%s'  |  classifier: %s\n",
             taxonomy.subject_name, len(allowed), args.level, classifier.name)

    model_holder: dict = {}
    taken: set[str] = set()
    plan: list[dict] = []

    for path in files:
        log.info("%s", path.name)
        recorded_at, date_source = recorded_at_for(path)
        transcript = transcribe(path, cache_dir, args.model, taxonomy, model_holder=model_holder)

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

        candidates = retriever.retrieve(transcript)
        # Drop candidates deeper than the requested naming level *before*
        # classifying, so confidence reflects the choice actually being made.
        candidates.candidates = [c for c in candidates.candidates if c.node.id in allowed]
        classification = classifier.classify(recording, transcript, candidates)

        low = (
            classification.abstain
            or classification.primary is None
            or classification.primary.confidence < classify_settings.review_threshold
        )
        name_plan = build_name(
            recording, classification, taxonomy, naming,
            existing_names=taken, is_low_confidence=low,
        )
        taken.add(name_plan.filename)

        top = candidates.top(3)
        conf = classification.primary.confidence if classification.primary else 0.0
        log.info("  date      %s  (from %s)", recorded_at.strftime("%b %-d, %Y")
                 if sys.platform != "win32" else recorded_at.strftime("%b %#d, %Y"), date_source)
        log.info("  duration  %.0f min, %d segments", transcript.duration_s / 60,
                 len(transcript.segments))
        log.info("  ranking   %s", "  |  ".join(
            f"{c.node.title} {c.coverage:.0%}" for c in top) or "nothing matched")
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
                "recorded_at": recorded_at.isoformat(),
                "date_source": date_source,
                "ranking": [
                    {"node": c.node.title, "id": c.node.id, "coverage": round(c.coverage, 3)}
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
        log.info(
            "%d of %d below the review threshold - marked '%s' in the new name",
            len(flagged), len(plan), naming.low_confidence_marker.strip(),
        )

    if not args.apply:
        log.info("\nDry run. Nothing renamed. Re-run with --apply to perform these renames.")
        return 0

    # Rename via a temporary name first: a plan can legitimately swap two
    # names, and a direct rename would clobber one of them.
    undo: list[dict] = []
    for row in plan:
        src = args.folder / row["from"]
        if row["from"] == row["to"]:
            continue
        staging = args.folder / f".renaming-{row['from']}"
        src.rename(staging)
        undo.append({"staging": staging.name, "to": row["to"], "from": row["from"]})
    for step in undo:
        (args.folder / step["staging"]).rename(args.folder / step["to"])

    undo_path = args.folder / "rename_undo.json"
    undo_path.write_text(
        json.dumps([{"from": s["to"], "to": s["from"]} for s in undo], indent=1),
        encoding="utf-8",
    )
    log.info("renamed %d file(s). Undo plan: %s", len(undo), undo_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
