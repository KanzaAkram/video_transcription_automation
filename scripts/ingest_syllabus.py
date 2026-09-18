"""Turn an official Cambridge syllabus PDF into a taxonomy YAML.

    python scripts/ingest_syllabus.py \
        --pdf 9700-2025-2027-syllabus.pdf \
        --overlay config/syllabi/overlays/a2-biology-9700.yaml \
        --out config/syllabi/a2-biology-9700.yaml

Why this exists
---------------
The learning outcomes are the highest-value text in the whole pipeline: they
are what retrieval scores against and what the classifier reads. Typing them
by hand is both tedious and the single easiest place to introduce an error
that then silently misfiles lectures for a year. So they are extracted from
the PDF verbatim and marked ``verified: true``; everything the PDF does *not*
contain comes from a hand-written **overlay** and is marked accordingly.

The split matters:

======================  ==========================================
From the PDF            topic/subtopic numbers, titles, outcomes
From the overlay        keywords, aliases, Drive folder names,
                        teaching order, lesson-level children
======================  ==========================================

A syllabus subtopic is not a lecture. "12.2 Respiration" is fourteen learning
outcomes and, in practice, six or seven lessons. Filenames need the finer
granularity, so the overlay adds a lesson layer *underneath* the syllabus and
each lesson node cites the outcome numbers it covers. Nothing is invented at
the syllabus level.

Parsing notes
-------------
Cambridge lays subject content out in two columns, which flattens to: the
topic or subtopic code, its title wrapped over one or more lines, the string
"Learning outcomes", then outcome numbers alone on their own lines followed by
the outcome text wrapped over several lines. Page furniture (running header,
URL, "Back to contents page", bare page numbers) is interleaved throughout.

The one genuinely ambiguous case is a bare number on its own line: "12" is a
topic heading, an outcome number and a page number, all in the same document.
Everything in the PDF is set at 10pt, so size cannot separate them — but
headings are set in the bold cut of the face and body text in the light cut.
So the parser reads spans rather than plain text and carries a bold flag,
which is what makes a bold "12" a topic and a light "12" an outcome.

A heading that contains a Greek letter is still a heading. Cambridge substitutes
Δ from TimesNewRomanPS-BoldMT into an otherwise HelveticaNeue …-Bd line, and
that font spells bold as "Bold", not "Bd". Requiring the substring "Bd" on
every span treated "23.3 Entropy change, ΔS" as body text, so it was appended
to the previous outcome instead of becoming a subtopic.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - import guard
    print(
        "PyMuPDF is required for syllabus ingestion:  pip install pymupdf",
        file=sys.stderr,
    )
    raise SystemExit(1) from None

import yaml

# ── Section markers ───────────────────────────────────────────────────────────
# Cambridge splits subject content into an AS half and an A Level half. Which
# half a topic sits in decides whether it belongs in an AS or an A2 taxonomy,
# and it is not derivable from the topic number alone across subjects.
SECTION_MARKERS = {
    "AS Level subject content": "as",
    "A Level subject content": "a-level",
}

#: What the section is called in a syllabus that has no AS/A Level split.
#:
#: IGCSE and O Level syllabuses carry neither marker anywhere, because there is
#: only one tier of subject content. The section machinery exists to separate
#: two halves that Cambridge interleaves in one PDF; where there are not two
#: halves, "no marker found" means "it is all one section", not "there is no
#: content" — which is what the parser used to conclude, returning zero topics
#: from a perfectly well-formed 0620 syllabus.
SINGLE_SECTION = "single"

#: Lines that introduce a block of learning outcomes.
#:
#: AS/A Level writes "Learning outcomes" once per subtopic. IGCSE writes the
#: tier name instead - "Core", then "Supplement" partway down the same list,
#: with the numbering running straight through both. Matched exactly rather
#: than by prefix: "Core" as a prefix would swallow any outcome beginning with
#: the word, and several do.
_TIER_HEADINGS = ("Core", "Supplement", "Core and Supplement")

# Page furniture to drop.
_FURNITURE = (
    "Cambridge International",
    "www.",
    "Back to contents page",
    "Learning outcomes",
    "Candidates should be able",
)

_TOPIC = re.compile(r"^(\d{1,2})(?: ([A-Z][A-Za-z ,\-']+))?$")
_SUBTOPIC = re.compile(r"^(\d{1,2}\.\d) ?(.*)$")
_BARE_NUMBER = re.compile(r"^\d{1,3}$")
_NUMBERED = re.compile(r"^(\d{1,2}) (.+)$")
#: Everything after this heading is assessment information, not subject content.
_END_OF_CONTENT = re.compile(r"\bDetails of the assessment\b")


def _font_is_bold(font: str) -> bool:
    """Whether this span is a bold cut of the face.

    Headings are HelveticaNeue …-Bd except where a Greek letter is taken from
    TimesNewRomanPS-BoldMT. Both are bold; only the first contains the letters
    "Bd". See the module docstring.
    """
    lowered = font.casefold()
    return "bd" in lowered or "bold" in lowered


def _clean(line: str) -> str:
    """Normalise the PDF's mixed tab/en-space/nbsp separators to single spaces."""
    line = (
        line.replace("’", "'")
        .replace("‘", "'")
        .replace("–", "-")
        .replace("‑", "-")
    )
    return re.sub(r"\s+", " ", line).strip()


def _has_outcomes(topics: dict[str, dict[str, Any]]) -> bool:
    return any(sub["outcomes"] for t in topics.values() for sub in t["subs"].values())


def _page_lines(page: Any) -> list[tuple[str, bool]]:
    """Flatten a page to ``(text, is_bold)`` pairs, one per rendered line."""
    out: list[tuple[str, bool]] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = _clean("".join(s["text"] for s in spans))
            if text:
                out.append((text, all(_font_is_bold(s["font"]) for s in spans)))
    return out


def _plain_lines(page: Any) -> list[str]:
    """Just the text of a page's lines, for the section pre-scan."""
    return [text for text, _ in _page_lines(page)]


def parse_pdf(pdf_path: Path) -> dict[str, dict[str, Any]]:
    """Extract ``{topic_number: {title, section, subs: {code: {title, outcomes}}}}``."""
    doc = fitz.open(pdf_path)
    topics: dict[str, dict[str, Any]] = {}

    # Decided before parsing, because it is a property of the document rather
    # than of any one page: a syllabus carrying neither section marker anywhere
    # is single-tier, and every page of it belongs to the one section. Scanning
    # up front rather than falling back mid-parse keeps the two cases from
    # interleaving - a split syllabus must never be silently treated as single
    # just because its first content page came before its first marker.
    split = any(
        marker in _plain_lines(page) for page in doc for marker in SECTION_MARKERS
    )
    doc = fitz.open(pdf_path)  # the generator above consumed the page cursor
    section: str | None = None if split else SINGLE_SECTION
    current_topic: str | None = None
    current_sub: dict[str, Any] | None = None
    mode: str | None = None  # None | "topic_title" | "title" | "await" | "outcome"

    for page in doc:
        lines = _page_lines(page)
        plain = [text for text, _ in lines]

        # The contents page lists *both* section headings. A real content page
        # carries exactly one, as its running heading — so a page with two is
        # the table of contents and must not set the section or be parsed.
        present = [name for marker, name in SECTION_MARKERS.items() if marker in plain]
        if len(present) > 1:
            continue
        if present:
            section = present[0]

        for line, bold in lines:
            if not line or line in ("continued",) or line in SECTION_MARKERS:
                continue
            # IGCSE's equivalent of "Learning outcomes". A subtopic carries a
            # Core block and often a Supplement block, and both belong to it.
            if line in _TIER_HEADINGS:
                mode = "await"
                continue
            if line.startswith(_FURNITURE):
                if line.startswith(("Learning outcomes", "Candidates should be able")):
                    mode = "await"
                continue
            # "Details of the assessment" also appears in the contents page
            # listing, long before any subject content — so it only terminates
            # the parse once real outcomes have actually been collected.
            if _END_OF_CONTENT.search(line) and _has_outcomes(topics):
                return topics
            if section is None:
                continue

            # A topic heading is bold. That is the only thing separating a
            # topic number from an outcome number set in the light cut.
            topic_match = _TOPIC.fullmatch(line) if bold else None
            if topic_match:
                number, title = topic_match.groups()
                entry = topics.setdefault(number, {"title": "", "section": section, "subs": {}})
                entry["title"] = (title or "").strip()
                entry["section"] = section
                current_topic = number
                current_sub = None
                # Single-digit topics set the number and the title on separate
                # lines; two-digit ones keep them together. Both occur.
                mode = None if title else "topic_title"
                continue

            sub_match = _SUBTOPIC.match(line) if bold else None
            # A subtopic code only counts under the topic heading it belongs to.
            # Without this, any decimal in prose ("38.5 °C", "a ratio of 1.5")
            # is read as a subtopic and invents a topic to hang it from.
            if sub_match and current_topic and sub_match.group(1).split(".")[0] == current_topic:
                code, remainder = sub_match.group(1), (sub_match.group(2) or "")
                remainder = re.sub(r"\bcontinued\b", "", remainder).strip()
                parent = code.split(".")[0]
                entry = topics.setdefault(parent, {"title": "", "section": section, "subs": {}})
                is_new = code not in entry["subs"]
                if is_new:
                    entry["subs"][code] = {"title": remainder, "outcomes": []}
                current_sub = entry["subs"][code]
                # A repeat of the code is a "…continued" page: the title is
                # already known, so go straight back to collecting outcomes.
                mode = "title" if is_new else "await"
                continue

            # Checked after the subtopic branch: the first subtopic heading is
            # bold too, and would otherwise be swallowed into the topic title.
            if mode == "topic_title" and bold and current_topic:
                entry = topics[current_topic]
                entry["title"] = f"{entry['title']} {line}".strip()
                continue

            if current_sub is None:
                continue

            expected = len(current_sub["outcomes"]) + 1

            # An outcome number alone on a line. Any *other* bare number is a
            # page number — the sequence check is what tells them apart.
            if _BARE_NUMBER.fullmatch(line):
                if mode in ("await", "outcome") and int(line) == expected:
                    current_sub["outcomes"].append("")
                    mode = "outcome"
                continue

            numbered = _NUMBERED.match(line)
            if mode in ("await", "outcome") and numbered and int(numbered.group(1)) == expected:
                current_sub["outcomes"].append(numbered.group(2).strip())
                mode = "outcome"
                continue

            if mode == "title":
                current_sub["title"] = f"{current_sub['title']} {line}".strip()
                continue

            if mode == "outcome" and current_sub["outcomes"]:
                text = f"- {line.lstrip('• ').strip()}" if line.startswith("•") else line
                current_sub["outcomes"][-1] = f"{current_sub['outcomes'][-1]} {text}".strip()

    return topics


# ── Overlay merge ─────────────────────────────────────────────────────────────


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def build_taxonomy(
    topics: dict[str, dict[str, Any]],
    overlay: dict[str, Any],
    *,
    section: str,
) -> dict[str, Any]:
    """Merge parsed PDF content with the hand-written overlay."""
    prefix = overlay["node_id_prefix"]
    topic_overlays: dict[str, Any] = overlay.get("topics", {})
    nodes: list[dict[str, Any]] = []

    # A topic with no learning outcomes anywhere under it is not subject
    # content. On a single-tier syllabus the contents page is parsed like any
    # other - there is no second section marker to identify it by - so it
    # contributes headings such as "Details of the assessment" and "What else
    # you need to know", which look exactly like topics and carry no outcomes.
    # Requiring outcomes drops them without needing to recognise them by name.
    wanted = {
        number: topic
        for number, topic in topics.items()
        if topic["section"] == section
        and any(sub["outcomes"] for sub in topic["subs"].values())
    }
    unknown = set(topic_overlays) - set(wanted)
    if unknown:
        raise SystemExit(
            f"overlay names topics not present in the {section} section of the PDF: "
            f"{sorted(unknown)}"
        )

    # Chapters the syllabus does not contain, because they are not subject
    # content: the course-orientation session, a midterm review week, a
    # practical paper. They are real folders holding real recordings, and a
    # taxonomy that cannot name them sends every such lecture to review.
    #
    # Marked `exclude_from_classification`, so they are shown to the model as
    # structure but never offered as an answer for a lesson - an intro session
    # name-checks half the syllabus in forty minutes and would otherwise
    # outscore the lecture that actually taught it. `classify.openai_classifier`
    # reaches the Introduction one only after the model has said the recording
    # *is* an orientation session.
    for extra in overlay.get("extra_chapters", []):
        nodes.append(
            {
                "id": f"{prefix}.{extra['id']}",
                "title": extra["title"],
                "syllabus_topic": None,
                "teaching_order": extra.get("teaching_order"),
                "coursebook_chapter": None,
                "drive_folder_name": extra.get("drive_folder_name"),
                "verified": True,
                "exclude_from_classification": True,
                "aliases": extra.get("aliases", []),
                "keywords": extra.get("keywords", []),
                "children": [],
            }
        )

    for number, topic in sorted(wanted.items(), key=lambda kv: int(kv[0])):
        over = topic_overlays.get(number, {})
        topic_node: dict[str, Any] = {
            "id": f"{prefix}.topic_{number}",
            "title": topic["title"],
            "syllabus_topic": number,
            "teaching_order": over.get("teaching_order"),
            "coursebook_chapter": over.get("coursebook_chapter"),
            "drive_folder_name": over.get("drive_folder_name"),
            "verified": True,
            "aliases": over.get("aliases", []),
            "keywords": over.get("keywords", []),
            "children": [],
        }

        sub_overlays: dict[str, Any] = over.get("subtopics", {})
        unknown_subs = set(sub_overlays) - set(topic["subs"])
        if unknown_subs:
            raise SystemExit(f"overlay names unknown subtopics: {sorted(unknown_subs)}")

        for code, sub in sorted(topic["subs"].items()):
            sub_over = sub_overlays.get(code, {})
            outcomes = [
                {"code": f"{code}.{index}", "text": text, "verified": True}
                for index, text in enumerate(sub["outcomes"], start=1)
            ]
            sub_node: dict[str, Any] = {
                "id": f"{prefix}.{code.replace('.', '_')}",
                "title": sub["title"],
                # Keywords-only form for the filename, from the overlay.
                #
                # Biology never needed this: its lesson layer supplies the
                # filename and lesson titles are already written that way. A
                # syllabus with no lesson layer has nothing between the
                # subtopic and the file, so without a short title the name
                # falls back to `title` — which Cambridge writes as a whole
                # sentence. "Similarities and trends in the properties of the
                # Group 2 metals, magnesium to barium, and their compounds" is
                # not a filename. `ingest_syllabus_json.py` has always read
                # this field; the PDF path simply never did.
                "short_title": sub_over.get("short_title"),
                "syllabus_topic": code,
                "drive_folder_name": sub_over.get("drive_folder_name"),
                "verified": True,
                "aliases": sub_over.get("aliases", []),
                "keywords": sub_over.get("keywords", []),
                "learning_outcomes": outcomes,
                "children": [],
            }

            # Lesson layer. Not in the syllabus — this is teaching structure,
            # so it is marked unverified and each lesson cites the outcomes it
            # covers, which is where its retrieval text comes from.
            by_number = {index: text for index, text in enumerate(sub["outcomes"], start=1)}
            for lesson in sub_over.get("lessons", []):
                cited = lesson.get("outcomes", [])
                missing = [n for n in cited if n not in by_number]
                if missing:
                    raise SystemExit(
                        f"lesson {lesson['title']!r} cites outcomes {missing} "
                        f"which do not exist in {code} (it has {len(by_number)})"
                    )
                sub_node["children"].append(
                    {
                        "id": f"{prefix}.{code.replace('.', '_')}.{_slug(lesson['title'])}",
                        "title": lesson["title"],
                        "verified": False,
                        "aliases": lesson.get("aliases", []),
                        "keywords": lesson.get("keywords", []),
                        "learning_outcomes": [
                            {"code": f"{code}.{n}", "text": by_number[n], "verified": True}
                            for n in cited
                        ],
                    }
                )

            topic_node["children"].append(sub_node)
        nodes.append(topic_node)

    return {
        "version": 1,
        "subject_id": overlay["subject_id"],
        "subject_name": overlay["subject_name"],
        "syllabus_code": overlay["syllabus_code"],
        "syllabus_years": overlay["syllabus_years"],
        "awarding_body": "Cambridge International",
        "source": overlay.get("source", "Extracted from the official syllabus PDF"),
        "subject_vocabulary": overlay.get("subject_vocabulary", []),
        "nodes": nodes,
    }


def _prune(value: Any) -> Any:
    """Drop empty/None fields so the emitted YAML stays readable."""
    if isinstance(value, dict):
        return {k: _prune(v) for k, v in value.items() if v not in (None, [], {}, "")}
    if isinstance(value, list):
        return [_prune(v) for v in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--overlay", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--section",
        default="a-level",
        choices=sorted({*SECTION_MARKERS.values(), SINGLE_SECTION}),
        help="Which half of the syllabus to emit (default: a-level). Pass "
        f"'{SINGLE_SECTION}' for an IGCSE or O Level syllabus, which has no "
        "AS/A Level split and so has only one section to emit",
    )
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    topics = parse_pdf(args.pdf)
    if not topics:
        print(f"no subject content found in {args.pdf}", file=sys.stderr)
        return 1

    overlay = yaml.safe_load(args.overlay.read_text(encoding="utf-8")) or {}
    taxonomy = _prune(build_taxonomy(topics, overlay, section=args.section))

    text = yaml.safe_dump(taxonomy, sort_keys=False, allow_unicode=True, width=100)
    header = (
        f"# {taxonomy['subject_name']} {taxonomy['syllabus_code']} "
        f"({taxonomy['syllabus_years']})\n"
        f"#\n"
        # The real command, with the real paths. A header reading
        # "--pdf <syllabus.pdf>" says the file was generated without saying
        # what from, which is the one thing the next person needs.
        f"# GENERATED — do not edit by hand. Regenerate with:\n"
        f"#   python scripts/ingest_syllabus.py \\\n"
        f"#       --pdf {args.pdf.as_posix()} \\\n"
        f"#       --overlay {args.overlay.as_posix()} \\\n"
        f"#       --out {args.out.as_posix()} --section {args.section}\n"
        f"#\n"
        f"# Titles and learning outcomes are verbatim from the official PDF and are\n"
        f"# marked verified. Keywords, aliases, Drive folder names and the lesson\n"
        f"# layer come from the overlay: edit those there, not here.\n\n"
    )

    if args.print_only:
        print(header + text)
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(header + text, encoding="utf-8")

    subtopics = sum(len(n.get("children", [])) for n in taxonomy["nodes"])
    lessons = sum(
        len(sub.get("children", []))
        for node in taxonomy["nodes"]
        for sub in node.get("children", [])
    )
    outcomes = sum(
        len(sub.get("learning_outcomes", []))
        for node in taxonomy["nodes"]
        for sub in node.get("children", [])
    )
    print(
        f"wrote {args.out}\n"
        f"  {len(taxonomy['nodes'])} topics, {subtopics} subtopics, "
        f"{lessons} lessons, {outcomes} learning outcomes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
