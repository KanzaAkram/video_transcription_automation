"""Filename construction.

The default template reproduces the convention already in your Drive:

    "Aug, 5 - Atomic Structure.mp4"
    "Sep, 23 - Solubility in Solvents + Induced Dipoles.mp4"

That is deliberate. A pipeline that renames everything into some new scheme
would leave the existing archive inconsistent, so the default is your format
and the template is configuration rather than code.

The topic is one subtopic, or up to three joined by `subtopic_separator`. The
second example above is a typical two-subtopic name, not the upper bound: most
recordings teach one subtopic and are named for it, and a name reaches three
only when the lecture genuinely worked through three.

Only the topic may be shortened. The date, the "(review)" marker, the part
suffix and the extension each carry something a reader cannot recover from the
rest of the name, so the length cap is spent on the topic and never on them:
the budget for the topic is worked out *before* the name is rendered, rather
than by chopping the finished string. Chopping the finished string is what made
a low-confidence recording render byte-identical to a confident one, and what
made the collision search never terminate — " - Part 2" and " - Part 3"
truncated back to the same string, so no part number was ever free.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from pipeline.config import NamingSettings
from pipeline.models import Classification, NamePlan, Recording
from pipeline.syllabus import Taxonomy

MONTH_ABBR = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

_WHITESPACE = re.compile(r"\s+")

#: Joiners and punctuation left stranded at the end of a shortened label. "&"
#: and "+" are in the class because "Solubility in Solvents &.mp4" reads as a
#: name with a word missing rather than as a name that was shortened.
_DANGLING_TAIL = re.compile(r"[\s\-&+,;:]+$")

#: Separators a free-text label might have been written with, tried only after
#: the configured one. Not "," and not " and ": both occur inside single
#: syllabus titles ("Intermolecular forces, electronegativity and bond
#: properties"), so splitting on them would invent subtopics rather than find
#: them.
_FALLBACK_SEPARATORS = (" & ", " + ")

#: A character no sanitised label can contain, used to measure how much of
#: `max_length` the protected parts of a name consume. Deliberately not a
#: space: the marker step rstrips the stem, so a whitespace probe would measure
#: one character short and let the finished name run over the cap.
_PROBE = "\x00"

#: How far the collision search may walk before giving up. Fifty recordings of
#: one topic on one day is far past implausible, so a search that gets this far
#: is looping rather than counting.
_MAX_COLLISION_ATTEMPTS = 50


def sanitise(text: str, strip_characters: str) -> str:
    """Remove characters that break Drive, Windows sync clients and shells.

    Also collapses whitespace and trims trailing dots — Windows silently drops
    a trailing dot from a filename, which turns a rename into a surprise.
    """
    cleaned = "".join(ch for ch in text if ch not in strip_characters)
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()
    return cleaned.rstrip(". ")


def _fit(text: str, budget: int) -> str:
    """Shorten `text` to `budget` characters without stranding a joiner.

    A plain slice can stop at "Solubility in Solvents &", which reads as a
    filename with a word missing rather than as a shortened one.
    """
    if budget <= 0:
        return ""
    if len(text) <= budget:
        return text
    return _DANGLING_TAIL.sub("", text[:budget])


def _fit_label(label: str, budget: int, settings: NamingSettings) -> str:
    """Fit a joined subtopic label into `budget`, dropping whole names first.

    A plain `_fit` on "A + B + C" stops mid-word and leaves "A + B + Muscle
    Structure and the Neuro", which reads as a third subtopic whose name is
    wrong rather than as a name with a third subtopic left off. Two subtopics
    named correctly is a true, shorter statement about the lecture; two and a
    fragment is not.

    **A label that already fits is returned untouched.** That is not an
    optimisation, it is the whole correctness condition. Rebuilding it from its
    parts re-joins with `subtopic_separator`, and 83 of the shipped syllabi's
    labels contain a " & " of their own — "Chlorine & Disproportionation",
    "Estimation & Rounding". Splitting those on the fallback separator and
    re-joining rewrote one subtopic's true title into "Estimation + Rounding",
    which reads as two subtopics that the classifier never chose. Caught on the
    19 August maths lecture, where the file came out named for a pair of topics
    that do not exist.

    When trimming *is* needed, the parts are re-joined with the separator they
    were actually split on, for the same reason.
    """
    if len(label) <= budget:
        return label
    separator, parts = _split_label(label, settings)
    while len(parts) > 1:
        parts.pop()
        joined = separator.join(parts)
        if len(joined) <= budget:
            return joined
    return _fit(parts[0] if parts else label, budget)


def _split_label(label: str, settings: NamingSettings) -> tuple[str, list[str]]:
    """Split a joined label, reporting which separator did it.

    Conservative on purpose. Half the syllabus short titles contain an "&" of
    their own ("Chlorine & Disproportionation"), so the configured separator is
    tried first and the alternatives only if it found nothing: a label this
    module joined is split exactly where it was joined, and a single title
    carrying an "&" is left intact.

    The separator comes back with the parts because any caller that rebuilds
    the label has to put back what it took out — see `_fit_label`.
    """
    for separator in (settings.subtopic_separator, *_FALLBACK_SEPARATORS):
        if not separator or separator not in label:
            continue
        parts = [part.strip() for part in label.split(separator)]
        parts = [part for part in parts if part]
        if len(parts) > 1:
            return separator, parts
    stripped = label.strip()
    return settings.subtopic_separator, ([stripped] if stripped else [])


def _split_subtopics(label: str, settings: NamingSettings) -> list[str]:
    """The parts of a joined label, for counting them against the cap."""
    return _split_label(label, settings)[1]


def _topic_label(
    classification: Classification | None,
    taxonomy: Taxonomy | None,
    settings: NamingSettings,
) -> str:
    """What goes in the filename, governed whatever its source.

    Five sources in falling order of authority:

      1. `focus` — the learning-outcome keywords, joined by
         `subtopic_separator`. The most specific true name available, and what
         the ChatGPT path produces. Every word of it is the syllabus'.
      2. `topic_node_ids` — the one to three subtopics the classifier
         committed to, rendered through each node's `label` so the filename
         gets the keyword form ("Shapes of Molecules") rather than the
         syllabus sentence. The fallback when no keyword survived grounding.
      3. `suggested_topic_label` — a free-text label, from the Claude path.
      4. the primary node's own label.
      5. "Unclassified".

    The order matters for compatibility as much as for quality: callers that
    set none of the new fields — the older pipeline stages, hand-built
    fixtures — fall straight through to the behaviour they had before.

    The subtopic cap and the length rule are applied to the *result*, not to
    source 1 alone. Source 2 is a string somebody else chose: an abstaining
    model reply still carries one, because the reply schema makes every field
    mandatory, and the unknown-node-id path clears the ids without clearing the
    label. Guarding only the call sites that exist today would leave the next
    one free to name a fourth subtopic, so the guard sits at the single point
    every caller passes through.
    """
    label = _raw_topic_label(classification, taxonomy, settings)
    parts = _split_subtopics(label, settings)
    if len(parts) > settings.max_subtopics:
        label = settings.subtopic_separator.join(parts[: settings.max_subtopics])
    # One field of a filename can never exceed the whole filename; build_name
    # trims further, to what the protected parts actually leave.
    return _fit_label(label, settings.max_length, settings)


def _raw_topic_label(
    classification: Classification | None,
    taxonomy: Taxonomy | None,
    settings: NamingSettings,
) -> str:
    """The best label available, ungoverned. See `_topic_label`."""
    if classification is None:
        return "Unclassified"
    # The learning-outcome keywords, where the classifier produced any. They
    # are the most specific true thing available about the lecture: a syllabus
    # subtopic heading is the folder-level answer and names three lessons
    # identically, where "Behaviour in an Electric Field" names one of them.
    #
    # Every word of them came from the syllabus text of the outcomes the
    # lecture cited - see `classify.openai_classifier.grounded_keywords` - so
    # this is still the syllabus naming the file, just one level below the
    # heading.
    if classification.focus:
        return classification.focus
    if classification.topic_node_ids and taxonomy is not None:
        # Capped here as well as at selection: this is the last point before
        # the name is written, and it is the one every caller passes through.
        labels = [
            node.label
            for node in (
                taxonomy.get(node_id)
                for node_id in classification.topic_node_ids[: settings.max_subtopics]
            )
            if node is not None
        ]
        if labels:
            return settings.subtopic_separator.join(labels)
    if classification.suggested_topic_label:
        return classification.suggested_topic_label
    if classification.primary and taxonomy is not None:
        node = taxonomy.get(classification.primary.node_id)
        if node:
            return node.label
    return "Unclassified"


def _chapter_label(classification: Classification | None, taxonomy: Taxonomy | None) -> str:
    """The one chapter this recording belongs to.

    Prefers the decision the classifier actually made over re-deriving it from
    the primary subtopic: the two agree today, but the chapter is the placement
    decision and the subtopic is the naming one, and a filename should show the
    chapter the file was filed into rather than a second opinion about it.
    """
    if not (classification and taxonomy):
        return ""
    if classification.chapter_node_id:
        node = taxonomy.get(classification.chapter_node_id)
        if node is not None:
            return node.title
    if not classification.primary:
        return ""
    node = taxonomy.get(classification.primary.node_id)
    if node is None:
        return ""
    # Walk up to the chapter-level ancestor if this is a subtopic.
    ancestors = taxonomy.ancestors(node.id)
    chapter = ancestors[0] if ancestors else node
    return chapter.title


def _first_free_suffix(
    render: Callable[[str], str],
    suffix_for: Callable[[int], str],
    existing_names: set[str],
) -> tuple[int, str]:
    """Walk n = 2, 3, ... to the first name nobody has taken.

    Bounded, and it refuses a suffix that renders to a name it has already
    produced. Both guards exist because the unbounded version hung: the suffix
    used to be appended and then truncated away again, so every n rendered to
    the same string and the search spun forever in the middle of a folder-wide
    rename. A rename tool that raises is recoverable; one wedged at 100% CPU on
    a server is not, and it does not look broken either.
    """
    seen: set[str] = set()
    for number in range(2, _MAX_COLLISION_ATTEMPTS + 2):
        candidate = render(suffix_for(number))
        if candidate in seen:
            raise RuntimeError(
                f"collision suffix {suffix_for(number)!r} does not change the rendered "
                f"filename {candidate!r}, so no suffix can make it unique. Check "
                "naming.template and naming.max_length."
            )
        if candidate not in existing_names:
            return number, candidate
        seen.add(candidate)
    raise RuntimeError(
        f"gave up after {_MAX_COLLISION_ATTEMPTS} attempts to find a free filename "
        f"near {render(suffix_for(2))!r}"
    )


def build_name(
    recording: Recording,
    classification: Classification | None,
    taxonomy: Taxonomy | None,
    settings: NamingSettings,
    *,
    existing_names: set[str] | None = None,
    is_low_confidence: bool = False,
) -> NamePlan:
    """Render the filename for a classified recording.

    Collision handling is by *meaning*, not by counter: same date plus same
    topic almost always means one lecture split across two recordings, so the
    result is "- Part 1" / "- Part 2" rather than "(1)" / "(2)". That reads
    correctly to a student browsing the folder.

    The length cap is spent on the topic alone. Everything else in the name is
    protected, so a name that has to lose characters loses them from the one
    field a reader can still guess at — never the date, the review marker or
    the part number.
    """
    existing_names = existing_names or set()
    extension = Path(recording.original_filename).suffix or ".mp4"
    when = recording.recorded_at

    topic = sanitise(_topic_label(classification, taxonomy, settings), settings.strip_characters)
    chapter = sanitise(_chapter_label(classification, taxonomy), settings.strip_characters)
    teacher = sanitise(recording.teacher or "", settings.strip_characters)

    fields: dict[str, str] = {
        "date": when.strftime("%Y-%m-%d"),
        "year": str(when.year),
        "month_abbr": MONTH_ABBR[when.month - 1],
        "month": f"{when.month:02d}",
        "day": str(when.day),
        "day_padded": f"{when.day:02d}",
        "subject": sanitise(recording.subject_id, settings.strip_characters),
        "chapter": chapter,
        "subtopic": topic,
        "topic": topic,
        "teacher": teacher,
        "ext": extension,
        "part_suffix": "",
        "confidence_marker": settings.low_confidence_marker if is_low_confidence else "",
    }

    def render(topic_text: str, part_suffix: str) -> str:
        merged = {
            **fields,
            "topic": topic_text,
            "subtopic": topic_text,
            "part_suffix": part_suffix,
        }
        name = settings.template.format(**merged)
        # The marker goes before the extension so sorting stays sane.
        if is_low_confidence and settings.low_confidence_marker not in name:
            if name.endswith(extension):
                stem = name[: -len(extension)]
                name = f"{stem.rstrip()}{settings.low_confidence_marker}{extension}"
            else:
                name = f"{name.rstrip()}{settings.low_confidence_marker}"
        return name

    def render_fitted(part_suffix: str) -> str:
        """Render with the topic trimmed to whatever the protected parts leave.

        The budget is measured rather than derived, because the template is
        configuration: rendering once with a one-character probe prices the
        date, the separators, the marker, this part suffix and the extension
        together, in whatever order the template puts them.
        """
        overhead = len(render(_PROBE, part_suffix)) - len(_PROBE)
        fitted = _fit_label(topic, settings.max_length - overhead, settings)
        return _enforce_length(render(fitted, part_suffix), extension, settings.max_length)

    candidate = render_fitted("")
    part_number: int | None = None

    if candidate in existing_names:
        if settings.collision_strategy == "part_suffix":
            # The first of the pair is retroactively "Part 1" only if it is
            # already on disk under the bare name; we never rename other
            # people's files, so this one becomes Part 2 onward.
            part_number, candidate = _first_free_suffix(
                render_fitted, lambda n: f" - Part {n}", existing_names
            )
        elif settings.collision_strategy == "counter":
            _, candidate = _first_free_suffix(render_fitted, lambda n: f" ({n})", existing_names)
        else:  # timestamp
            candidate = render_fitted(f" [{when.strftime('%H%M')}]")

    return NamePlan(
        filename=candidate,
        original_filename=recording.original_filename,
        template_used=settings.template,
        is_low_confidence=is_low_confidence,
        part_number=part_number,
    )


def _enforce_length(name: str, extension: str, max_length: int) -> str:
    """Last-resort clamp: truncate the stem, never the date.

    A no-op on the normal path, where `build_name` has already fitted the topic
    to the space the protected parts leave. It still fires for a custom
    template with no topic field to shrink — there something has to give, and
    the date is the part a human scans for, so the loss is taken at the end.
    """
    if len(name) <= max_length:
        return name
    stem = name[: -len(extension)] if name.endswith(extension) else name
    return _fit(stem, max_length - len(extension)) + extension
