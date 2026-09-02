"""Turn a structured syllabus JSON export into a taxonomy YAML.

    python scripts/ingest_syllabus_json.py \
        --json 9701-2025-2027-syllabus.json \
        --overlay config/syllabi/overlays/as-chemistry-9701.yaml \
        --out config/syllabi/as-chemistry-9701.yaml

Why a second ingester
---------------------
``ingest_syllabus.py`` reads the official PDF, and nearly every fragile line in
it exists to undo two-column layout, page furniture, and the fact that a bold
"12" is a topic while a light "12" is an outcome. When a structured export of
the same document is already available, none of that guessing is worth
repeating. This script keeps the *contract* of the PDF ingester — syllabus text
verbatim and ``verified: true``, everything else from a hand-written overlay —
and replaces only the front end.

Why the tree is chapter -> subtopic
-----------------------------------
A **chapter is a Drive folder**, not a syllabus topic, and the two do not
correspond one-to-one. Cambridge 9701 topic 1 "Atomic structure" is split
across two folders: "1. Atomic Structure (Chapter 1)" holds 1.1 and 1.2, and
"2. Electrons in Atoms (Chapter 2)" holds 1.3 and 1.4. Emitting syllabus topics
as the top level would therefore describe a folder tree that does not exist.
So::

    depth 0   chapter    one Drive folder   — the unit a recording is filed into
    depth 1   subtopic   one syllabus code  — the unit a filename is built from

A recording belongs to exactly one chapter and is never split across two. That
rule only means anything if each subtopic code belongs to exactly one chapter,
which is why a code claimed twice is a hard error here and not a warning: the
alternative is a lecture that legitimately resolves to two destinations, and
nothing downstream can break that tie.

Why ``short_title``
-------------------
Filenames carry keywords, not syllabus prose — "Sep, 4 - Sigma and Pie
Bonds.mp4", never "Sep, 4 - Intermolecular forces, electronegativity and bond
properties.mp4". The verbatim title is retrieval text and stays on the node for
that; the overlay's ``short_title`` is what the naming layer puts on disk.

There is no lesson layer beneath the subtopic, unlike the A2 Biology overlay.
For these subjects a subtopic is about one lecture ("3.5 Shapes of molecules"),
where 9700's "12.2 Respiration" is fourteen outcomes and seven lessons.

Why the keywords are checked here
---------------------------------
Keywords are the only part of a taxonomy written freehand and the only part
whose mistakes are silent: a wrong title is obvious on sight, a wrong keyword
produces a confident, well-formed, wrong filename. Generation is the last
moment anyone is looking, so five checks run over every keyword and alias
before anything is written, and errors stop the write. ``--lint-only`` runs
them and writes nothing, which is the loop to use while editing an overlay.

Encoding
--------
These exports have been seen after a UTF-8/cp1252 round trip, which renders
"ΔH" as "Î”H" and "σ" as "Ïƒ". Each string is repaired independently, and
only when re-encoding both succeeds *and* reduces the number of suspicious
sequences — a blind round trip applied to clean text corrupts it instead.

Two limitations follow, and neither is fixable here. Text that already lost
characters to a lossy decode (a literal U+FFFD) carries no information to
recover and is left alone rather than mangled further. And a string round
tripped twice is only unwound one level. The count of repaired strings is
printed on every run so an operator can tell "the input was clean" apart from
"the repair pass silently did nothing".
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# ── Encoding repair ───────────────────────────────────────────────────────────


def _continuation_chars() -> str:
    """Characters a UTF-8 continuation byte (0x80-0xBF) is displayed as.

    Two spellings, because both decoders occur in the wild and which one
    mangled a given export is not knowable afterwards: latin-1 maps the range
    straight to U+0080-U+00BF, cp1252 maps most of it to punctuation (" ” ƒ …).
    Computed rather than typed out — the cp1252 half is a scattered table that
    is easy to get subtly wrong by hand, and a wrong character class here means
    real damage is silently left in place.
    """
    chars = {chr(byte) for byte in range(0x80, 0xC0)}
    chars |= {bytes([byte]).decode("cp1252", errors="replace") for byte in range(0x80, 0xC0)}
    chars.discard("�")
    return "".join(sorted(chars))


#: A UTF-8 lead byte (0xC2-0xEF) followed by a continuation byte, both read as
#: single characters — the signature of a round trip. U+FFFD is damage some
#: earlier decode already did.
_MOJIBAKE = re.compile(rf"[Â-ï][{re.escape(_continuation_chars())}]|�")


def _repair(text: str) -> str:
    """Undo one UTF-8 round trip, but only when it demonstrably helps.

    cp1252 is tried before latin-1 because cp1252 is what Windows actually
    applies, and it is the only one of the two that can put back the dashes and
    smart quotes living in 0x80-0x9F. latin-1 is the fallback for the five
    bytes cp1252 leaves undefined.
    """
    damage = len(_MOJIBAKE.findall(text))
    if not damage:
        return text
    for codec in ("cp1252", "latin-1"):
        try:
            fixed = text.encode(codec).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if len(_MOJIBAKE.findall(fixed)) < damage:
            return fixed
    return text


def repair_encoding(value: Any) -> tuple[Any, int]:
    """Walk a decoded JSON tree, repairing strings one at a time.

    Per string rather than over the whole document, so that one string the
    repair cannot handle does not veto the fix for all the others.
    """
    if isinstance(value, str):
        fixed = _repair(value)
        return fixed, int(fixed != value)
    if isinstance(value, list):
        repaired = [repair_encoding(item) for item in value]
        return [item for item, _ in repaired], sum(n for _, n in repaired)
    if isinstance(value, dict):
        pairs = {key: repair_encoding(item) for key, item in value.items()}
        return {key: item for key, (item, _) in pairs.items()}, sum(n for _, n in pairs.values())
    return value, 0


def load_syllabus(path: Path) -> tuple[dict[str, Any], int]:
    """Read the export as bytes, decode, then repair. Returns the repair count."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1252")
    return repair_encoding(json.loads(text))


# ── Syllabus content ──────────────────────────────────────────────────────────


#: How Cambridge marks the worked detail that belongs to the objective above
#: it, rather than being an objective of its own.
_NOTES_PREFIX = ("notes and examples", "note:", "notes:")


def _outcomes(code: str, objectives: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Learning outcomes for one subtopic, with the Notes rows folded in.

    0580 states a third of its entries as "Notes and examples: e.g. find the
    value of 7^-2 ..." - 90 of its 262. Those are not objectives. They are the
    worked detail for the objective immediately above, and Cambridge lays them
    out that way; the JSON simply numbers them alongside.

    Left as separate outcomes they do real damage, because the classifier is
    now asked to *cite* the outcomes a lecture taught. A subtopic that has
    three objectives and two Notes rows looks like five, and a citation of
    "1.7.5" names a row of examples rather than anything that was taught - so
    two lectures citing different Notes rows appear to differ when they do not.
    Merging restores the count and gives each objective its own examples, which
    is where the concrete spoken vocabulary lives ("rationalise the
    denominator", "three-figure bearing") and is worth keeping in the text the
    model reads.

    A Notes row with no objective before it - which no syllabus has produced,
    but the format permits - is kept as itself rather than dropped.
    """
    out: list[dict[str, Any]] = []
    for objective in objectives:
        text = str(objective["description"]).strip()
        if out and text.lower().startswith(_NOTES_PREFIX):
            out[-1]["text"] = f"{out[-1]['text']} {text}".strip()
            continue
        out.append(
            {
                "code": f"{code}.{objective['objective_number']}",
                "text": text,
                "verified": True,
            }
        )
    return out


def index_subtopics(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """``code -> {title, objectives}``, flattened across every topic.

    The topic layer is dropped deliberately. Chapters are the top level here,
    and a chapter's membership is defined by the codes its overlay entry lists,
    so a code only ever needs looking up by itself.
    """
    subtopics: dict[str, dict[str, Any]] = {}
    for topic in document["topics"]:
        for sub in topic["sub_topics"]:
            subtopics[str(sub["sub_topic_number"])] = {
                "title": sub["sub_topic_name"],
                "objectives": sub["learning_objectives"],
            }
    return subtopics


def _code_key(code: str) -> tuple[int, ...]:
    """Sort "1.10" after "1.2" rather than before it."""
    return tuple(int(part) if part.isdigit() else 0 for part in code.split("."))


def syllabus_revision(document: dict[str, Any]) -> str | None:
    """One line of provenance from ``syllabus_changes``.

    The two exports disagree on shape — a list of sentences in one, a list of
    ``{version, date, changes}`` objects in the other — so both are flattened
    to the same short string. Recorded in the generated header because
    regenerating against a newer revision of the same syllabus should be
    visible in the diff.
    """
    for entry in document.get("syllabus_changes") or []:
        if isinstance(entry, dict) and entry.get("version"):
            return f"version {entry['version']}, {entry.get('date', 'undated')}"
        if isinstance(entry, str) and entry.strip():
            return entry.strip()[:100]
    return None


# ── Overlay validation ────────────────────────────────────────────────────────


def check_overlay(
    overlay: dict[str, Any], subtopics: dict[str, dict[str, Any]]
) -> tuple[list[str], dict[str, str]]:
    """The three overlay mistakes that would misfile recordings silently.

    Returns the problems and the ``code -> chapter id`` placement map. All
    problems are collected rather than raised on the first one: an overlay is
    curated by hand in one sitting, and fixing four mistakes in four runs is
    four times the work.
    """
    problems: list[str] = []
    chapters_seen: set[str] = set()
    placement: dict[str, str] = {}

    for chapter in overlay["chapters"]:
        chapter_id = chapter["id"]
        if chapter_id in chapters_seen:
            problems.append(f"chapter id {chapter_id!r} is defined twice")
        chapters_seen.add(chapter_id)

        for entry in chapter.get("subtopics") or []:
            code = str(entry["code"])
            if code not in subtopics:
                problems.append(
                    f"chapter {chapter_id!r} lists subtopic {code!r}, "
                    f"which is not in the syllabus JSON"
                )
            owner = placement.get(code)
            if owner is not None:
                problems.append(
                    f"subtopic {code!r} is claimed by chapter {owner!r} and again by "
                    f"{chapter_id!r}; a recording must resolve to exactly one folder"
                )
            else:
                placement[code] = chapter_id

    return problems, placement


# ── Keyword lint ──────────────────────────────────────────────────────────────
#
# Keywords are the only part of a taxonomy a human writes freehand, and the only
# part whose mistakes are invisible. A wrong title is obvious on sight; a wrong
# keyword produces a confident, well-formed, wrong filename and no error at all.
#
# The audit that motivated this found the failure mode in its purest form.
# math.8_4 Conditional Probability carried "given that the first", "out of the
# ones who" and "now that we know" — a curator writing what the topic MEANS
# rather than what a teacher SAYS. Every token in those phrases is a stopword,
# so they fire on any transcript in any subject, and a 56-minute course
# orientation with no probability content in it was filed as Conditional
# Probability at maximum confidence.
#
# So the checks below are one question asked five ways: *can this phrase
# distinguish this node from the rest of the syllabus?* The three that can
# answer "no, never" are errors and stop generation; the two that can only say
# "this is weaker than you think" are warnings and print.


#: The retriever's tokeniser, duplicated on purpose.
#:
#: The lint's whole job is to predict what ``pipeline.classify.retriever`` will
#: do to a phrase, so the two patterns have to be the same one — but this script
#: is a standalone JSON-in/YAML-out tool, and importing the pipeline package for
#: one regex would drag pydantic and a sys.path hack in with it.
#: ``tests/test_keyword_lint.py`` asserts the two patterns are identical, so
#: drift fails the suite instead of quietly making the lint wrong.
_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-']*")


def tokenise(text: str) -> list[str]:
    """Exactly what the retriever does to a phrase before matching it."""
    return _TOKEN.findall(text.lower())


# Stopwords, built here rather than imported. A rejection has to be arguable —
# a curator reading "every token is a stopword" needs to be able to look at the
# list and agree — and that is worth more than 200 strings are worth as a
# dependency.
#
# What belongs: function words, auxiliaries, and the handful of content words
# that mean the same thing in every subject. What deliberately does NOT belong:
# any word that names a mathematical or scientific object, however generic it
# feels. "mean", "number", "value", "line", "set", "group", "point", "range",
# "term", "order", "form", "base", "side" and "average" are all real vocabulary
# in one of these syllabi, and a fixed list has no way to know which syllabus it
# is being applied to. Judging those is check 2's job, which measures a word
# against the syllabus in front of it instead of guessing from a list written
# somewhere else.

#: Articles, pronouns, prepositions, conjunctions.
#:
#: The direction words are absent on purpose. "Down a group" and "across a
#: period" are the two commonest sentences in a lecture on periodicity, and the
#: preposition is the only part of either worth matching; "up", "over", "under",
#: "above" and "below" go the same way. A preposition is a stopword when it is
#: doing grammar, and these are doing chemistry.
_FUNCTION_WORDS = """
    a an the this that these those there here
    i you he she it we they me him her us them
    my your his its our their whose
    who whom which what
    of in on at to for with without from by about into onto within
    between among through along around
    before after during while until since than then so because if unless
    and or but nor as like per out off back
"""

#: Auxiliaries and modals.
_AUXILIARIES = """
    is are was were be been being am
    do does did done doing have has had having
    will would shall should can could may might must let lets
"""

#: Degree, quantity and sequencing adverbs. Every one of these describes a
#: comparison or a moment, and every lecture in every subject contains both.
_DEGREE_WORDS = """
    not no none very just only also too more most less least much many few
    several all both each every any some other others another same different
    new old next last previous
    now already still yet again ever never always often sometimes usually
    quite rather almost nearly really actually basically simply obviously
    therefore however though although whether
"""

#: Light and discourse verbs — the vocabulary of explaining rather than of a
#: subject. Four of these are borderline and are here for a named reason:
#:
#:   given, taken  "given that the first" and "already been taken" are two of
#:                 the three keywords that filed a course orientation as
#:                 Conditional Probability. Real keywords survive because they
#:                 carry a noun: "divide in a given ratio" keeps "ratio".
#:   know          "now that we know" is the third of those three.
#:   use, used     opens a third of all Cambridge learning objectives ("use the
#:                 term...", "use a calculator"), and in none of them is it the
#:                 thing being taught.
#:   work          risky in a subject where work is a quantity — there is no
#:                 physics syllabus here — but "work out the answer" is pure
#:                 classroom filler and has to be catchable.
#:
#: "make" and its inflections were here and were removed: "bond making" is the
#: other half of "bond breaking" in every enthalpy lecture ever given, and with
#: "making" struck out the phrase is left leaning on "bond" alone, which
#: chemistry says everywhere.
_LIGHT_VERBS = """
    know knows knowing known think thinks thought want wants need needs
    go goes going gone get gets getting got
    take takes taking taken put puts
    look looks looking see sees seen watch
    say says said tell tells told ask asks
    come comes coming give gives given giving
    use uses used using call calls called
    remember notice consider happen happens
    work works working worked
"""

#: Nouns so general they name no topic, plus the first three ordinals.
#:
#: "time" is here on the Time & Timetables curator's own evidence: bare "time"
#: is said in every lecture in the subject, which is why that node's keywords
#: are "time zone" and "departure time" and never "time".
#:
#: The ordinals are the other half of "given that the first" — without them
#: that phrase keeps one non-stopword token, "first" appears in one node of the
#: whole Maths syllabus, and the keyword that misfiled a course orientation
#: reads as maximally discriminative. The spoken numerals "one" ... "ten" are
#: NOT here for the opposite reason: they were, and they turned "group two",
#: "group seven" and "period three" — three of the most reliable things a
#: chemistry teacher says out loud — into errors.
_GENERIC_NOUNS = """
    thing things stuff way ways time times part parts bit bits lot lots
    kind kinds sort sorts type types case cases
    question questions answer answers example examples
    ones first second third
    people everyone everybody someone somebody anyone nobody
    something anything everything nothing
"""

#: Classroom filler that survives into a transcript verbatim.
_FILLER = """
    okay ok right well yeah yes please guys alright
"""

STOPWORDS = frozenset(
    (
        _FUNCTION_WORDS + _AUXILIARIES + _DEGREE_WORDS + _LIGHT_VERBS + _GENERIC_NOUNS + _FILLER
    ).split()
)

#: A term spread over more than this share of the syllabus's nodes cannot
#: identify one of them.
#:
#: 0.15 of the ~75 nodes in these two syllabi is 11 nodes: a word spread over a
#: dozen of them can at best narrow the syllabus to a sixth of itself, and
#: narrowing is not identifying.
#:
#: Calibrated against both real overlays rather than picked for roundness. 0.15
#: is the tightest value that rejects nothing defensible: it catches "more
#: electrons", which leans entirely on a word chemistry says in a fifth of its
#: nodes. Tightening to 0.12 immediately starts rejecting "chlorine",
#: "catalyst" and "bonding", each of which really is the subject of the node
#: carrying it. Loosening to 0.22 catches nothing at all.
MAX_DOCUMENT_FREQUENCY = 0.15

#: Two nodes is not a corpus. Below this many, the document-frequency check does
#: not run at all rather than run on numbers that mean nothing. The stopword
#: check still applies — it is the fixed half of the same idea.
MIN_NODES_FOR_DOCUMENT_FREQUENCY = 8

#: A one-to-three-letter symbol with a numeric exponent: "dm-3", "s-1".
_UNIT_EXPONENT = re.compile(r"^[a-z]{1,3}-\d+$")


@dataclass(frozen=True)
class Finding:
    """One problem with one term, rendered as one line."""

    check: str
    severity: str  # "error" | "warning"
    node_id: str
    field: str  # "keywords" | "aliases"
    term: str
    detail: str

    def __str__(self) -> str:
        return f"{self.node_id} {self.field} {self.term!r}: {self.detail} [{self.check}]"


def _render(tokens: list[str]) -> str:
    return ", ".join(repr(token) for token in tokens)


def _terms_of(node: dict[str, Any]) -> list[tuple[str, str]]:
    """A node's curated terms as ``(field, term)``, in the order the retriever
    concatenates them."""
    return [(field, term) for field in ("keywords", "aliases") for term in node.get(field) or []]


# ── The five checks ───────────────────────────────────────────────────────────


def check_degenerate_tokenisation(term: str) -> str | None:
    """5. The tokeniser destroys the phrase before it is ever matched.

    Three shapes, all of them found in real overlays:

      * ``M+1`` becomes ``m`` and ``1``. Single characters are not words.
      * ``3:1 ratio`` becomes ``3``, ``1``, ``ratio`` — the colon carrying the
        whole meaning is dropped, and a teacher saying "three to one" matches
        neither digit. (The A2 Biology overlay still has ``3:1`` and
        ``9:3:3:1`` as keywords.)
      * ``mol dm-3`` survives tokenisation intact and is dead anyway, because
        no transcript contains "dm-3": the teacher says "moles per decimetre
        cubed".

    The unit rule is deliberately narrow — one to three letters before the
    hyphen. ``carbon-12`` and ``chlorine-35`` have the same shape, are said
    aloud exactly as written, and are correct keywords.
    """
    tokens = tokenise(term)
    if not tokens:
        return "tokenises to nothing, so it can never match"
    if all(len(token) == 1 for token in tokens):
        return (
            f"tokenises to {_render(tokens)} — single characters carry no meaning "
            f"and match any window that happens to contain them"
        )
    for word in term.lower().split():
        parts = tokenise(word)
        if len(parts) > 1 and all(len(part) == 1 for part in parts):
            return (
                f"{word!r} tokenises to {_render(parts)} — the punctuation holding it "
                f"together is dropped and nothing recognisable is left"
            )
    for token in tokens:
        if _UNIT_EXPONENT.match(token):
            return (
                f"{token!r} is printed unit notation, not speech; it survives "
                f"tokenisation unchanged and no transcript ever contains it"
            )
    return None


def check_stopword_only(term: str) -> str | None:
    """1. Every token is a stopword, so the phrase fires on any transcript."""
    tokens = tokenise(term)
    if tokens and all(token in STOPWORDS for token in tokens):
        return (
            f"every token is a stopword ({_render(tokens)}) — this is what the topic "
            f"means, not what a teacher says, and it matches any lecture in any subject"
        )
    return None


def check_no_discriminative_token(term: str, documents: list[frozenset[str]]) -> str | None:
    """2. Nothing in the phrase is rare enough *in this syllabus* to identify a node.

    The principled form of check 1, and the half a fixed list cannot do: a
    stopword list written in London has no way of knowing that "equation" is
    noise in a maths syllabus and decisive in a chemistry one.

    Two stages, and a keyword has to fail both to be rejected:

      1. Is any single word of it rare? A word in at most
         ``MAX_DOCUMENT_FREQUENCY`` of the nodes can carry the phrase on its
         own. Single characters do not count as words here — a bare "k" or "x"
         is a symbol, and a transcript spells symbols however Whisper felt.
      2. Are its words rare *together*? Counted exactly, by asking how many
         nodes' own text contains every word of the phrase — not estimated by
         multiplying the separate frequencies, which assumes an independence
         that subject vocabulary does not have.

    Stage 2 is the tuning that made this check usable. Judging a phrase by its
    commonest word alone rejected "hydrogen bond", "oxidation state" and
    "group 2" — terms whose words are individually everywhere in Chemistry and
    which are, jointly, the name of exactly one topic.
    """
    node_count = len(documents)
    if node_count < MIN_NODES_FOR_DOCUMENT_FREQUENCY:
        return None
    tokens = tokenise(term)
    carriers = [token for token in tokens if token not in STOPWORDS]
    if not carriers:
        return None  # check 1 says this better
    ceiling = max(1, int(MAX_DOCUMENT_FREQUENCY * node_count))
    frequency = {token: sum(token in document for document in documents) for token in carriers}
    if any(count <= ceiling for token, count in frequency.items() if len(token) > 1):
        return None
    together = sum(1 for document in documents if all(token in document for token in carriers))
    if together <= ceiling:
        return None
    spread = ", ".join(f"{token!r} in {count}" for token, count in frequency.items())
    return (
        f"nothing here is rare in this syllabus — of {node_count} nodes, {spread}, and "
        f"{together} contain all of it; a term that identifies one node appears in at most "
        f"{ceiling}"
    )


def find_alias_keyword_duplicates(node: dict[str, Any]) -> list[str]:
    """4. The same term in both ``keywords`` and ``aliases`` of one node.

    The retriever concatenates the two lists into the node's BM25 document, so
    a duplicated term is indexed twice — three times, since keywords are
    repeated in that document — and outweighs the terms beside it by an amount
    nobody chose. A warning rather than an error: it costs precision, not
    correctness, and the gate itself now de-duplicates identical tokenisations
    before counting hits.
    """
    keywords = {tuple(tokenise(term)) for term in node.get("keywords") or []}
    duplicates = []
    for term in node.get("aliases") or []:
        key = tuple(tokenise(term))
        if key and key in keywords:
            duplicates.append(term)
    return sorted(duplicates)


def find_sibling_collisions(children: list[dict[str, Any]]) -> dict[str, list[str]]:
    """3. One term on two nodes under one parent, as ``term -> node ids``.

    Separating siblings is the whole job of a keyword — by the time they are
    compared the parent is already decided — so a term carried by two of them
    does nothing but split the evidence between them. A warning, because the
    honest fix is sometimes to move the term up onto the parent rather than to
    delete it, and only the curator knows which.
    """
    owners: dict[tuple[str, ...], list[str]] = {}
    spelling: dict[tuple[str, ...], str] = {}
    for child in children:
        for _, term in _terms_of(child):
            key = tuple(tokenise(term))
            if not key:
                continue
            spelling.setdefault(key, term)
            if child["id"] not in owners.setdefault(key, []):
                owners[key].append(child["id"])
    return {spelling[key]: ids for key, ids in owners.items() if len(ids) > 1}


# ── Running the checks over a taxonomy ────────────────────────────────────────


def _classifiable(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every node at any depth, mirroring ``Taxonomy.classifiable_nodes()``.

    A node flagged out of classification is skipped: nothing it carries can
    ever match a transcript, and counting its keywords in the document
    frequencies would make the rest of the syllabus look rarer than it is.
    Recursive rather than two nested loops, so this also lints a taxonomy with
    a lesson layer under the subtopics.
    """
    out: list[dict[str, Any]] = []
    for node in nodes:
        if not node.get("exclude_from_classification"):
            out.append(node)
        out.extend(_classifiable(node.get("children") or []))
    return out


def node_documents(nodes: list[dict[str, Any]]) -> list[frozenset[str]]:
    """One bag of words per node — the corpus check 2 measures against.

    The document is the one the retriever indexes — title, short title,
    aliases, keywords and learning-outcome text — because it is that corpus's
    IDF that decides what BM25 pays attention to. A set, not a count: a term
    repeated ten times in one node is still one node, and the question here is
    how far a word is spread, never how loud it is.
    """
    documents = []
    for node in nodes:
        parts = [node.get("title") or "", node.get("short_title") or ""]
        parts += [term for _, term in _terms_of(node)]
        parts += [outcome["text"] for outcome in node.get("learning_outcomes") or []]
        documents.append(frozenset(tokenise(" ".join(parts))))
    return documents


def lint_taxonomy(taxonomy: dict[str, Any]) -> list[Finding]:
    """Every keyword problem in a built taxonomy, errors first.

    All of them rather than the first: an overlay is curated by hand in one
    sitting, and fixing four mistakes in four runs is four times the work. That
    is the contract ``check_overlay`` already keeps, and this keeps it too.
    """
    # ``pending_nodes`` via .get: this ingester always emits the key, and
    # reading it defensively is what lets the same lint be pointed at a
    # taxonomy YAML off disk — including the PDF ingester's A2 Biology one,
    # which has a lesson layer this handles and four errors this catches.
    roots = [*taxonomy["nodes"], *(taxonomy.get("pending_nodes") or [])]
    nodes = _classifiable(roots)
    documents = node_documents(nodes)
    findings: list[Finding] = []

    for node in nodes:
        for field, term in _terms_of(node):
            for check, detail in (
                ("degenerate-tokenisation", check_degenerate_tokenisation(term)),
                ("stopword-only", check_stopword_only(term)),
                (
                    "no-discriminative-token",
                    check_no_discriminative_token(term, documents),
                ),
            ):
                if detail:
                    findings.append(Finding(check, "error", node["id"], field, term, detail))

        for term in find_alias_keyword_duplicates(node):
            findings.append(
                Finding(
                    "duplicate-keyword-alias",
                    "warning",
                    node["id"],
                    "aliases",
                    term,
                    "is also in this node's keywords, so it is indexed twice and "
                    "outweighs the terms beside it",
                )
            )

        children = _classifiable(node.get("children") or [])
        for term, owners in find_sibling_collisions(children).items():
            findings.append(
                Finding(
                    "sibling-collision",
                    "warning",
                    node["id"],
                    "keywords",
                    term,
                    f"is on {len(owners)} of this chapter's subtopics "
                    f"({', '.join(owners)}); it can never tell them apart",
                )
            )

    findings.sort(key=lambda f: (f.severity != "error", f.node_id, f.check, f.term))
    return findings


#: How many warnings to spell out before summarising the rest. One sibling
#: collision usually means the same term is on several of a chapter's
#: subtopics, and a taxonomy with a lesson layer under the subtopics can raise
#: hundreds; a wall of them scrolls the error list off the screen, and the
#: error list is the half that stops the build.
MAX_WARNINGS_SHOWN = 25


def report_lint(findings: list[Finding], source: Path) -> list[Finding]:
    """Print the warnings to stderr, hand the errors back to the caller."""
    warnings = [f for f in findings if f.severity == "warning"]
    if warnings:
        shown = warnings[:MAX_WARNINGS_SHOWN]
        rest = len(warnings) - len(shown)
        print(
            f"warning: {len(warnings)} weak keyword(s) in {source}:\n    "
            + "\n    ".join(str(finding) for finding in shown)
            + (f"\n    ... and {rest} more" if rest else ""),
            file=sys.stderr,
        )
    return [f for f in findings if f.severity == "error"]


# ── Overlay merge ─────────────────────────────────────────────────────────────


def _subtopic_node(
    prefix: str, entry: dict[str, Any], subtopics: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    code = str(entry["code"])
    sub = subtopics[code]
    return {
        "id": f"{prefix}.{code.replace('.', '_')}",
        # Verbatim from the syllabus. This is what retrieval scores against.
        "title": sub["title"],
        # ...and this is what goes in the filename. Keywords only.
        "short_title": entry.get("short_title"),
        "syllabus_topic": code,
        # No drive_folder_name, ever: the placement unit is the chapter above.
        # A subtopic with a folder of its own splits a chapter in two.
        "verified": True,
        "aliases": entry.get("aliases", []),
        "keywords": entry.get("keywords", []),
        "learning_outcomes": _outcomes(code, sub["objectives"]),
    }


def build_taxonomy(
    overlay: dict[str, Any],
    subtopics: dict[str, dict[str, Any]],
    document: dict[str, Any],
) -> dict[str, Any]:
    """Merge syllabus content with the hand-written overlay."""
    prefix = overlay["node_id_prefix"]
    nodes: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    for chapter in overlay["chapters"]:
        entries = chapter.get("subtopics") or []
        topic_numbers = {str(entry["code"]).split(".")[0] for entry in entries}
        node = {
            "id": f"{prefix}.{chapter['id']}",
            "title": chapter["title"],
            "teaching_order": chapter.get("teaching_order"),
            "coursebook_chapter": chapter.get("coursebook_chapter"),
            # Set only when the chapter happens to sit inside a single syllabus
            # topic. One that straddles two carries no topic number rather than
            # an arbitrary one — see the module docstring.
            "syllabus_topic": next(iter(topic_numbers)) if len(topic_numbers) == 1 else None,
            "drive_folder_name": chapter.get("drive_folder_name"),
            # The chapter's identity is curated rather than extracted, but it is
            # asserted against a real Drive folder and carries no learning
            # outcomes of its own for anyone to have mistyped.
            "verified": True,
            "exclude_from_classification": chapter.get("exclude_from_classification", False),
            "aliases": chapter.get("aliases", []),
            "keywords": chapter.get("keywords", []),
            "children": [_subtopic_node(prefix, entry, subtopics) for entry in entries],
        }
        # No folder in Drive yet: still emitted, so the classifier recognises
        # the topic and routes to review instead of inventing a destination.
        (nodes if node["drive_folder_name"] else pending).append(node)

    return {
        "version": 1,
        "subject_id": overlay["subject_id"],
        "subject_name": overlay["subject_name"],
        "syllabus_code": overlay["syllabus_code"],
        "syllabus_years": overlay.get("syllabus_years") or document.get("syllabus_years"),
        "awarding_body": overlay.get("awarding_body", "Cambridge International"),
        "source": overlay.get("source", "Extracted from the official syllabus JSON export"),
        "subject_vocabulary": overlay.get("subject_vocabulary", []),
        "nodes": nodes,
        "pending_nodes": pending,
    }


def _prune(value: Any) -> Any:
    """Drop empty, null and false-flag fields so the emitted YAML stays readable.

    ``False`` is tested by identity, not membership: ``0 == False``, and a
    membership test would silently delete the Introduction chapter's
    ``teaching_order: 0`` and file it last instead of first.
    """
    if isinstance(value, dict):
        return {
            k: _prune(v) for k, v in value.items() if v is not False and v not in (None, [], {}, "")
        }
    if isinstance(value, list):
        return [_prune(v) for v in value]
    return value


# ── Output ────────────────────────────────────────────────────────────────────


def build_header(taxonomy: dict[str, Any], args: argparse.Namespace, revision: str | None) -> str:
    """The comment block at the top of the generated file.

    It names the *exact* command, arguments included, because the thing an
    editor of a generated file needs first is the way to stop editing it.
    """
    lines = [
        f"{taxonomy['subject_name']} {taxonomy['syllabus_code']} "
        f"({taxonomy['syllabus_years']})",
        "",
        "GENERATED — do not edit by hand. Regenerate with:",
        "  python scripts/ingest_syllabus_json.py \\",
        f"      --json {args.json.as_posix()} \\",
        f"      --overlay {args.overlay.as_posix()} \\",
        # --print-only has no --out to quote, but the command still needs one:
        # a header that names an incomplete command is worse than a placeholder
        # the reader can obviously see they have to fill in.
        f"      --out {args.out.as_posix() if args.out else '<path to the taxonomy YAML>'}",
    ]
    if revision:
        lines += ["", f"Syllabus revision: {revision}"]
    lines += [
        "",
        "Depth 0 is a CHAPTER — one Drive folder, and the unit a recording is filed",
        "into. Depth 1 is a SYLLABUS SUBTOPIC — the unit a filename is built from.",
        "They are not the same thing: one syllabus topic can be split across two of",
        "the chapters below.",
        "",
        "Subtopic titles and learning outcomes are verbatim from the syllabus export",
        "and are marked verified. short_title (what goes in the filename), keywords,",
        "aliases, Drive folder names and the chapter grouping come from the overlay:",
        "edit those there, not here.",
    ]
    return "".join(f"#{' ' + line if line else ''}\n" for line in lines) + "\n"


def report(
    taxonomy: dict[str, Any],
    *,
    repaired: int,
    unplaced: list[str],
    without_short_title: list[str],
) -> None:
    """Diagnostics go to stderr so ``--print-only`` stays pipeable."""
    subtopic_nodes = [
        sub for node in taxonomy["nodes"] + taxonomy["pending_nodes"] for sub in node["children"]
    ]
    outcomes = sum(len(sub["learning_outcomes"]) for sub in subtopic_nodes)
    chapters = len(taxonomy["nodes"]) + len(taxonomy["pending_nodes"])
    print(
        f"  {chapters} chapters ({len(taxonomy['nodes'])} with a Drive folder), "
        f"{len(subtopic_nodes)} subtopics, {outcomes} learning outcomes, "
        f"{len(unplaced)} unplaced subtopic codes\n"
        f"  repaired {repaired} mojibake strings",
        file=sys.stderr,
    )
    if unplaced:
        # Not fatal — a subject is often ingested before its overlay is
        # complete — but it is a silent coverage hole: no recording can ever be
        # classified into a code that no chapter contains.
        print(
            f"warning: {len(unplaced)} syllabus subtopic codes are in no chapter and are "
            f"therefore unreachable:\n    {', '.join(unplaced)}",
            file=sys.stderr,
        )
    if without_short_title:
        print(
            f"warning: {len(without_short_title)} subtopics have no short_title, so their "
            f"filenames would use the verbatim syllabus title:\n"
            f"    {', '.join(without_short_title)}",
            file=sys.stderr,
        )


def main() -> int:
    # The generated YAML is UTF-8 by construction — 0580 alone carries A′, ξ,
    # ⩽ and ∪, and 9701 carries Δ, σ, π. On Windows the console defaults to
    # cp1252, so --print-only died with UnicodeEncodeError on a file that
    # writes perfectly well with --out. Reconfigure rather than asking every
    # caller to set PYTHONIOENCODING: the encoding of our own output is this
    # script's business, and a diagnostic that crashes is worse than none.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--overlay", required=True, type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        help="Where to write the taxonomy YAML. Required unless --print-only.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Write the YAML to stdout instead of to a file, so it can be "
        "piped or diffed against the checked-in taxonomy. Diagnostics go to "
        "stderr, so the stdout side stays clean.",
    )
    parser.add_argument(
        "--lint-only",
        action="store_true",
        help="Run the overlay and keyword checks and report, writing nothing. "
        "Exits non-zero when there are errors, so a curator can iterate on an "
        "overlay's keywords without regenerating a taxonomy each time.",
    )
    args = parser.parse_args()
    # Conditional rather than `required=True`: --print-only writes to stdout
    # and --lint-only writes nothing at all, and demanding a path neither of
    # them will write to is what made the documented pipeable mode unusable.
    if args.out is None and not (args.print_only or args.lint_only):
        parser.error("--out is required unless --print-only or --lint-only is given")

    document, repaired = load_syllabus(args.json)
    subtopics = index_subtopics(document)
    overlay = yaml.safe_load(args.overlay.read_text(encoding="utf-8")) or {}

    problems, placement = check_overlay(overlay, subtopics)
    if problems:
        raise SystemExit(
            f"{args.overlay}: {len(problems)} problem(s) against {args.json}:\n  "
            + "\n  ".join(problems)
        )

    taxonomy = build_taxonomy(overlay, subtopics, document)
    unplaced = sorted(set(subtopics) - set(placement), key=_code_key)
    without_short_title = sorted(
        (
            str(entry["code"])
            for chapter in overlay["chapters"]
            for entry in chapter.get("subtopics") or []
            if not entry.get("short_title")
        ),
        key=_code_key,
    )

    # Keywords are checked against the *built* taxonomy rather than the raw
    # overlay: document frequency is a property of the whole syllabus, and
    # half the text it is measured over is the syllabus's own, which only
    # exists after the merge. Errors stop the run before anything is written —
    # a taxonomy that fails the lint would classify recordings wrongly and
    # silently, which is worse than not existing.
    errors = report_lint(lint_taxonomy(taxonomy), args.overlay)

    if args.lint_only:
        report(
            taxonomy,
            repaired=repaired,
            unplaced=unplaced,
            without_short_title=without_short_title,
        )
        if errors:
            print(
                f"{args.overlay}: {len(errors)} keyword error(s):\n  "
                + "\n  ".join(str(error) for error in errors),
                file=sys.stderr,
            )
            return 1
        print("  no keyword errors", file=sys.stderr)
        return 0

    if errors:
        raise SystemExit(
            f"{args.overlay}: {len(errors)} keyword error(s); nothing was written:\n  "
            + "\n  ".join(str(error) for error in errors)
        )

    text = yaml.safe_dump(_prune(taxonomy), sort_keys=False, allow_unicode=True, width=100)
    header = build_header(taxonomy, args, syllabus_revision(document))

    if args.print_only:
        print(header + text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(header + text, encoding="utf-8")
        print(f"wrote {args.out}", flush=True)  # ordered ahead of the stderr report

    report(
        taxonomy,
        repaired=repaired,
        unplaced=unplaced,
        without_short_title=without_short_title,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
