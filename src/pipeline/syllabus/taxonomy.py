"""The syllabus tree.

Subject -> Chapter -> Subtopic -> Sub-subtopic, loaded from YAML.

The design point worth understanding is that a node carries **three
independent numbering fields** and never derives one from another:

    teaching_order      the leading "3." in your folder names
    coursebook_chapter  the "(Chapter 4)" in your folder names
    syllabus_topic      the 9701 topic number

In your Drive these disagree in both directions — "3. Chemical Bonding
(Chapter 4)" and "4. Moles and Stoichiometry (Chapter 3)" are transposed. Code
that assumes they agree files bonding lectures under stoichiometry. See
ADR-013.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, PrivateAttr

log = logging.getLogger(__name__)


class LearningOutcome(BaseModel):
    code: str
    text: str
    #: False until checked against the real syllabus PDF. The loader warns.
    verified: bool = False


class SyllabusNode(BaseModel):
    id: str
    title: str

    # Three numbering systems, none derived from another. See module docstring.
    teaching_order: int | None = None
    coursebook_chapter: int | None = None
    syllabus_topic: str | None = None

    #: The folder name observed in Drive. None means "no destination yet" —
    #: the classifier can still recognise the topic, but placement routes to
    #: review rather than inventing a folder.
    drive_folder_name: str | None = None

    #: What teachers actually say, which is not what the syllabus writes.
    #: "AOD" appears in your own filenames and in no syllabus document.
    aliases: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    learning_outcomes: list[LearningOutcome] = Field(default_factory=list)

    verified: bool = False
    #: Admin sessions (course intros) should not absorb real content.
    exclude_from_classification: bool = False

    children: list[SyllabusNode] = Field(default_factory=list)

    # Populated by the loader.
    parent_id: str | None = None
    depth: int = 0
    path: list[str] = Field(default_factory=list)

    @property
    def title_path(self) -> str:
        return " > ".join([*self.path, self.title])

    def search_document(self) -> str:
        """Text used for lexical retrieval.

        Title, aliases, keywords and outcome text concatenated. Keywords are
        repeated because rare domain terms should dominate the BM25 score —
        "halogenoalkane" appearing at all is close to conclusive.
        """
        parts = [self.title, *self.aliases, *self.keywords, *self.keywords]
        parts += [o.text for o in self.learning_outcomes]
        return " ".join(parts)

    def semantic_document(self) -> str:
        """Text used for embedding. Outcome prose, which paraphrases well."""
        parts = [self.title_path]
        parts += [o.text for o in self.learning_outcomes]
        if not self.learning_outcomes:
            parts += self.keywords
        return ". ".join(parts)


SyllabusNode.model_rebuild()


class Taxonomy(BaseModel):
    version: int = 1
    subject_id: str
    subject_name: str
    syllabus_code: str | None = None
    syllabus_years: str | None = None
    awarding_body: str | None = None
    source: str | None = None
    subject_vocabulary: list[str] = Field(default_factory=list)
    nodes: list[SyllabusNode] = Field(default_factory=list)
    #: Topics with no Drive folder yet. Kept so the classifier can recognise
    #: them and route to review instead of forcing a wrong match.
    pending_nodes: list[SyllabusNode] = Field(default_factory=list)

    #: id -> node, built by the loader. Private so it is not serialised.
    _index: dict[str, SyllabusNode] = PrivateAttr(default_factory=dict)

    # ── Access ───────────────────────────────────────────────────────────────

    def all_nodes(self, *, include_pending: bool = True) -> list[SyllabusNode]:
        out: list[SyllabusNode] = []

        def walk(nodes: list[SyllabusNode]) -> None:
            for node in nodes:
                out.append(node)
                walk(node.children)

        walk(self.nodes)
        if include_pending:
            walk(self.pending_nodes)
        return out

    def classifiable_nodes(self) -> list[SyllabusNode]:
        return [n for n in self.all_nodes() if not n.exclude_from_classification]

    def placeable_nodes(self) -> list[SyllabusNode]:
        """Nodes that have somewhere in Drive to put a file."""
        return [n for n in self.all_nodes() if n.drive_folder_name]

    def get(self, node_id: str) -> SyllabusNode | None:
        return self._index.get(node_id)

    def require(self, node_id: str) -> SyllabusNode:
        node = self.get(node_id)
        if node is None:
            raise KeyError(f"unknown syllabus node: {node_id!r}")
        return node

    def ancestors(self, node_id: str) -> list[SyllabusNode]:
        """Root-first chain above `node_id`. Drives folder descent."""
        chain: list[SyllabusNode] = []
        node = self.get(node_id)
        while node and node.parent_id:
            parent = self.get(node.parent_id)
            if parent is None:
                break
            chain.append(parent)
            node = parent
        return list(reversed(chain))

    # ── ASR priming ──────────────────────────────────────────────────────────

    def vocabulary(self, limit: int = 60) -> list[str]:
        """Terms to prime the ASR decoder with.

        Subject-level vocabulary first, then per-node keywords by frequency.
        These are precisely the words generic speech recognition mangles, and
        precisely the words classification then depends on — so getting them
        right compounds.
        """
        seen: dict[str, int] = {}
        for term in self.subject_vocabulary:
            seen[term] = seen.get(term, 0) + 10  # subject terms outrank node terms
        for node in self.all_nodes():
            for term in node.keywords + node.aliases:
                if len(term.split()) <= 3:  # single tokens and short phrases only
                    seen[term] = seen.get(term, 0) + 1
        ranked = sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))
        return [term for term, _ in ranked[:limit]]

    def initial_prompt(self, limit: int = 40) -> str:
        """A sentence of context for the ASR decoder.

        Whisper conditions on this, so seeing the correct spellings before it
        hears them measurably improves specialist terms.
        """
        terms = ", ".join(self.vocabulary(limit))
        return (
            f"A {self.subject_name} lecture covering: {terms}."
        )


def _link(
    nodes: list[SyllabusNode],
    parent: SyllabusNode | None,
    index: dict[str, SyllabusNode],
) -> None:
    for node in nodes:
        node.parent_id = parent.id if parent else None
        node.depth = (parent.depth + 1) if parent else 0
        node.path = ([*parent.path, parent.title] if parent else [])
        if node.id in index:
            raise ValueError(f"duplicate syllabus node id: {node.id!r}")
        index[node.id] = node
        _link(node.children, node, index)


def load_taxonomy(path: Path | str) -> Taxonomy:
    """Load and validate a taxonomy YAML file.

    Warns about unverified nodes rather than failing: the shipped Chemistry
    file is a working template whose learning outcomes still need checking
    against the real syllabus PDF, and a hard failure would block the dry run
    for no good reason.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    taxonomy = Taxonomy.model_validate(raw)

    index: dict[str, SyllabusNode] = {}
    _link(taxonomy.nodes, None, index)
    _link(taxonomy.pending_nodes, None, index)
    taxonomy._index = index

    # Warn about *text that might be wrong*, not about every node whose
    # `verified` flag is false. A hand-added lesson node is unverified by
    # design — it is teaching structure, not syllabus content — and warning
    # about it on every load would be a permanent, expected warning, which is
    # the fastest way to train someone to ignore warnings entirely.
    #
    # The real risk is learning-outcome text that was typed rather than
    # extracted, because that is what retrieval scores against.
    suspect = [
        n.id
        for n in taxonomy.all_nodes()
        if any(not o.verified for o in n.learning_outcomes)
        or (not n.verified and not n.learning_outcomes)
    ]
    if suspect:
        log.warning(
            "%s: %d of %d syllabus nodes carry unverified learning-outcome text "
            "(%s%s). Check them against the syllabus PDF before production use — "
            "see the header of %s.",
            taxonomy.subject_id,
            len(suspect),
            len(taxonomy.all_nodes()),
            ", ".join(suspect[:3]),
            "..." if len(suspect) > 3 else "",
            path.name,
        )
    return taxonomy


# ── Folder-name normalisation ─────────────────────────────────────────────────

_ORDINAL_PREFIX = re.compile(r"^\s*\d+\s*[.)-]\s*")
_CHAPTER_SUFFIX = re.compile(r"\s*\((?:chapter|ch\.?)\s*\d+\)\s*$", re.IGNORECASE)


def normalise_folder_name(name: str) -> str:
    """Strip the ordinal prefix and chapter suffix for fuzzy matching.

        "3. Chemical Bonding (Chapter 4)" -> "chemical bonding"

    Used only as a *fallback* when exact matching fails. The numbers are
    deliberately discarded here rather than compared, because comparing them
    is exactly the mistake that transposes bonding and stoichiometry.
    """
    stripped = _ORDINAL_PREFIX.sub("", name)
    stripped = _CHAPTER_SUFFIX.sub("", stripped)
    return " ".join(stripped.lower().split())
