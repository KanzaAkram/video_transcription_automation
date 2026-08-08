"""Candidate retrieval — narrowing the syllabus before the LLM sees it.

Two independent scorers over windowed transcript text, fused:

  * **Lexical (BM25)** catches the decisive rare terms. "Halogenoalkane"
    appears in one chapter and nowhere else; fourteen occurrences in a window
    is close to conclusive, and BM25's rarity weighting captures that.

  * **Semantic (embeddings)** catches paraphrase. A teacher can spend ten
    minutes explaining ionisation energy without ever saying the phrase.

They fail differently, which is exactly why fusing them beats either alone.

Aggregating per-window rather than over the whole transcript produces a
**coverage profile** — "78% relative atomic mass, 18% atomic structure" — which
is what makes multi-topic lectures tractable. See docs/CLASSIFICATION.md.

BM25 has a pure-Python fallback so the dry run works with no extra
dependencies; embeddings degrade to lexical-only if sentence-transformers is
not installed.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from pipeline.config import ClassifySettings, RetrievalSettings
from pipeline.models import Transcript
from pipeline.syllabus import SyllabusNode, Taxonomy

log = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-']*")

#: A window must contain at least this many of the winning node's own terms to
#: count toward that node's coverage. Without it, windows containing no subject
#: content at all — "can everybody hear me at the back", homework admin, the
#: silence before a lecture starts — attach to whichever node scores least
#: badly and inflate its coverage. Scores alone cannot express this, because
#: normalising per window always puts *some* node at 1.0.
MIN_KEYWORD_HITS = 1


def tokenise(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Window:
    index: int
    start_s: float
    end_s: float
    text: str


@dataclass
class Candidate:
    node: SyllabusNode
    score: float
    #: Share of **all** windows in which this node was the top match, 0-1.
    #: Answers "how much of the lecture was this topic" — the right question
    #: for the coverage floor, since a topic occupying four minutes of ninety
    #: is suspicious however confident the match.
    coverage: float
    #: Share of the windows that matched **anything**, 0-1. Answers "of the
    #: content that could be classified, how dominant is this" — the right
    #: question for confidence, because a lecture full of admin chatter and
    #: exam advice should not look uncertain when its subject matter is
    #: unambiguous.
    share: float
    peak_score: float
    top_windows: list[int] = field(default_factory=list)


@dataclass
class CandidateSet:
    candidates: list[Candidate]
    windows: list[Window]
    #: Windows that matched no node at all — silence, admin, homework notices.
    unmatched_windows: int = 0

    def top(self, k: int) -> list[Candidate]:
        return self.candidates[:k]

    def coverage_profile(self) -> dict[str, float]:
        return {c.node.id: c.coverage for c in self.candidates if c.coverage > 0}


# ── BM25 ──────────────────────────────────────────────────────────────────────


class _BM25:
    """Minimal BM25 Okapi.

    Implemented inline rather than pulled from a library because it is twenty
    lines, and it keeps the dry run dependency-free. `rank_bm25` is used
    automatically when installed.
    """

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.corpus = corpus
        self.k1, self.b = k1, b
        self.doc_len = [len(d) for d in corpus]
        self.avg_len = (sum(self.doc_len) / len(corpus)) if corpus else 0.0
        self.freqs = [Counter(d) for d in corpus]
        df: Counter[str] = Counter()
        for doc in corpus:
            df.update(set(doc))
        n = len(corpus)
        self.idf = {
            term: math.log(1 + (n - count + 0.5) / (count + 0.5)) for term, count in df.items()
        }

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for i, freq in enumerate(self.freqs):
            score = 0.0
            for term in query:
                if term not in freq:
                    continue
                tf = freq[term]
                denom = tf + self.k1 * (
                    1 - self.b + self.b * self.doc_len[i] / (self.avg_len or 1)
                )
                score += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / denom
            out.append(score)
        return out


# ── Retriever ─────────────────────────────────────────────────────────────────


class HybridRetriever:
    def __init__(
        self,
        taxonomy: Taxonomy,
        classify_settings: ClassifySettings,
        retrieval_settings: RetrievalSettings,
    ) -> None:
        self.taxonomy = taxonomy
        self.settings = classify_settings
        self.weights = retrieval_settings
        self.nodes = taxonomy.classifiable_nodes()
        self._bm25 = _BM25([tokenise(n.search_document()) for n in self.nodes])
        self._embedder = None
        self._node_embeddings = None

    # ── Windowing ────────────────────────────────────────────────────────────

    def windows(self, transcript: Transcript) -> list[Window]:
        """Split into overlapping windows.

        Four minutes is long enough to hold a coherent explanation and short
        enough that a window rarely spans two topics. The overlap stops a
        topic transition landing on a boundary and being attributed to neither.
        """
        size = self.settings.window_seconds
        overlap = self.settings.window_overlap_seconds
        step = max(size - overlap, 1)
        duration = transcript.duration_s or (
            transcript.segments[-1].end_s if transcript.segments else 0.0
        )

        out: list[Window] = []
        start = 0.0
        index = 0
        while start < max(duration, 1.0):
            end = min(start + size, duration)
            text = transcript.window(start, end)
            if text.strip():
                out.append(Window(index=index, start_s=start, end_s=end, text=text))
                index += 1
            if end >= duration:
                break
            start += step
        return out

    # ── Embeddings (optional) ────────────────────────────────────────────────

    def _load_embedder(self):
        if self._embedder is not None:
            return self._embedder
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            log.info(
                "sentence-transformers not installed; retrieval is lexical-only. "
                'Install with: pip install -e ".[classify]"'
            )
            self._embedder = False
            return False
        self._embedder = SentenceTransformer(self.settings.embedding_model)
        self._node_embeddings = self._embedder.encode(
            [n.semantic_document() for n in self.nodes], normalize_embeddings=True
        )
        return self._embedder

    def _semantic_scores(self, text: str) -> list[float] | None:
        embedder = self._load_embedder()
        if embedder is False:
            return None
        vector = embedder.encode([text], normalize_embeddings=True)[0]
        return [float(vector @ node_vec) for node_vec in self._node_embeddings]

    # ── Retrieval ────────────────────────────────────────────────────────────

    def retrieve(self, transcript: Transcript) -> CandidateSet:
        windows = self.windows(transcript)
        if not windows:
            return CandidateSet(candidates=[], windows=[])

        totals = [0.0] * len(self.nodes)
        peaks = [0.0] * len(self.nodes)
        top_hits: Counter[int] = Counter()
        top_windows: dict[int, list[int]] = {}

        for window in windows:
            fused = self._score_window(window.text)
            tokens = set(tokenise(window.text))
            for i, score in enumerate(fused):
                totals[i] += score
                peaks[i] = max(peaks[i], score)
            best = max(range(len(fused)), key=lambda i: fused[i])
            # Gate on actual vocabulary presence, not on score. Per-window
            # normalisation puts the best node at 1.0 every time, so a score
            # threshold here would be dead code (and was, until this fix).
            if self._keyword_hits(self.nodes[best], tokens) >= MIN_KEYWORD_HITS:
                top_hits[best] += 1
                top_windows.setdefault(best, []).append(window.index)

        total_windows = len(windows)
        matched_windows = sum(top_hits.values())
        candidates = [
            Candidate(
                node=self.nodes[i],
                score=totals[i] / total_windows,
                coverage=top_hits.get(i, 0) / total_windows,
                share=(top_hits.get(i, 0) / matched_windows) if matched_windows else 0.0,
                peak_score=peaks[i],
                top_windows=top_windows.get(i, []),
            )
            for i in range(len(self.nodes))
        ]
        # Coverage first: a node that dominated the lecture beats one that
        # scored high in a single window.
        candidates.sort(key=lambda c: (c.coverage, c.score), reverse=True)
        return CandidateSet(
            candidates=candidates,
            windows=windows,
            unmatched_windows=total_windows - matched_windows,
        )

    @staticmethod
    def _keyword_hits(node: SyllabusNode, tokens: set[str]) -> int:
        """How many of `node`'s own terms appear in a window.

        A multi-word keyword counts only when every one of its tokens is
        present, so "relative atomic mass" does not fire on a bare "mass".
        """
        hits = 0
        for term in (*node.keywords, *node.aliases):
            parts = tokenise(term)
            if parts and all(part in tokens for part in parts):
                hits += 1
        return hits

    def _score_window(self, text: str) -> list[float]:
        query = tokenise(text)
        lexical = self._bm25.scores(query)
        lexical = _normalise(lexical)

        semantic = self._semantic_scores(text)
        if semantic is None:
            return lexical
        semantic = _normalise(semantic)
        wl, ws = self.weights.lexical_weight, self.weights.semantic_weight
        return [wl * lexical[i] + ws * semantic[i] for i in range(len(lexical))]


def _normalise(scores: list[float]) -> list[float]:
    """Scale to 0-1 so lexical and semantic scores can be combined.

    Note the consequence: within a window the best node is always 1.0,
    regardless of how good the match actually is. Absolute quality must
    therefore be judged some other way — see MIN_KEYWORD_HITS.
    """
    if not scores:
        return scores
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [0.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]
