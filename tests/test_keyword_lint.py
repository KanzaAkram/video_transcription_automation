"""The build-time keyword lint in ``scripts/ingest_syllabus_json.py``.

Every check gets the same two tests: a keyword that must be rejected, taken
where possible from a real overlay, and a legitimate one beside it that must
survive. The second half matters as much as the first — a lint that rejects
"hydrogen bond" or "group 2" would be turned off within a week, and then the
next "given that the first" would go in unnoticed.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "ingest_syllabus_json.py"


def _load_ingester():
    """Import the ingester, which is a script rather than a package module."""
    spec = importlib.util.spec_from_file_location("ingest_syllabus_json", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # Registered before exec, not after: @dataclass resolves its annotations
    # through sys.modules[cls.__module__], and a module loaded outside the
    # import system is not in there unless it is put there.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ingest = _load_ingester()


def _documents(*texts: str) -> list[frozenset[str]]:
    """A node corpus for check 2, one bag of words per node."""
    return [frozenset(ingest.tokenise(text)) for text in texts]


# ── The tokeniser the lint models ─────────────────────────────────────────────


def test_tokeniser_matches_the_retrievers():
    """The lint predicts what the retriever will do to a phrase.

    If the retriever's tokeniser changes and this copy does not, every check
    below silently starts answering a question nobody asked — so the drift is
    a test failure rather than a comment nobody rereads.
    """
    from pipeline.classify import retriever

    assert ingest._TOKEN.pattern == retriever._TOKEN.pattern
    assert ingest.tokenise("Mol dm-3, M+1") == retriever.tokenise("Mol dm-3, M+1")


# ── 1. Stopword-only ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "term",
    [
        "given that the first",  # filed a course orientation as Conditional Probability
        "now that we know",
        "out of the ones who",
        "already been taken",
        "work out the answer",
    ],
)
def test_stopword_only_rejects_english_dressed_as_a_keyword(term):
    detail = ingest.check_stopword_only(term)
    assert detail and "every token is a stopword" in detail


@pytest.mark.parametrize(
    "term",
    [
        "conditional probability",
        "reduced sample space",
        "one less in the bag",  # colloquial, but "bag" is said in one topic only
        "divide in a given ratio",  # "given" survives when it carries a noun
        "down a group",  # direction words are chemistry here, not grammar
        "group two",  # spoken numerals are not stopwords, on purpose
    ],
)
def test_stopword_only_keeps_real_teacher_speech(term):
    assert ingest.check_stopword_only(term) is None


# ── 2. No discriminative token ────────────────────────────────────────────────

#: Twenty nodes that all talk about equations and one that does not — the
#: shape of a real maths syllabus, where "equation" is not a stopword in
#: English and is noise here.
MATHS_LIKE = _documents(
    *(
        ["solve the equation to find the value of x"] * 20
        + ["rationalise the denominator of a surd"]
    )
)


def test_no_discriminative_token_rejects_a_word_this_syllabus_says_everywhere():
    detail = ingest.check_no_discriminative_token("solve the equation", MATHS_LIKE)
    assert detail and "nothing here is rare in this syllabus" in detail
    assert "'equation' in 20" in detail


def test_no_discriminative_token_keeps_a_word_this_syllabus_says_once():
    assert ingest.check_no_discriminative_token("rationalise the denominator", MATHS_LIKE) is None


def test_no_discriminative_token_keeps_common_words_that_are_rare_together():
    """The tuning that makes the check usable on a real syllabus.

    "hydrogen" and "bond" are each all over a chemistry syllabus, and the two
    of them side by side name exactly one subtopic. Judging the phrase by its
    commonest word alone rejected "hydrogen bond", "oxidation state" and
    "group 2"; judging it by how often the words appear *together* does not.
    """
    corpus = _documents(
        *(
            ["hydrogen is transferred"] * 10
            + ["a covalent bond forms"] * 10
            + ["hydrogen bond between molecules"]
        )
    )
    assert ingest.check_no_discriminative_token("hydrogen bond", corpus) is None
    assert ingest.check_no_discriminative_token("hydrogen", corpus) is not None


def test_no_discriminative_token_is_silent_on_a_corpus_too_small_to_measure():
    assert ingest.check_no_discriminative_token("solve the equation", MATHS_LIKE[:4]) is None


# ── 3. Sibling collision ──────────────────────────────────────────────────────


def test_sibling_collision_finds_a_keyword_on_two_subtopics_of_one_chapter():
    children = [
        {"id": "bio.12_2.the_link_reaction", "keywords": ["decarboxylation", "acetyl coenzyme A"]},
        {"id": "bio.12_2.the_krebs_cycle", "keywords": ["Decarboxylation"], "aliases": ["krebs"]},
    ]
    collisions = ingest.find_sibling_collisions(children)
    assert list(collisions) == ["decarboxylation"]
    assert collisions["decarboxylation"] == [
        "bio.12_2.the_link_reaction",
        "bio.12_2.the_krebs_cycle",
    ]


def test_sibling_collision_leaves_siblings_that_separate_cleanly_alone():
    children = [
        {"id": "chem.1_1", "keywords": ["proton", "neutron"]},
        {"id": "chem.1_2", "keywords": ["isotope"], "aliases": ["isotopes"]},
    ]
    assert ingest.find_sibling_collisions(children) == {}


# ── 4. Duplicated between keywords and aliases ────────────────────────────────


def test_duplicate_keyword_alias_is_found_however_it_is_spelled():
    node = {
        "id": "bio.12_1.the_respiratory_quotient",
        "keywords": ["respiratory quotient", "RQ"],
        "aliases": ["Respiratory Quotient", "carbon dioxide produced"],
    }
    assert ingest.find_alias_keyword_duplicates(node) == ["Respiratory Quotient"]


def test_an_alias_that_is_not_a_keyword_is_not_a_duplicate():
    node = {
        "id": "chem.1_2",
        "keywords": ["mass number", "same chemical properties"],
        "aliases": ["isotopes", "relative isotopic mass"],
    }
    assert ingest.find_alias_keyword_duplicates(node) == []


# ── 5. Degenerate tokenisation ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "term, expected",
    [
        ("M+1", "'m', '1'"),  # the audit's example
        ("m/z", "'m', 'z'"),  # live in the Chemistry overlay
        ("3:1 ratio", "'3', '1'"),  # live in the A2 Biology overlay
        ("9:3:3:1", "'9', '3', '3', '1'"),
        ("mol dm-3", "'dm-3'"),  # printed units are not spoken units
        ("+", "never match"),
    ],
)
def test_degenerate_tokenisation_rejects_notation_the_tokeniser_destroys(term, expected):
    detail = ingest.check_degenerate_tokenisation(term)
    assert detail and expected in detail


@pytest.mark.parametrize(
    "term",
    [
        "carbon-12",  # said out loud exactly as written, unlike "dm-3"
        "chlorine-35",
        "s orbital",  # a single character is fine beside a real word
        "p orbitals overlap",
        "109.5 degrees",  # "109" survives tokenisation and carries the phrase
        "relative atomic mass",
        "sp3 hybridisation",
    ],
)
def test_degenerate_tokenisation_keeps_notation_that_survives_the_tokeniser(term):
    assert ingest.check_degenerate_tokenisation(term) is None


# ── The lint over a whole taxonomy ────────────────────────────────────────────


def _taxonomy(**chapter: Any) -> dict[str, Any]:
    return {"nodes": [chapter], "pending_nodes": []}


def test_lint_reports_every_problem_at_once_with_errors_first():
    """One run has to be enough to fix an overlay in one sitting."""
    taxonomy = _taxonomy(
        id="math.probability",
        title="Probability",
        keywords=["probability"],
        children=[
            {
                "id": "math.8_4",
                "title": "Conditional probability",
                "keywords": ["given that the first", "conditional probability", "tree diagram"],
                "aliases": ["conditional probability"],
            },
            {
                "id": "math.8_3",
                "title": "Combined events",
                "keywords": ["tree diagram", "and rule"],
            },
        ],
    )
    findings = ingest.lint_taxonomy(taxonomy)
    assert [f.severity for f in findings] == sorted(
        (f.severity for f in findings), key=lambda s: s != "error"
    )
    by_check = {f.check: f for f in findings}
    assert by_check["stopword-only"].term == "given that the first"
    assert by_check["stopword-only"].node_id == "math.8_4"
    assert by_check["duplicate-keyword-alias"].node_id == "math.8_4"
    assert by_check["sibling-collision"].term == "tree diagram"
    assert {f.check for f in findings if f.severity == "warning"} == {
        "duplicate-keyword-alias",
        "sibling-collision",
    }


def test_a_node_out_of_classification_is_not_linted():
    """Its keywords can never match a transcript, so they can never misfile one.

    The course-introduction chapter carries "paper 1" and "paper 2" for the
    naming layer's benefit and is excluded from retrieval; linting it would
    report problems whose fix is to delete something that is doing its job.
    """
    taxonomy = _taxonomy(
        id="chem.introduction",
        title="Introduction",
        exclude_from_classification=True,
        keywords=["out of the ones who"],
        children=[],
    )
    assert ingest.lint_taxonomy(taxonomy) == []


# ── --lint-only, end to end ───────────────────────────────────────────────────


def _write_inputs(tmp_path: Path, *, keywords: list[str]) -> tuple[Path, Path]:
    syllabus = {
        "syllabus_years": "2025-2027",
        "topics": [
            {
                "sub_topics": [
                    {
                        "sub_topic_number": "1.1",
                        "sub_topic_name": "Surds",
                        "learning_objectives": [
                            {"objective_number": 1, "description": "Understand and use surds"}
                        ],
                    }
                ]
            }
        ],
    }
    overlay = {
        "node_id_prefix": "demo",
        "subject_id": "demo",
        "subject_name": "Demo",
        "syllabus_code": "0000",
        "chapters": [
            {
                "id": "number",
                "title": "Number",
                "drive_folder_name": "1. Number",
                "subtopics": [{"code": "1.1", "short_title": "Surds", "keywords": keywords}],
            }
        ],
    }
    json_path = tmp_path / "syllabus.json"
    overlay_path = tmp_path / "overlay.yaml"
    json_path.write_text(json.dumps(syllabus), encoding="utf-8")
    overlay_path.write_text(yaml.safe_dump(overlay), encoding="utf-8")
    return json_path, overlay_path


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def test_lint_only_reports_without_writing_anything(tmp_path):
    json_path, overlay_path = _write_inputs(
        tmp_path, keywords=["rationalise the denominator", "now that we know"]
    )
    out = tmp_path / "taxonomy.yaml"
    result = _run(
        "--json", str(json_path), "--overlay", str(overlay_path), "--lint-only", "--out", str(out)
    )
    assert result.returncode == 1
    assert "now that we know" in result.stderr
    assert not out.exists()


def test_lint_only_needs_no_output_path_and_passes_a_clean_overlay(tmp_path):
    json_path, overlay_path = _write_inputs(tmp_path, keywords=["rationalise the denominator"])
    result = _run("--json", str(json_path), "--overlay", str(overlay_path), "--lint-only")
    assert result.returncode == 0, result.stderr
    assert "no keyword errors" in result.stderr


def test_a_keyword_error_fails_generation_and_writes_no_taxonomy(tmp_path):
    """The whole point: a taxonomy that would misfile recordings never exists.

    A stale taxonomy on disk is recoverable; a fresh one that quietly names
    lectures wrongly is what the audit found.
    """
    json_path, overlay_path = _write_inputs(tmp_path, keywords=["given that the first"])
    out = tmp_path / "taxonomy.yaml"
    result = _run("--json", str(json_path), "--overlay", str(overlay_path), "--out", str(out))
    assert result.returncode != 0
    assert "keyword error" in result.stderr
    assert not out.exists()
