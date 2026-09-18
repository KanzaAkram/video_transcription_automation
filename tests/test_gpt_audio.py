"""gpt-audio placement is validated against the taxonomy; invented names are dropped."""

from __future__ import annotations

from pathlib import Path

from pipeline.m4a.gpt_audio import match_node, taxonomy_outline, validate_placement
from pipeline.m4a.routing import detect_subject
from pipeline.syllabus import load_taxonomy

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_expected_entropy_labels_map_onto_a2_chemistry_nodes():
    taxonomy = load_taxonomy(REPO_ROOT / "config/syllabi/a2-chemistry-9701.yaml")
    hint = detect_subject(
        "2026-09-14 19.08.53 RA A2 Chemistry M_J 27", repo_root=REPO_ROOT
    )
    fields = validate_placement(
        {
            "class": "A2",
            "subject": "Chemistry",
            "chapter": "23. Chemical Energetics",
            "topic": "23.3 Entropy Change, ΔS",
            "subtopic": "Entropy change",
        },
        taxonomy,
        hint,
    )
    assert fields["class"] == "A2"
    assert fields["subject"] == "Chemistry"
    assert "Chemical Energetics" in fields["chapter"]
    assert fields["topic"] == "Entropy Change"
    assert fields["_chapter_id"] == "a2chem.topic_23"
    assert fields["_topic_id"] == "a2chem.23_3"


def test_invented_chapter_is_rejected():
    taxonomy = load_taxonomy(REPO_ROOT / "config/syllabi/a2-chemistry-9701.yaml")
    hint = detect_subject(
        "2026-09-14 19.08.53 RA A2 Chemistry M_J 27", repo_root=REPO_ROOT
    )
    fields = validate_placement(
        {
            "class": "A2",
            "subject": "Chemistry",
            "chapter": "Quantum Astrology",
            "topic": "Made Up Topic",
            "subtopic": "Nope",
        },
        taxonomy,
        hint,
    )
    assert fields["chapter"] == ""
    assert fields["topic"] == ""
    assert fields["_chapter_id"] == ""


def test_outline_lists_existing_chapters():
    taxonomy = load_taxonomy(REPO_ROOT / "config/syllabi/a2-chemistry-9701.yaml")
    outline = taxonomy_outline(taxonomy)
    assert "Chemical Energetics" in outline
    assert "Entropy" in outline
    energetics = next(n for n in taxonomy.nodes if n.id == "a2chem.topic_23")
    assert match_node("1. Chemical Energetics", [energetics]) is energetics
