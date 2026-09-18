"""Parser invariants for ``scripts/ingest_syllabus.py``.

The A2 Chemistry 23.3 regression is locked against the generated YAML in
``test_taxonomy_and_placement.py``. This file guards the font check that
caused it, so a future PDF with Δ in a heading fails here rather than by
gluing two subtopics together again.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "ingest_syllabus.py"


def _load_ingester():
    spec = importlib.util.spec_from_file_location("ingest_syllabus", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ingest = _load_ingester()


def test_times_new_roman_bold_counts_as_bold():
    """Δ in a Cambridge heading is TimesNewRomanPS-BoldMT, not Helvetica …-Bd."""
    assert ingest._font_is_bold("HelveticaNeueLTW1G-Bd")
    assert ingest._font_is_bold("TimesNewRomanPS-BoldMT")
    assert ingest._font_is_bold("HelveticaNeueLTW1G-BdIt")
    assert not ingest._font_is_bold("HelveticaNeueLTW1G-Lt")
    assert not ingest._font_is_bold("TimesNewRomanPSMT")


def test_pdf_parse_keeps_23_3_out_of_23_2():
    pdf = REPO_ROOT / "config" / "syllabi" / "sources" / "a2-chemistry-9701.pdf"
    if not pdf.exists():
        pytest.skip("A2 Chemistry 9701 PDF is not in sources/")
    topics = ingest.parse_pdf(pdf)
    energetics = topics["23"]
    assert "23.3" in energetics["subs"]
    assert energetics["subs"]["23.3"]["title"].startswith("Entropy change")
    last_22 = energetics["subs"]["23.2"]["outcomes"][-1]
    assert "23.3" not in last_22
    assert energetics["subs"]["23.3"]["outcomes"]
