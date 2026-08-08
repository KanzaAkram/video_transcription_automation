"""Taxonomy invariants and folder resolution.

The tests that matter most here guard ADR-013: the folder ordinal, the
coursebook chapter, and the syllabus topic are three different numbers that
disagree in the user's Drive. Code that conflates them files bonding lectures
under stoichiometry, silently.
"""

from __future__ import annotations

from pipeline.config import PlacementSettings
from pipeline.placement import FolderResolver
from pipeline.syllabus import normalise_folder_name

RECORDINGS_ROOT = "5. AS Chemistry/1. Notes, Assignments & Class Recordings"


# ── The three numbering systems ───────────────────────────────────────────────


def test_teaching_order_and_coursebook_chapter_genuinely_disagree(taxonomy):
    """The premise behind ADR-013, asserted directly against the real data.

    If this ever stops being true the extra fields are harmless; if code
    starts *assuming* it is false, files go to the wrong chapter.
    """
    bonding = taxonomy.require("chem.chemical_bonding")
    moles = taxonomy.require("chem.moles_stoichiometry")

    assert bonding.teaching_order == 3 and bonding.coursebook_chapter == 4
    assert moles.teaching_order == 4 and moles.coursebook_chapter == 3
    # Transposed relative to each other — the exact trap.
    assert bonding.teaching_order == moles.coursebook_chapter
    assert moles.teaching_order == bonding.coursebook_chapter


def test_syllabus_topic_differs_from_both(taxonomy):
    halogeno = taxonomy.require("chem.halogenoalkanes")
    assert halogeno.teaching_order == 9
    assert halogeno.coursebook_chapter == 16
    assert halogeno.syllabus_topic == "15"


def test_two_chapters_can_share_a_coursebook_chapter(taxonomy):
    """Alcohols and Carboxylic Acids are both coursebook chapter 17 but are
    separate folders in Drive, so they must be separate nodes."""
    alcohols = taxonomy.require("chem.alcohols")
    acids = taxonomy.require("chem.carboxylic_acids_esters")
    assert alcohols.coursebook_chapter == acids.coursebook_chapter == 17
    assert alcohols.drive_folder_name != acids.drive_folder_name


# ── Structure ─────────────────────────────────────────────────────────────────


def test_node_ids_are_unique(taxonomy):
    ids = [n.id for n in taxonomy.all_nodes()]
    assert len(ids) == len(set(ids))


def test_children_are_linked_to_parents(taxonomy):
    child = taxonomy.require("chem.atomic_structure.rel_atomic_mass")
    assert child.parent_id == "chem.atomic_structure"
    assert child.depth == 1
    assert [a.id for a in taxonomy.ancestors(child.id)] == ["chem.atomic_structure"]


def test_intro_is_excluded_from_classification(taxonomy):
    """Admin sessions must not absorb real teaching content."""
    assert taxonomy.require("chem.introduction").exclude_from_classification
    assert "chem.introduction" not in {n.id for n in taxonomy.classifiable_nodes()}


def test_pending_nodes_are_recognised_but_not_placeable(taxonomy):
    """Topics with no folder yet still classify, so they route to review
    rather than being forced into a wrong folder."""
    energetics = taxonomy.require("chem.energetics")
    assert energetics.drive_folder_name is None
    assert energetics in taxonomy.classifiable_nodes()
    assert energetics not in taxonomy.placeable_nodes()


def test_asr_vocabulary_prioritises_subject_terms(taxonomy):
    vocab = taxonomy.vocabulary(limit=20)
    assert "stoichiometry" in vocab
    assert "halogenoalkane" in vocab
    assert len(vocab) <= 20


def test_teacher_shorthand_is_captured_as_an_alias(taxonomy):
    """"AOD" appears in the user's own filenames and in no syllabus document."""
    assert "AOD" in taxonomy.require("chem.atomic_structure").aliases


# ── Folder-name normalisation ─────────────────────────────────────────────────


def test_normalisation_strips_ordinal_and_chapter():
    assert normalise_folder_name("3. Chemical Bonding (Chapter 4)") == "chemical bonding"
    assert normalise_folder_name("10. Carboxylic Acids & Esters (Chapter 17)") == (
        "carboxylic acids & esters"
    )
    assert normalise_folder_name("0. Introduction") == "introduction"


def test_normalisation_discards_numbers_rather_than_comparing_them():
    """Two folders differing only in their numbers normalise identically.

    That is intentional: the numbers are unreliable, so matching falls back to
    the title rather than to arithmetic.
    """
    assert normalise_folder_name("3. Bonding (Chapter 4)") == normalise_folder_name(
        "9. Bonding (Chapter 1)"
    )


# ── Resolution ────────────────────────────────────────────────────────────────


def _resolver(drive, taxonomy, **kwargs):
    return FolderResolver(
        drive,
        taxonomy,
        PlacementSettings(**kwargs),
        recordings_root_id=RECORDINGS_ROOT,
    )


def test_exact_match_resolves(drive, taxonomy):
    result = _resolver(drive, taxonomy).resolve("chem.halogenoalkanes")
    assert result.resolved
    assert result.method == "exact"
    assert result.folder_path == "9. Halogenoalkanes (Chapter 16)"


def test_resolution_respects_the_number_mismatch(drive, taxonomy):
    """Chapter 4 must land in the folder numbered 3, and vice versa."""
    resolver = _resolver(drive, taxonomy)
    assert resolver.resolve("chem.chemical_bonding").folder_path == (
        "3. Chemical Bonding (Chapter 4)"
    )
    assert resolver.resolve("chem.moles_stoichiometry").folder_path == (
        "4. Moles and Stoichiometry (Chapter 3)"
    )


def test_subtopic_stops_at_deepest_existing_folder(drive, taxonomy):
    """The user's recordings sit directly in chapter folders, so a subtopic
    resolves to its chapter rather than inventing a subfolder."""
    result = _resolver(drive, taxonomy).resolve("chem.atomic_structure.rel_atomic_mass")
    assert result.resolved
    assert result.folder_path == "1. Atomic Structure (Chapter 1)"


def test_renumbered_folder_still_matches(drive, taxonomy):
    """A folder renamed from "9." to "12." must not break placement."""
    root = drive.root / RECORDINGS_ROOT
    (root / "9. Halogenoalkanes (Chapter 16)").rename(
        root / "12. Halogenoalkanes (Chapter 16)"
    )
    result = _resolver(drive, taxonomy).resolve("chem.halogenoalkanes")
    assert result.resolved
    assert result.method == "normalised"
    assert result.folder_path == "12. Halogenoalkanes (Chapter 16)"


def test_missing_folder_routes_to_review_not_creation(drive, taxonomy):
    """Never invent top-level structure in someone's Drive by default."""
    import shutil

    shutil.rmtree(drive.root / RECORDINGS_ROOT / "8. Alcohols (Chapter 17)")
    result = _resolver(drive, taxonomy).resolve("chem.alcohols")
    assert not result.resolved
    assert result.method == "review"


def test_folder_creation_when_explicitly_enabled(drive, taxonomy):
    import shutil

    shutil.rmtree(drive.root / RECORDINGS_ROOT / "8. Alcohols (Chapter 17)")
    result = _resolver(drive, taxonomy, allow_create_folders=True).resolve("chem.alcohols")
    assert result.resolved
    assert result.method == "created"


def test_results_are_cached(drive, taxonomy):
    resolver = _resolver(drive, taxonomy)
    first = resolver.resolve("chem.hydrocarbons")
    assert "chem.hydrocarbons" in resolver.cache
    assert resolver.resolve("chem.hydrocarbons") is first
