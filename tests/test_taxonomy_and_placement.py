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
    """Three numbers, three different values, on one chapter."""
    periodicity = taxonomy.require("chem.periodicity")
    assert periodicity.teaching_order == 13
    assert periodicity.coursebook_chapter == 10
    assert periodicity.syllabus_topic == "9"


def test_a_chapter_spanning_two_topics_carries_no_topic_number(taxonomy):
    """"9. Halogenoalkanes" holds 15.1 *and* 19.1 (primary amines), because the
    only AS route to an amine is a halogenoalkane reaction and a lecture that
    spends ten minutes on amines still belongs in one folder.

    The ingester gives such a chapter `syllabus_topic: None` rather than
    picking one of the two. An arbitrary pick would be worse than none: it
    reads as fact and there is no way to tell it was a guess.
    """
    halogeno = taxonomy.require("chem.halogenoalkanes")
    assert halogeno.teaching_order == 9
    assert halogeno.coursebook_chapter == 16
    assert halogeno.syllabus_topic is None
    assert {c.syllabus_topic for c in halogeno.children} == {"15.1", "19.1"}


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
    child = taxonomy.require("chem.1_1")
    assert child.parent_id == "chem.atomic_structure"
    assert child.depth == 1
    assert [a.id for a in taxonomy.ancestors(child.id)] == ["chem.atomic_structure"]


def test_the_tree_is_exactly_two_deep(taxonomy):
    """Chapter over subtopic, and nothing below.

    A third level would reintroduce the thing rule 4 forbids: a node under a
    node that has its own Drive folder, and therefore a recording with two
    defensible destinations.
    """
    assert {n.depth for n in taxonomy.all_nodes()} == {0, 1}
    for node in taxonomy.all_nodes():
        if node.depth == 1:
            assert taxonomy.require(node.parent_id).depth == 0
            assert node.drive_folder_name is None, (
                f"{node.id} has its own folder; placement splits at the chapter"
            )


def test_one_syllabus_topic_can_span_two_chapters(taxonomy):
    """9701 topic 1 is two Drive folders, which is why chapters are not topics."""
    assert [c.syllabus_topic for c in taxonomy.require("chem.atomic_structure").children] == [
        "1.1",
        "1.2",
    ]
    assert [c.syllabus_topic for c in taxonomy.require("chem.electrons_in_atoms").children] == [
        "1.3",
        "1.4",
    ]


def test_no_subtopic_code_is_claimed_by_two_chapters(taxonomy):
    """The invariant rule 4 rests on: one code, one folder."""
    owners: dict[str, str] = {}
    for node in taxonomy.all_nodes():
        if node.depth != 1:
            continue
        assert node.syllabus_topic not in owners, (
            f"{node.syllabus_topic} in both {owners.get(node.syllabus_topic)} and "
            f"{node.parent_id}"
        )
        owners[node.syllabus_topic] = node.parent_id


def test_every_subtopic_has_a_short_filename_label(taxonomy):
    """Rule 5. `label` falls back to the verbatim syllabus title, so a missing
    short_title is not an error anywhere — it is just a long filename."""
    for node in taxonomy.all_nodes():
        if node.depth == 1:
            assert node.short_title, f"{node.id} would be named with its syllabus sentence"
            assert len(node.short_title) <= 40, f"{node.id}: {node.short_title!r}"


def test_intro_is_excluded_from_classification(taxonomy):
    """Admin sessions must not absorb real teaching content."""
    assert taxonomy.require("chem.introduction").exclude_from_classification
    assert "chem.introduction" not in {n.id for n in taxonomy.classifiable_nodes()}


def test_pending_nodes_are_recognised_but_not_placeable(taxonomy):
    """Topics with no folder yet still classify, so they route to review
    rather than being forced into a wrong folder.

    21.1 Organic synthesis is the real case: it is taught inside no single
    organic chapter, and attaching it to one would let a multi-step-synthesis
    lecture drag a whole recording into the wrong folder.
    """
    synthesis = taxonomy.require("chem.organic_synthesis")
    assert synthesis.drive_folder_name is None
    assert synthesis in taxonomy.classifiable_nodes()
    assert synthesis not in taxonomy.placeable_nodes()


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
    result = _resolver(drive, taxonomy).resolve("chem.1_1")
    assert result.resolved
    assert result.folder_path == "1. Atomic Structure (Chapter 1)"


def test_a_split_topics_subtopics_resolve_to_different_folders(drive, taxonomy):
    """1.1 and 1.3 are one syllabus topic and two Drive folders.

    Resolving both by topic number would put them in the same place, which is
    the mistake the chapter layer exists to prevent.
    """
    resolver = _resolver(drive, taxonomy)
    assert resolver.resolve("chem.1_1").folder_path == "1. Atomic Structure (Chapter 1)"
    assert resolver.resolve("chem.1_3").folder_path == "2. Electrons in Atoms (Chapter 2)"


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
