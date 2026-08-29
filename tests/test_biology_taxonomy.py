"""Invariants for the generated A2 Biology taxonomy.

This file is machine-generated from the official 9700 PDF by
``scripts/ingest_syllabus.py``, so these tests guard the *generator* as much as
the data. A parser regression there is silent and expensive: it does not crash,
it just produces a taxonomy that classifies slightly worse, and nobody notices
until lectures start landing in the wrong place.

The counts below come from the published syllabus and are not free parameters —
if a topic gains a subtopic in a future syllabus, that is a real change and the
test should be updated deliberately.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.syllabus import load_taxonomy

REPO_ROOT = Path(__file__).resolve().parents[1]
BIOLOGY_YAML = REPO_ROOT / "config" / "syllabi" / "a2-biology-9700.yaml"

#: topic number -> (title, number of subtopics, number of learning outcomes)
EXPECTED_TOPICS = {
    "12": ("Energy and respiration", 2, 21),
    "13": ("Photosynthesis", 2, 16),
    "14": ("Homeostasis", 2, 15),
    "15": ("Control and coordination", 2, 15),
    "16": ("Inheritance", 3, 18),
    "17": ("Selection and evolution", 3, 14),
    "18": ("Classification, biodiversity and conservation", 3, 18),
    "19": ("Genetic technology", 3, 17),
}


@pytest.fixture(scope="module")
def bio():
    if not BIOLOGY_YAML.exists():  # pragma: no cover - generated artefact
        pytest.skip(f"{BIOLOGY_YAML.name} not generated")
    return load_taxonomy(BIOLOGY_YAML)


def test_topics_match_the_published_syllabus(bio):
    topics = [n for n in bio.nodes]
    assert [t.syllabus_topic for t in topics] == list(EXPECTED_TOPICS)

    for topic in topics:
        title, subtopic_count, outcome_count = EXPECTED_TOPICS[topic.syllabus_topic]
        assert topic.title == title
        assert len(topic.children) == subtopic_count
        assert sum(len(s.learning_outcomes) for s in topic.children) == outcome_count


def test_every_learning_outcome_came_from_the_pdf(bio):
    """Outcomes are extracted verbatim, so all of them are verified.

    An unverified outcome means the generator invented text, which is the one
    failure mode that would quietly corrupt retrieval.
    """
    for node in bio.all_nodes():
        for outcome in node.learning_outcomes:
            assert outcome.verified, f"{node.id} carries an unverified outcome"
            assert outcome.text.strip()


def test_outcome_codes_belong_to_their_subtopic(bio):
    for topic in bio.nodes:
        for subtopic in topic.children:
            for outcome in subtopic.learning_outcomes:
                assert outcome.code.startswith(f"{subtopic.syllabus_topic}.")


def test_lessons_are_marked_unverified_and_cite_real_outcomes(bio):
    """The lesson layer is teaching structure, not syllabus content.

    Marking it verified would claim Cambridge authority for names this repo
    made up. Each lesson must, however, quote outcome text that genuinely
    exists in its parent subtopic — that text is what it gets retrieved on.
    """
    lessons = [n for n in bio.all_nodes() if n.depth == 2]
    assert lessons, "no lesson layer generated"

    for lesson in lessons:
        assert not lesson.verified, f"{lesson.id} claims to be syllabus content"
        parent = bio.require(lesson.parent_id)
        parent_text = {o.text for o in parent.learning_outcomes}
        assert lesson.learning_outcomes, f"{lesson.id} has nothing to retrieve on"
        for outcome in lesson.learning_outcomes:
            assert outcome.text in parent_text


def test_every_lesson_has_retrieval_vocabulary(bio):
    """A lesson with no keywords can never win the MIN_KEYWORD_HITS gate.

    It would be invisible to the retriever no matter what was said in the
    lecture — present in the taxonomy, unreachable in practice.
    """
    for lesson in (n for n in bio.all_nodes() if n.depth == 2):
        assert lesson.keywords, f"{lesson.id} has no keywords"


def test_chapter_twelve_can_name_six_distinct_lectures(bio):
    """The reason the lesson layer exists at all.

    Topic 12 is two syllabus subtopics but six recordings. Naming from
    subtopics would give six files two names between them.
    """
    topic12 = bio.require("bio.topic_12")
    assert len(topic12.children) == 2

    lessons = [child for sub in topic12.children for child in sub.children]
    assert len(lessons) >= 6
    assert len({lesson.title for lesson in lessons}) == len(lessons)


def test_lesson_keywords_are_terms_not_prose(bio):
    """A keyword counts only when *every* token is in the same window, in order.

    See `_WindowVocabulary.hits`. Real technical terms survive that
    happily — a teacher saying "proximal convoluted tubule" puts all three
    tokens in one window — but a descriptive gloss written by whoever filled in
    the overlay ("explain why the energy yield is much greater in aerobic")
    never fires at all. It is not an error the code can raise: the keyword
    simply never contributes, which is how it hides in a config file for a
    year.

    Four tokens is the line. Every genuine 9700 term is under it; nothing over
    it is a phrase anyone says.
    """
    prose = [
        (lesson.id, keyword)
        for lesson in bio.all_nodes()
        if lesson.depth == 2
        for keyword in lesson.keywords
        if len(keyword.split()) > 4
    ]
    assert not prose, f"keywords too long to ever fire: {prose[:5]}"


def test_every_lesson_has_keywords_its_siblings_do_not(bio):
    """Retrieval picks between siblings, so each needs its own vocabulary.

    Shared terms are fine and often unavoidable — all three lessons under
    "13.2 Investigation of limiting factors" legitimately talk about light
    intensity. What is fatal is a lesson with *nothing* its siblings lack,
    because then no transcript can ever select it over them.

    This is the generalisation of a real failure: bare "ATP" was a keyword on
    the ATP lesson, every lecture in topic 12 says it constantly, and it pulled
    two unrelated lectures onto that node.
    """
    thin = []
    for subtopic in (n for n in bio.all_nodes() if n.depth == 1):
        lessons = subtopic.children
        if len(lessons) < 2:
            continue
        for lesson in lessons:
            mine = {k.lower() for k in lesson.keywords}
            theirs = {
                k.lower() for other in lessons if other is not lesson for k in other.keywords
            }
            unique = mine - theirs
            if len(unique) < 3:
                thin.append((lesson.id, sorted(unique)))
    assert not thin, f"lessons with fewer than 3 distinctive keywords: {thin[:5]}"


def test_node_ids_are_unique(bio):
    ids = [n.id for n in bio.all_nodes()]
    assert len(ids) == len(set(ids))


def test_asr_prompt_carries_the_terms_whisper_mangles(bio):
    prompt = bio.initial_prompt(40).lower()
    for term in ("krebs", "chemiosmosis", "rubisco"):
        assert term in prompt
