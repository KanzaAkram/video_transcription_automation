"""Map an OpenAI classifier result onto the small preview JSON."""

from __future__ import annotations

from pipeline.config import NamingSettings
from pipeline.models import Classification, Recording
from pipeline.naming import build_name
from pipeline.placement import hierarchy_labels
from pipeline.syllabus import Taxonomy

from pipeline.m4a.routing import SubjectHint


def classification_fields(
    classification: Classification,
    taxonomy: Taxonomy,
    hint: SubjectHint,
) -> dict[str, str]:
    if classification.primary is not None:
        labels = hierarchy_labels(
            taxonomy,
            classification.primary.node_id,
            subject_name=f"{hint.class_name} {hint.subject}",
        )
        return {
            "class": hint.class_name,
            "subject": hint.subject,
            "chapter": labels.chapter,
            "topic": labels.topic,
            "subtopic": labels.subtopic,
        }
    chapter = ""
    if classification.chapter_node_id:
        node = taxonomy.get(classification.chapter_node_id)
        if node is not None:
            chapter = node.label
    topic_nodes = [
        taxonomy.require(node_id)
        for node_id in classification.topic_node_ids
        if taxonomy.get(node_id) is not None
    ]
    topic = topic_nodes[0].label if topic_nodes else (classification.suggested_topic_label or "")
    if len(topic_nodes) > 1:
        subtopic = " + ".join(node.label for node in topic_nodes[1:])
    else:
        subtopic = classification.focus or ""
        if subtopic and subtopic == topic:
            subtopic = ""
    return {
        "class": hint.class_name,
        "subject": hint.subject,
        "chapter": chapter,
        "topic": topic,
        "subtopic": subtopic,
    }


def generated_filename(
    recording: Recording,
    classification: Classification,
    taxonomy: Taxonomy,
    naming: NamingSettings,
    *,
    serial: int = 1,
    existing_names: set[str] | None = None,
) -> str:
    return build_name(
        recording,
        classification,
        taxonomy,
        naming,
        existing_names=existing_names,
        serial=serial,
    ).filename
