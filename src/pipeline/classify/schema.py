"""The classifier's response schema.

Defined as Pydantic models and converted to a JSON Schema for the API's
structured-output mode, so the response is schema-validated at the API layer.
The pipeline never parses free text and cannot receive a malformed reply.

The important constraint here is that ``evidence`` is required and non-empty
unless the model abstains. A classification without timestamped quotes is
unauditable, and an unauditable classifier degrades a folder tree silently —
see docs/CLASSIFICATION.md.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvidenceOut(BaseModel):
    quote: str = Field(description="Verbatim quote from the transcript supporting this match")
    start_s: float = Field(description="Start time in seconds of the quote")


class PrimaryOut(BaseModel):
    node_id: str = Field(description="The id of the chosen syllabus node")
    confidence: float = Field(description="How confident this is the right node, 0.0 to 1.0")
    coverage: float = Field(description="Fraction of the lecture spent on this node, 0.0 to 1.0")
    evidence: list[EvidenceOut] = Field(
        description="At least two timestamped quotes justifying the choice"
    )


class SecondaryOut(BaseModel):
    node_id: str
    coverage: float = Field(description="Fraction of the lecture spent on this node")


class ClassificationOut(BaseModel):
    primary: PrimaryOut
    secondary: list[SecondaryOut] = Field(
        description="Other syllabus nodes with meaningful coverage; empty if none"
    )
    is_continuation: bool = Field(
        description="True if this lecture continues the previous lecture's topic"
    )
    suggested_topic_label: str = Field(
        description=(
            "Short label for the filename, e.g. 'Relative Atomic Mass'. "
            "May be more specific than the node title."
        )
    )
    reasoning: str = Field(description="Two or three sentences explaining the decision")
    abstain: bool = Field(
        description="True if none of the candidates fits; the lecture then goes to human review"
    )
    abstain_reason: str = Field(description="Why abstaining, or empty string if not abstaining")


def _strictify(schema: dict[str, Any]) -> dict[str, Any]:
    """Make a Pydantic JSON Schema acceptable to structured outputs.

    Structured outputs require ``additionalProperties: false`` on every object
    and an explicit ``required`` list. Pydantic emits neither by default.
    """
    if schema.get("type") == "object" or "properties" in schema:
        schema["additionalProperties"] = False
        if "properties" in schema:
            schema["required"] = list(schema["properties"].keys())
    for key in ("properties", "$defs"):
        for value in schema.get(key, {}).values():
            if isinstance(value, dict):
                _strictify(value)
    if isinstance(schema.get("items"), dict):
        _strictify(schema["items"])
    return schema


def response_schema() -> dict[str, Any]:
    return _strictify(ClassificationOut.model_json_schema())
