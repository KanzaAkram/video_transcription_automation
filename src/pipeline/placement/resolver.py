"""Syllabus node -> Drive folder.

Resolution is attempted in decreasing order of trust:

    1. exact       the recorded folder name matches a child exactly
    2. normalised  match after stripping "3. " and "(Chapter 4)"
    3. fuzzy       similarity above a threshold — logged for confirmation
    4. create      only if the subject opts in
    5. review      otherwise; never guess

Results are cached by node id, so after the first run everything is by folder
ID. IDs survive folder renames, which matters because people rename folders.

Placement uses a **deepest-existing** strategy by default: descend the tree as
far as folders that actually exist and stop. Your recordings sit directly in
chapter folders with no subtopic subfolders, and the system should fit that
rather than impose a deeper structure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from difflib import SequenceMatcher

from pipeline.config import PlacementSettings
from pipeline.storage import DriveFile, DrivePort
from pipeline.syllabus import Taxonomy, normalise_folder_name

log = logging.getLogger(__name__)

FUZZY_THRESHOLD = 0.82


@dataclass
class ResolutionResult:
    folder_id: str | None
    folder_path: str
    method: str  # exact | normalised | fuzzy | created | review
    confidence: float
    note: str = ""

    @property
    def resolved(self) -> bool:
        return self.folder_id is not None


class FolderResolver:
    def __init__(
        self,
        drive: DrivePort,
        taxonomy: Taxonomy,
        settings: PlacementSettings,
        *,
        recordings_root_id: str,
    ) -> None:
        self.drive = drive
        self.taxonomy = taxonomy
        self.settings = settings
        self.recordings_root_id = recordings_root_id
        #: node_id -> ResolutionResult. Populated on first resolution.
        self.cache: dict[str, ResolutionResult] = {}

    # ── Public ───────────────────────────────────────────────────────────────

    def resolve(self, node_id: str) -> ResolutionResult:
        """Find the Drive folder for a syllabus node.

        Walks from the recordings root down the node's ancestor chain, stopping
        at the deepest folder that exists (or creating, if allowed).
        """
        if node_id in self.cache:
            return self.cache[node_id]

        node = self.taxonomy.get(node_id)
        if node is None:
            result = ResolutionResult(None, "", "review", 0.0, f"unknown node {node_id!r}")
            self.cache[node_id] = result
            return result

        chain = [*self.taxonomy.ancestors(node_id), node]
        current_id = self.recordings_root_id
        path_parts: list[str] = []
        deepest_method = "exact"
        deepest_confidence = 1.0

        for step in chain:
            if not step.drive_folder_name:
                # No destination recorded for this level. Under the default
                # deepest-existing strategy that is fine — we stop here rather
                # than inventing a folder.
                if self.settings.strategy == "deepest_existing" and path_parts:
                    break
                if not self.settings.allow_create_folders:
                    result = ResolutionResult(
                        None,
                        "/".join(path_parts),
                        "review",
                        0.0,
                        f"node {step.id!r} has no drive_folder_name and folder "
                        "creation is disabled",
                    )
                    self.cache[node_id] = result
                    return result
                created = self.drive.create_folder(step.title, current_id)
                current_id = created.id
                path_parts.append(created.name)
                deepest_method, deepest_confidence = "created", 0.6
                continue

            match = self._find_folder(current_id, step.drive_folder_name)
            if match is None:
                if self.settings.allow_create_folders:
                    created = self.drive.create_folder(step.drive_folder_name, current_id)
                    current_id = created.id
                    path_parts.append(created.name)
                    deepest_method, deepest_confidence = "created", 0.6
                    continue
                if self.settings.strategy == "deepest_existing" and path_parts:
                    # Stop at the deepest folder that does exist.
                    break
                result = ResolutionResult(
                    None,
                    "/".join(path_parts),
                    "review",
                    0.0,
                    f"no folder named {step.drive_folder_name!r} under "
                    f"{self.drive.path_of(current_id)}",
                )
                self.cache[node_id] = result
                return result

            folder, method, confidence = match
            current_id = folder.id
            path_parts.append(folder.name)
            # The overall result is only as trustworthy as its weakest link.
            if confidence < deepest_confidence:
                deepest_method, deepest_confidence = method, confidence

        result = ResolutionResult(
            folder_id=current_id,
            folder_path="/".join(path_parts),
            method=deepest_method,
            confidence=deepest_confidence,
        )
        if result.method != "exact":
            log.warning(
                "folder for node %s resolved by %s (confidence %.2f) -> %s. "
                "Confirm this is correct.",
                node_id, result.method, result.confidence, result.folder_path,
            )
        self.cache[node_id] = result
        return result

    # ── Matching ─────────────────────────────────────────────────────────────

    def _find_folder(
        self, parent_id: str, wanted: str
    ) -> tuple[DriveFile, str, float] | None:
        children = self.drive.list_children(parent_id, folders_only=True)

        # 1. Exact.
        for child in children:
            if child.name == wanted:
                return child, "exact", 1.0

        # 2. Normalised — strips the ordinal prefix and "(Chapter N)" suffix,
        #    so a folder renumbered from "3." to "4." still matches. The
        #    numbers are discarded rather than compared, deliberately: they
        #    disagree with each other in your tree (see ADR-013).
        wanted_norm = normalise_folder_name(wanted)
        normalised = [c for c in children if normalise_folder_name(c.name) == wanted_norm]
        if len(normalised) == 1:
            return normalised[0], "normalised", 0.95
        if len(normalised) > 1:
            log.warning(
                "ambiguous normalised match for %r under %s: %s",
                wanted, parent_id, [c.name for c in normalised],
            )
            return None

        # 3. Fuzzy — last resort, always logged.
        best: tuple[DriveFile, float] | None = None
        for child in children:
            score = SequenceMatcher(
                None, wanted_norm, normalise_folder_name(child.name)
            ).ratio()
            if best is None or score > best[1]:
                best = (child, score)
        if best and best[1] >= FUZZY_THRESHOLD:
            return best[0], "fuzzy", best[1]
        return None

    # ── Bootstrapping ────────────────────────────────────────────────────────

    def resolve_all(self) -> dict[str, ResolutionResult]:
        """Resolve every placeable node. Used by bootstrap_drive_map.py.

        Run this with --dry-run before going live: it is the cheapest way to
        confirm the taxonomy's folder names actually match what is in Drive.
        """
        return {node.id: self.resolve(node.id) for node in self.taxonomy.placeable_nodes()}
