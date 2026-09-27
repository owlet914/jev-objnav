"""Validate a navigation snapshot and select one planner-supplied candidate."""

from __future__ import annotations

import math
import json
import os
import copy
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Union

from .api_diagnostics import encode_request, http_error_evidence, redact_text
from .region_graph import (
    RegionGraphError,
    instructions as region_graph_instructions,
    project_region_graph,
)


STATE_PROFILES = ("candidates", "spatial", "full", "region_graph")
CURRENT_SCHEMA_VERSION = "2.0"
LEGACY_SCHEMA_VERSIONS = {"1.0"}


class SnapshotError(ValueError):
    """A non-retryable input-contract failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class ProviderError(RuntimeError):
    """A classified provider or provider-response failure."""

    def __init__(self, code: str, message: str, *, retryable: bool = True,
                 diagnostics: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.diagnostics = diagnostics or {}


@dataclass(frozen=True)
class PreparedDecision:
    state: dict[str, Any]
    options: dict[str, str]
    model_request: dict[str, Any]
    eligible_ids: tuple[str, ...]
    option_to_candidate: dict[str, str]


class DecisionProvider(Protocol):
    def choose(self, state: dict[str, Any], options: dict[str, str]) -> tuple[str, float, dict[str, float]]:
        """Return (candidate ID, confidence, option probabilities)."""


class OpenRouterJevProvider:
    """Call Jev through OpenRouter's Decisions API, not Chat Completions."""

    ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
    MODEL = "typesafe/jev-1.13"

    def __init__(self, timeout_s: float = 4.0, model: str = MODEL, api_key: Optional[str] = None):
        self.timeout_s = timeout_s
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")

    @staticmethod
    def instructions(state: dict[str, Any]) -> str:
        instructions = (
            "Find the object named in state.target_object. Choose one reachable candidate ID. "
            "Read state.decision_summary first, then inspect the full raw state when a compact "
            "field needs provenance or geometric detail. "
            "Approach an object only when its detection evidence supports the target. "
            "When reliable target evidence is absent, choose a safe reachable frontier that can "
            "acquire new information or an in-place observation rotation; weak target evidence is "
            "expected during exploration and is not a reason to wait. Compare original-policy "
            "stage conditions, semantic evidence, frontier state, path-graph costs, information-"
            "gain provenance and path cost. Accumulated object-map confidence is not the same as "
            "current-frame confirmation. Avoid selecting the same entity again when repeated "
            "selections produced zero or negligible progress and perception did not improve; "
            "prefer a different safe frontier or the opposite observation direction. Observation "
            "rotations are evidence-gathering actions, not zero-cost defaults. Never invent a "
            "position or a candidate."
        )
        if "local_preview" in state.get("map", {}) or "local_grid" in state.get("map", {}):
            instructions += (
                " Use the local map only as the explicitly labelled lossy preview; use exact "
                "frontier/path evidence for decisions. Do not treat unknown cells as free."
            )
        if "perception" in state:
            instructions += (
                " Check validity, observation_id, backend and fallback status before treating any "
                "perception score as measured evidence. Missing evidence is not negative evidence."
            )
        if "room_prior" in state or "target_subcategories" in state:
            instructions += (
                " Room priors, related objects and target subcategories are textual context, not "
                "observed target positions. Prefer a well-supported target detection when present."
            )
        return instructions

    def request_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send the exact request; retain bounded/redacted upstream HTTP evidence."""
        if not self.api_key:
            raise ProviderError(
                "PROVIDER_CONFIGURATION_ERROR", "OPENROUTER_API_KEY is not set", retryable=False
            )
        request = urllib.request.Request(
            self.ENDPOINT,
            data=encode_request(payload),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = json.load(response)
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or exc.code >= 500
            diagnostics = http_error_evidence(exc, payload, self.api_key)
            error_code = (
                "INPUT_CAPACITY_EXCEEDED"
                if diagnostics.get("classification") == "context_length_reported_by_upstream"
                else "PROVIDER_HTTP_ERROR"
            )
            raise ProviderError(
                error_code, f"HTTP {exc.code}", retryable=retryable,
                diagnostics=diagnostics,
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderError("PROVIDER_TEMPORARY_FAILURE", type(exc).__name__) from exc
        except (ValueError, UnicodeError) as exc:
            raise ProviderError("INVALID_PROVIDER_RESPONSE", type(exc).__name__) from exc
        if not isinstance(body, dict):
            raise ProviderError("INVALID_PROVIDER_RESPONSE", "response must be an object")
        return body

    def choose_request(self, payload: dict[str, Any]) -> tuple[str, float, dict[str, float]]:
        selected, confidence, probabilities, _ = self.choose_request_with_metadata(payload)
        return selected, confidence, probabilities

    def choose_request_with_metadata(
        self, payload: dict[str, Any]
    ) -> tuple[str, float, dict[str, float], dict[str, Any]]:
        body = self.request_json(payload)
        def redact_response(value):
            if isinstance(value,dict):
                return {k: ("[REDACTED]" if k.lower() in ("authorization", "api_key", "openrouter_api_key") else redact_response(v)) for k,v in value.items()}
            if isinstance(value,list): return [redact_response(v) for v in value]
            if isinstance(value,str): return redact_text(value,self.api_key,max(len(value),32))
            return value
        try:
            answer = body["answers"]["next_goal"]
            metadata = {
                "actual_model": body.get("model"),
                "generation_id": body.get("id", body.get("generation_id")),
                "usage": copy.deepcopy(body.get("usage")),
                "raw_response": redact_response(body),
            }
            return answer["choice"], float(answer["confidence"]), dict(answer["probabilities"]), metadata
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError("INVALID_PROVIDER_RESPONSE", type(exc).__name__) from exc

    def choose(self, state: dict[str, Any], options: dict[str, str]) -> tuple[str, float, dict[str, float]]:
        return self.choose_request(build_model_request(state, options, self.model))


class DemoProvider:
    """Offline smoke test only; this is a heuristic, not a Jev simulation."""

    def choose(self, state: dict[str, Any], options: dict[str, str]) -> tuple[str, float, dict[str, float]]:
        candidates = [item for item in state["candidates"] if item.get("id") in options]
        if not candidates:
            return "hold", 1.0, {"hold": 1.0}
        best = max(
            candidates,
            key=lambda item: (
                float(item.get("semantic_score") or 0)
                + float(item.get("information_gain") or 0)
                - 0.05 * float(item["path"]["distance_m"])
            ),
        )
        return best["id"], 1.0, {best["id"]: 1.0}


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _valid_pose(pose: Any) -> bool:
    return isinstance(pose, dict) and all(_finite_number(pose.get(key)) for key in ("x", "y", "yaw"))


def _require_full_context_structure(snapshot: dict[str, Any]) -> None:
    for key in ("original_policy", "perception", "navigation_history"):
        if not isinstance(snapshot.get(key), dict):
            raise SnapshotError("INCOMPLETE_FULL_CONTEXT", f"{key} is required for complete context")
    if not isinstance(snapshot.get("search_branches"), list):
        raise SnapshotError("INCOMPLETE_FULL_CONTEXT", "search_branches is required")
    if not isinstance(snapshot.get("candidate_evidence"), list):
        raise SnapshotError("INCOMPLETE_FULL_CONTEXT", "candidate_evidence is required")
    if snapshot.get("map", {}).get("observation_id") != snapshot.get("observation_id"):
        raise SnapshotError("INCOMPLETE_FULL_CONTEXT", "map observation_id does not match snapshot")

    for item in snapshot.get("objects", []):
        if not isinstance(item, dict) or not isinstance(item.get("normal_path_search"), dict):
            raise SnapshotError("INCOMPLETE_FULL_CONTEXT", "every object requires normal_path_search")
    frontier_ids: set[str] = set()
    frontier_ids_by_state = {"active": set(), "dormant": set()}
    for item in snapshot.get("frontiers", []):
        features = item.get("semantic_features") if isinstance(item, dict) else None
        neighborhood = features.get("raw_neighborhood") if isinstance(features, dict) else None
        if (
            not isinstance(neighborhood, list)
            or len(neighborhood) != 25
            or "planning_status" not in item
            or not isinstance(item.get("path"), dict)
        ):
            raise SnapshotError(
                "INCOMPLETE_FULL_CONTEXT", "every frontier requires its 5x5 raw neighborhood and path"
            )
        frontier_id = item.get("id")
        if not isinstance(frontier_id, str) or not frontier_id or frontier_id in frontier_ids:
            raise SnapshotError("INCOMPLETE_FULL_CONTEXT", "frontier IDs must be unique and stable")
        frontier_ids.add(frontier_id)
        if item.get("state") in frontier_ids_by_state:
            frontier_ids_by_state[item["state"]].add(frontier_id)
    evidence_ids = {
        item.get("target_entity_id")
        for item in snapshot["candidate_evidence"]
        if isinstance(item, dict)
    }
    if evidence_ids != frontier_ids or len(snapshot["candidate_evidence"]) != len(frontier_ids):
        raise SnapshotError("INCOMPLETE_FULL_CONTEXT", "frontier candidate evidence is incomplete")

    branches = {
        item.get("name"): item
        for item in snapshot["search_branches"]
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    required_branches = {
        "high_confidence_object_normal", "current_over_depth_object",
        "suspicious_object_normal", "active_frontier_policy", "dormant_frontier_policy",
        "original_object_extreme", "relaxed_all_cells_extreme", "cached_over_depth_extreme",
    }
    if not required_branches.issubset(branches):
        raise SnapshotError("INCOMPLETE_FULL_CONTEXT", "required original-policy branches are missing")
    relaxed_count = sum(
        item.get("selection", {}).get("relaxed_eligible") is True
        for item in snapshot["objects"] if isinstance(item, dict)
    )
    mapped_count = sum(
        isinstance(item, dict) and isinstance(item.get("selection"), dict)
        for item in snapshot["objects"]
    )
    expected_relaxed = relaxed_count if relaxed_count else (1 if mapped_count else 0)
    for name, expected in (
        ("original_object_extreme", relaxed_count),
        ("relaxed_all_cells_extreme", expected_relaxed),
    ):
        rows = branches[name].get("entity_results")
        if not isinstance(rows, list) or len(rows) != expected:
            raise SnapshotError("INCOMPLETE_FULL_CONTEXT", f"{name} evidence count is incomplete")
    if not isinstance(branches["cached_over_depth_extreme"].get("entity_results"), list):
        raise SnapshotError("INCOMPLETE_FULL_CONTEXT", "cached extreme evidence is invalid")

    policy = snapshot["original_policy"]
    for name in ("active_frontier_path_graph", "dormant_frontier_path_graph"):
        graph = policy.get(name)
        if not isinstance(graph, dict) or not isinstance(graph.get("node_ids"), list):
            raise SnapshotError("INCOMPLETE_FULL_CONTEXT", f"{name} is required")
        matrix = graph.get("cost_matrix")
        size = len(graph["node_ids"])
        if not isinstance(matrix, list) or len(matrix) != size or any(
            not isinstance(row, list) or len(row) != size for row in matrix
        ):
            raise SnapshotError("INCOMPLETE_FULL_CONTEXT", f"{name} matrix dimensions are invalid")
        expected_state = "active" if name.startswith("active") else "dormant"
        if set(graph["node_ids"]) != {"robot"} | frontier_ids_by_state[expected_state]:
            raise SnapshotError("INCOMPLETE_FULL_CONTEXT", f"{name} node IDs do not match frontiers")
    perception = snapshot["perception"]
    if not isinstance(perception.get("object_detection"), dict) or not isinstance(
        perception.get("image_text_match"), dict
    ):
        raise SnapshotError("INCOMPLETE_FULL_CONTEXT", "perception provenance is incomplete")


def _validate_coverage(snapshot: dict[str, Any]) -> None:
    coverage = snapshot.get("coverage")
    if not isinstance(coverage, dict):
        raise SnapshotError("SCHEMA_ERROR", "schema 2.0 requires coverage")
    counts = coverage.get("entity_counts")
    if not isinstance(counts, dict):
        raise SnapshotError("SCHEMA_ERROR", "coverage.entity_counts is required")
    required = ("objects", "active_frontiers", "dormant_frontiers", "candidate_evidence", "candidates")
    for name in required:
        row = counts.get(name)
        if not isinstance(row, dict):
            raise SnapshotError("SCHEMA_ERROR", f"coverage.entity_counts.{name} is required")
        for field in ("total", "exported", "omitted"):
            if type(row.get(field)) is not int or row[field] < 0:
                raise SnapshotError("SCHEMA_ERROR", f"invalid {name}.{field}")
        if row["total"] != row["exported"] + row["omitted"]:
            raise SnapshotError("SCHEMA_ERROR", f"inconsistent coverage count for {name}")
    actual_counts = {
        "objects": len(snapshot.get("objects", [])),
        "active_frontiers": sum(
            item.get("state") == "active"
            for item in snapshot.get("frontiers", []) if isinstance(item, dict)
        ),
        "dormant_frontiers": sum(
            item.get("state") == "dormant"
            for item in snapshot.get("frontiers", []) if isinstance(item, dict)
        ),
        "candidate_evidence": len(snapshot.get("candidate_evidence", [])),
        "candidates": len(snapshot.get("candidates", [])),
    }
    for name, actual in actual_counts.items():
        if counts[name]["exported"] != actual:
            raise SnapshotError("SCHEMA_ERROR", f"coverage exported count does not match {name} array")
    if coverage.get("high_level_evidence_complete") is True:
        if any(counts[name]["omitted"] != 0 for name in required):
            raise SnapshotError("INCOMPLETE_FULL_CONTEXT", "complete context cannot omit entities")
        feature_status = coverage.get("required_feature_status")
        required_features = (
            "entity_enumeration", "normal_path_searches", "extreme_search_evidence",
            "frontier_path_graphs", "raw_decision_neighborhoods",
        )
        if not isinstance(feature_status, dict) or any(
            not str(feature_status.get(name, "")).startswith("complete")
            for name in required_features
        ):
            raise SnapshotError("INCOMPLETE_FULL_CONTEXT", "required high-level features are not complete")
        _require_full_context_structure(snapshot)


def validate_snapshot(snapshot: Any) -> None:
    """Check the boundary contract before any model call or goal dispatch."""
    if not isinstance(snapshot, dict):
        raise SnapshotError("SCHEMA_ERROR", "snapshot must be an object")
    schema = snapshot.get("schema_version")
    if schema not in LEGACY_SCHEMA_VERSIONS | {CURRENT_SCHEMA_VERSION}:
        raise SnapshotError("SCHEMA_ERROR", "unsupported schema_version")
    if not isinstance(snapshot.get("target_object"), str) or not snapshot["target_object"].strip():
        raise SnapshotError("SCHEMA_ERROR", "target_object must be a nonempty string")
    if not isinstance(snapshot.get("episode_id"), str) or not snapshot["episode_id"]:
        raise SnapshotError("SCHEMA_ERROR", "episode_id is required")
    if type(snapshot.get("timestamp_ms")) is not int or snapshot["timestamp_ms"] < 0:
        raise SnapshotError("SCHEMA_ERROR", "timestamp_ms must be a nonnegative integer")
    if schema == CURRENT_SCHEMA_VERSION:
        if not isinstance(snapshot.get("request_id"), str) or not snapshot["request_id"]:
            raise SnapshotError("SCHEMA_ERROR", "request_id is required")
        if type(snapshot.get("observation_id")) is not int or snapshot["observation_id"] < 0:
            raise SnapshotError("SCHEMA_ERROR", "observation_id must be a nonnegative integer")
        _validate_coverage(snapshot)
    if not isinstance(snapshot.get("map"), dict) or not isinstance(snapshot["map"].get("frame_id"), str) or not snapshot["map"]["frame_id"]:
        raise SnapshotError("SCHEMA_ERROR", "map.frame_id is required")
    if type(snapshot["map"].get("revision")) is not int or snapshot["map"]["revision"] < 0:
        raise SnapshotError("SCHEMA_ERROR", "map.revision must be a nonnegative integer")
    if not _valid_pose(snapshot.get("robot_pose")):
        raise SnapshotError("SCHEMA_ERROR", "robot_pose requires finite x, y and yaw")
    for key in ("rooms", "objects", "semantic_matches", "frontiers", "candidates"):
        if not isinstance(snapshot.get(key), list):
            raise SnapshotError("SCHEMA_ERROR", f"{key} must be a list")
    observation_actions = snapshot.get("observation_actions", [])
    if not isinstance(observation_actions, list):
        raise SnapshotError("SCHEMA_ERROR", "observation_actions must be a list")
    object_ids = {item.get("id") for item in snapshot["objects"] if isinstance(item, dict)}
    frontier_ids = {item.get("id") for item in snapshot["frontiers"] if isinstance(item, dict)}
    observation_ids = {
        item.get("id") for item in observation_actions if isinstance(item, dict)
    }
    seen: set[str] = set()
    for item in snapshot["candidates"]:
        if not isinstance(item, dict):
            raise SnapshotError("SCHEMA_ERROR", "each candidate must be an object")
        candidate_id = item.get("id")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id == "hold" or candidate_id in seen:
            raise SnapshotError("SCHEMA_ERROR", "candidate IDs must be unique nonempty strings other than 'hold'")
        seen.add(candidate_id)
        if item.get("kind") not in ("explore_frontier", "approach_object", "observe_rotation"):
            raise SnapshotError("SCHEMA_ERROR", f"invalid kind for candidate {candidate_id}")
        if not isinstance(item.get("target_id"), str) or not item["target_id"]:
            raise SnapshotError("SCHEMA_ERROR", f"target_id is required for candidate {candidate_id}")
        valid_targets = (
            frontier_ids if item["kind"] == "explore_frontier" else
            observation_ids if item["kind"] == "observe_rotation" else object_ids
        )
        if item["target_id"] not in valid_targets:
            raise SnapshotError("SCHEMA_ERROR", f"target_id does not reference an available entity for candidate {candidate_id}")
        if not _valid_pose(item.get("goal_pose")):
            raise SnapshotError("SCHEMA_ERROR", f"invalid goal_pose for candidate {candidate_id}")
        path = item.get("path")
        if not isinstance(path, dict) or type(path.get("reachable")) is not bool or type(path.get("collision_free")) is not bool:
            raise SnapshotError("SCHEMA_ERROR", f"path flags must be booleans for candidate {candidate_id}")
        if not _finite_number(path.get("distance_m")) or path["distance_m"] < 0:
            raise SnapshotError("SCHEMA_ERROR", f"invalid path.distance_m for candidate {candidate_id}")
        for score in ("semantic_score", "information_gain"):
            value = item.get(score)
            if value is not None and not _finite_number(value):
                raise SnapshotError("SCHEMA_ERROR", f"{score} must be finite or null for candidate {candidate_id}")


def build_model_state(snapshot: dict[str, Any], profile: str) -> dict[str, Any]:
    """Choose observed fields for an ablation; never add dataset ground truth."""
    if profile not in STATE_PROFILES:
        raise SnapshotError("CONFIGURATION_ERROR", f"unknown state profile: {profile}")
    if profile == "full":
        if snapshot.get("schema_version") == CURRENT_SCHEMA_VERSION:
            return snapshot
        state = copy.deepcopy(snapshot)
        state["coverage"] = {
            "scope": "legacy schema",
            "high_level_evidence_complete": False,
            "raw_map_complete": False,
            "limitations": ["legacy schema lacks the information-parity coverage contract"],
        }
        return state

    identity_keys = ("schema_version", "timestamp_ms", "request_id", "episode_id", "observation_id", "target_object", "robot_pose", "candidates")
    state = {key: snapshot[key] for key in identity_keys if key in snapshot}
    state["coverage"] = {
        "scope": f"ablation:{profile}",
        "high_level_evidence_complete": False,
        "raw_map_complete": False,
        "limitations": ["profile intentionally removes decision evidence"],
    }
    state["observation_actions"] = snapshot.get("observation_actions", [])
    state["map"] = {
        key: value for key, value in snapshot["map"].items()
        if key not in ("local_grid", "local_preview", "evidence_encoding")
    }
    state["rooms"] = []
    state["objects"] = [
        {key: obj[key] for key in ("id", "position") if key in obj}
        for obj in snapshot["objects"]
    ]
    state["frontiers"] = [
        {key: frontier[key] for key in ("id", "position") if key in frontier}
        for frontier in snapshot["frontiers"]
    ]
    state["semantic_matches"] = []
    if profile == "spatial":
        for preview_key in ("local_preview", "local_grid"):
            if preview_key in snapshot["map"]:
                state["map"][preview_key] = snapshot["map"][preview_key]
        for projected, original in zip(state["objects"], snapshot["objects"]):
            for key in ("target_confidence", "target_observations", "score_available", "confidence_tier"):
                if key in original:
                    projected[key] = original[key]
        for projected, original in zip(state["frontiers"], snapshot["frontiers"]):
            for key in ("cluster_cells", "information_gain", "semantic_value", "semantic_confidence"):
                if key in original:
                    projected[key] = original[key]
        if "current_view" in snapshot:
            state["current_view"] = snapshot["current_view"]
        state["semantic_matches"] = snapshot["semantic_matches"]
    return state


def build_model_request(
    state: dict[str, Any], options: dict[str, str], model: str = OpenRouterJevProvider.MODEL
) -> dict[str, Any]:
    """Pure request builder used by both live sending and trace capture."""
    return {
        "model": model,
        "state": state,
        "questions": {
            "next_goal": {
                "type": "choice",
                "instructions": region_graph_instructions()
                if state.get("schema_version") == "jev-region-graph/1.1"
                else OpenRouterJevProvider.instructions(state),
                "criteria": options,
            }
        },
    }


def _class_evidence(obj: dict[str, Any]) -> dict[str, Any]:
    scores = [item for item in obj.get("class_scores", []) if isinstance(item, dict)]
    ranked = []
    for item in scores:
        confidence = item.get("fused_confidence", item.get("confidence"))
        if _finite_number(confidence):
            ranked.append((float(confidence), item.get("label")))
    ranked.sort(reverse=True)
    return {
        "best_label": ranked[0][1] if ranked else None,
        "best_confidence": ranked[0][0] if ranked else None,
        "runner_up_confidence": ranked[1][0] if len(ranked) > 1 else None,
        "confidence_margin": ranked[0][0] - ranked[1][0] if len(ranked) > 1 else None,
        "computed_best_label": obj.get("computed_best_label"),
        "stored_best_label": obj.get("stored_best_label"),
    }


def _frontier_evidence(frontier: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    features = frontier.get("semantic_features")
    features = features if isinstance(features, dict) else {}
    semantic_value = frontier.get("semantic_value", candidate.get("semantic_score"))
    if semantic_value is None:
        semantic_value = features.get(
            "hybrid_policy_5x5_center_then_max_excluding_occupied_and_inflated_neighbors"
        )
    semantic_fusion_weight = frontier.get("semantic_confidence")
    if semantic_fusion_weight is None:
        semantic_fusion_weight = features.get("value_map_fov_fusion_weight_at_hybrid_max")
    return {
        "state": frontier.get("state"),
        "planning_status": frontier.get("planning_status"),
        "cluster_cells": frontier.get("cluster_cells"),
        "semantic_value": semantic_value,
        "semantic_fusion_weight": semantic_fusion_weight,
        "information_gain": candidate.get("information_gain", frontier.get("information_gain")),
        "information_gain_status": frontier.get("information_gain_status"),
    }


def build_decision_summary(
    state: dict[str, Any], eligible: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build a compact, auditable view without deleting any raw state evidence."""
    objects = {
        item.get("id"): item for item in state.get("objects", [])
        if isinstance(item, dict) and item.get("id") is not None
    }
    frontiers = {
        item.get("id"): item for item in state.get("frontiers", [])
        if isinstance(item, dict) and item.get("id") is not None
    }
    observations = {
        item.get("id"): item for item in state.get("observation_actions", [])
        if isinstance(item, dict) and item.get("id") is not None
    }
    perception = state.get("perception")
    perception = perception if isinstance(perception, dict) else {}
    image_match = perception.get("image_text_match", state.get("current_view", {}))
    image_match = image_match if isinstance(image_match, dict) else {}
    detection = perception.get("object_detection", {})
    detection = detection if isinstance(detection, dict) else {}
    recent_observations = [
        item for item in perception.get("recent_observations", []) if isinstance(item, dict)
    ]
    recent_scores = [
        float(item["image_text_match_raw_score"])
        for item in recent_observations
        if _finite_number(item.get("image_text_match_raw_score"))
    ]
    zero_target_match_streak = 0
    for recent in reversed(recent_observations):
        if recent.get("target_match_count") != 0:
            break
        zero_target_match_streak += 1
    score_delta = recent_scores[-1] - recent_scores[0] if len(recent_scores) >= 2 else None
    history = state.get("navigation_history")
    history = history if isinstance(history, dict) else {}
    execution = history.get("last_goal_execution")
    execution = execution if isinstance(execution, dict) else {}
    current_observation = state.get("observation_id")
    last_entity = history.get("last_entity_id")
    repeat_count = history.get("consecutive_same_entity_selections", 0)
    compact_candidates = []

    for candidate in eligible:
        target_id = candidate.get("target_id")
        common = {
            "id": candidate.get("id"),
            "kind": candidate.get("kind"),
            "target_id": target_id,
            "distance_m": candidate.get("path", {}).get("distance_m"),
            "source_stage": candidate.get("source_stage"),
            "source_priority": candidate.get("source_priority"),
            "same_entity_as_last_choice": target_id == last_entity,
            "consecutive_same_entity_selections": repeat_count if target_id == last_entity else 0,
            "last_step_progress_m": execution.get("last_step_progress_m") if target_id == last_entity else None,
            "cumulative_progress_m": execution.get("cumulative_progress_m") if target_id == last_entity else None,
        }
        if candidate.get("kind") == "approach_object":
            obj = objects.get(target_id, {})
            provenance = obj.get("provenance")
            provenance = provenance if isinstance(provenance, dict) else {}
            common["object_evidence"] = {
                "target_confidence": obj.get("target_confidence", obj.get("confidence")),
                "target_observations": obj.get("target_observations"),
                "target_point_count": obj.get("target_point_count"),
                "updated_in_current_observation": provenance.get("observation_id") == current_observation,
                "provenance_observation_id": provenance.get("observation_id"),
                "historical_map_evidence": provenance.get("historical_map_evidence"),
                "selection": obj.get("selection"),
                "class_evidence": _class_evidence(obj),
            }
        elif candidate.get("kind") == "explore_frontier":
            common["frontier_evidence"] = _frontier_evidence(
                frontiers.get(target_id, {}), candidate
            )
        else:
            action = observations.get(target_id, {})
            common["observation_evidence"] = {
                "direction": action.get("direction"),
                "rotation_rad": action.get("rotation_rad"),
                "expected_yaw": action.get("expected_yaw"),
                "information_gain": candidate.get("information_gain"),
                "information_gain_status": action.get(
                    "information_gain_status", candidate.get("information_gain_status")
                ),
            }
        compact_candidates.append(common)

    return {
        "purpose": "decision-focused index into the complete raw state",
        "target_object": state.get("target_object"),
        "observation_id": current_observation,
        "perception": {
            "image_text_match": {
                "raw_score": image_match.get("raw_score", image_match.get("image_text_match_score")),
                "score_type": image_match.get("score_type"),
                "valid": image_match.get("valid"),
                "available": image_match.get("available"),
                "backend": image_match.get("backend"),
                "fallback": image_match.get("fallback"),
                "synchronized_with_map": image_match.get("synchronized_with_map"),
            },
            "object_detection": {
                "available": detection.get("available"),
                "fallback": detection.get("fallback"),
                "target_service_available": detection.get("target_service_available"),
                "target_match_count": detection.get("target_match_count"),
                "valid_mask_count": detection.get("valid_mask_count"),
                "successful_map_ingestion_count": detection.get("successful_map_ingestion_count"),
                "synchronized_with_map": detection.get("synchronized_with_map"),
            },
            "reliable_target_evidence_available": perception.get(
                "reliable_target_evidence_available"
            ),
            "recent_observations": recent_observations,
            "recent_evidence_trend": {
                "valid_score_sample_count": len(recent_scores),
                "first_raw_score": recent_scores[0] if recent_scores else None,
                "latest_raw_score": recent_scores[-1] if recent_scores else None,
                "minimum_raw_score": min(recent_scores) if recent_scores else None,
                "maximum_raw_score": max(recent_scores) if recent_scores else None,
                "latest_minus_first_raw_score": score_delta,
                "zero_target_match_streak": zero_target_match_streak,
            },
            "interpretation": (
                "raw image-text score is not a calibrated probability; current-frame detector "
                "support and accumulated object-map evidence are separate signals"
            ),
        },
        "navigation": {
            "last_candidate_id": history.get("last_candidate_id"),
            "last_entity_id": last_entity,
            "consecutive_same_entity_selections": repeat_count,
            "last_goal_execution": {
                "last_step_progress_m": execution.get("last_step_progress_m"),
                "cumulative_progress_m": execution.get("cumulative_progress_m"),
                "initial_distance_m": execution.get("initial_distance_m"),
                "current_distance_m": execution.get("current_distance_m"),
            },
            "collision_count": history.get("collision_count"),
            "latest_collision": history.get("latest_collision"),
        },
        "eligible_candidate_count": len(compact_candidates),
        "eligible_candidates": compact_candidates,
        "limitations": [
            "summary is derived only from runtime evidence already present in the raw state",
            "summary does not add dataset ground truth or remove any eligible candidate",
        ],
    }


def prepare_decision(snapshot: dict[str, Any], state_profile: str = "full") -> PreparedDecision:
    if state_profile == "region_graph":
        try:
            projection = project_region_graph(snapshot)
        except RegionGraphError as exc:
            raise SnapshotError(exc.code, str(exc)) from exc
        source_candidates = snapshot.get("candidates")
        if not isinstance(source_candidates, list):
            raise SnapshotError("REGION_GRAPH_SCHEMA_ERROR", "local candidate registry is missing")
        registry = {}
        for item in source_candidates:
            if not isinstance(item, dict) or not isinstance(item.get('id'), str) or not item['id'] or item['id'] in registry:
                raise SnapshotError('REGION_GRAPH_SCHEMA_ERROR', 'candidate ids must be unique')
            path = item.get('path', {})
            if (not _valid_pose(item.get('goal_pose')) or not isinstance(path,dict) or
                path.get('reachable') is not True or path.get('collision_free') is not True or
                not _finite_number(path.get('distance_m')) or path['distance_m']<0):
                raise SnapshotError('REGION_GRAPH_SCHEMA_ERROR','invalid local candidate safety or geometry')
            registry[item['id']] = item
        missing = set(projection.option_to_candidate.values()) - set(registry)
        if missing:
            raise SnapshotError('REGION_GRAPH_SCHEMA_ERROR', f'missing local candidates: {sorted(missing)}')
        if set(registry) != set(projection.option_to_candidate.values()):
            raise SnapshotError('REGION_GRAPH_SCHEMA_ERROR','local candidate coverage mismatch')
        for option in projection.state['options']:
            item = registry[option['candidate_id']]
            expected_kind = ('approach_object' if option['action_type']=='approach_object' else
                             'observe_rotation' if option.get('exploration_kind')=='observe_rotation' else 'explore_frontier')
            if (item.get('kind')!=expected_kind or item.get('target_id')!=option['target_id'] or
                abs(item['path']['distance_m']-option['distance_m'])>0.000501):
                raise SnapshotError('REGION_GRAPH_SCHEMA_ERROR','option/local candidate mismatch')
        model_request = build_model_request(projection.state, projection.criteria)
        return PreparedDecision(
            state=projection.state,
            options=projection.criteria,
            model_request=model_request,
            eligible_ids=tuple(projection.criteria),
            option_to_candidate=projection.option_to_candidate,
        )
    validate_snapshot(snapshot)
    if state_profile == "full" and snapshot.get("schema_version") == CURRENT_SCHEMA_VERSION:
        if snapshot["coverage"].get("high_level_evidence_complete") is not True:
            raise SnapshotError("INCOMPLETE_FULL_CONTEXT", "full profile requires complete high-level evidence")
    eligible = [
        item for item in snapshot["candidates"]
        if item["path"]["reachable"] and item["path"]["collision_free"]
    ]
    if not eligible:
        raise SnapshotError("NO_EXECUTABLE_CANDIDATE", "no reachable collision-free candidate")
    state = dict(build_model_state(snapshot, state_profile))
    state["decision_summary"] = build_decision_summary(state, eligible)
    objects_by_id = {item.get("id"): item for item in state["objects"] if isinstance(item, dict)}
    frontiers_by_id = {item.get("id"): item for item in state["frontiers"] if isinstance(item, dict)}
    observations_by_id = {item.get("id"): item for item in state.get("observation_actions", []) if isinstance(item, dict)}
    summary_candidates = {
        item.get("id"): item
        for item in state["decision_summary"]["eligible_candidates"]
        if isinstance(item, dict)
    }
    perception_summary = state["decision_summary"]["perception"]
    image_summary = perception_summary["image_text_match"]
    detection_summary = perception_summary["object_detection"]
    trend_summary = perception_summary["recent_evidence_trend"]
    options: dict[str, str] = {}
    for item in eligible:
        targets = objects_by_id if item["kind"] == "approach_object" else observations_by_id if item["kind"] == "observe_rotation" else frontiers_by_id
        target = targets[item["target_id"]]
        compact = summary_candidates.get(item["id"], {})
        repetition = (
            f"same_as_last={compact.get('same_entity_as_last_choice')}; "
            f"repeat_count={compact.get('consecutive_same_entity_selections')}; "
            f"last_step_progress_m={compact.get('last_step_progress_m')}; "
            f"cumulative_progress_m={compact.get('cumulative_progress_m')}"
        )
        current_evidence = (
            f"current_view_raw_score={image_summary.get('raw_score')}; "
            f"current_view_score_type={image_summary.get('score_type')}; "
            f"current_view_valid={image_summary.get('valid')}; "
            f"detector_target_matches={detection_summary.get('target_match_count')}; "
            f"detector_valid_masks={detection_summary.get('valid_mask_count')}; "
            f"reliable_current_target_evidence={perception_summary.get('reliable_target_evidence_available')}; "
            f"recent_score_delta={trend_summary.get('latest_minus_first_raw_score')}; "
            f"recent_score_min={trend_summary.get('minimum_raw_score')}; "
            f"recent_score_max={trend_summary.get('maximum_raw_score')}; "
            f"zero_target_match_streak={trend_summary.get('zero_target_match_streak')}"
        )
        if item["kind"] == "approach_object":
            object_evidence = compact.get("object_evidence", {})
            class_evidence = object_evidence.get("class_evidence", {})
            target_confidence = object_evidence.get("target_confidence")
            target_confidence = "unknown" if target_confidence is None else target_confidence
            target_observations = object_evidence.get("target_observations")
            target_observations = "unknown" if target_observations is None else target_observations
            target_point_count = object_evidence.get("target_point_count")
            target_point_count = "unknown" if target_point_count is None else target_point_count
            evidence = (
                f"target_confidence={target_confidence}; "
                f"target_observations={target_observations}; "
                f"target_point_count={target_point_count}; "
                f"updated_in_current_observation={object_evidence.get('updated_in_current_observation')}; "
                f"best_label={class_evidence.get('best_label')}; "
                f"class_confidence_margin={class_evidence.get('confidence_margin')}; "
                f"selection={object_evidence.get('selection', 'unknown')}"
            )
        elif item["kind"] == "observe_rotation":
            evidence = (
                f"direction={target.get('direction', 'unknown')}; "
                f"rotation_rad={target.get('rotation_rad', 'unknown')}; "
                f"expected_yaw={target.get('expected_yaw', 'unknown')}; "
                f"information_gain_status={target.get('information_gain_status', 'unknown')}"
            )
        else:
            frontier_evidence = compact.get("frontier_evidence", {})
            evidence = (
                f"semantic_value={frontier_evidence.get('semantic_value')}; "
                f"semantic_fusion_weight={frontier_evidence.get('semantic_fusion_weight')}; "
                f"cluster_cells={frontier_evidence.get('cluster_cells')}; "
                f"frontier_state={frontier_evidence.get('state')}; "
                f"planning_status={frontier_evidence.get('planning_status')}"
            )
        options[item["id"]] = (
            f"{item['kind']} target={item['target_id']}; distance_m={item['path']['distance_m']}; "
            f"source_stage={item.get('source_stage')}; information_gain={item.get('information_gain')}; "
            f"{evidence}; {current_evidence}; {repetition}"
        )
    return PreparedDecision(
        state=state,
        options=options,
        model_request=build_model_request(state, options),
        eligible_ids=tuple(item["id"] for item in eligible),
        option_to_candidate={item["id"]: item["id"] for item in eligible},
    )


def decide(
    snapshot: dict[str, Any],
    provider: DecisionProvider,
    *,
    min_confidence: float = 0.4,
    min_probability: float = 0.5,
    state_profile: str = "full",
    return_trace: bool = False,
) -> Union[dict[str, Any], tuple[dict[str, Any], dict[str, Any]]]:
    """Return a validated high-level goal; failures are explicitly classified."""
    if not 0 <= min_confidence <= 1 or not 0 <= min_probability <= 1:
        raise SnapshotError("CONFIGURATION_ERROR", "decision thresholds must be between 0 and 1")
    prepared = prepare_decision(snapshot, state_profile)
    provider_metadata: dict[str, Any] = {}
    try:
        if hasattr(provider, "choose_request_with_metadata"):
            selected, confidence, probabilities, provider_metadata = provider.choose_request_with_metadata(
                prepared.model_request
            )
        elif hasattr(provider, "choose_request"):
            selected, confidence, probabilities = provider.choose_request(prepared.model_request)
        else:
            selected, confidence, probabilities = provider.choose(prepared.state, prepared.options)
    except ProviderError as exc:
        exc.model_request = prepared.model_request
        exc.eligible_candidate_ids = list(prepared.eligible_ids)
        raise
    except Exception as exc:
        error = ProviderError("PROVIDER_TEMPORARY_FAILURE", type(exc).__name__)
        error.model_request = prepared.model_request
        error.eligible_candidate_ids = list(prepared.eligible_ids)
        raise error from exc

    source_by_id = {
        item["id"]: item for item in snapshot.get("candidates", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    selected_candidate_id = prepared.option_to_candidate.get(selected) if isinstance(selected, str) else None
    if selected_candidate_id not in source_by_id or not _finite_number(confidence) or not 0 <= confidence <= 1:
        error = ProviderError("INVALID_PROVIDER_RESPONSE", "invalid selected choice or confidence")
        error.model_request = prepared.model_request
        error.eligible_candidate_ids = list(prepared.eligible_ids)
        raise error
    if not isinstance(probabilities, dict):
        error = ProviderError("INVALID_PROVIDER_RESPONSE", "probabilities must be an object")
        error.model_request = prepared.model_request
        error.eligible_candidate_ids = list(prepared.eligible_ids)
        raise error
    if any((state_profile == "region_graph" and k not in prepared.options) or not _finite_number(v) or not 0 <= v <= 1 for k,v in probabilities.items()):
        error = ProviderError("INVALID_PROVIDER_RESPONSE", "invalid probability member")
        error.model_request = prepared.model_request
        error.eligible_candidate_ids = list(prepared.eligible_ids)
        raise error
    selected_probability = probabilities.get(selected)
    if not _finite_number(selected_probability) or not 0 <= selected_probability <= 1:
        error = ProviderError("INVALID_PROVIDER_RESPONSE", "selected probability is invalid")
        error.model_request = prepared.model_request
        error.eligible_candidate_ids = list(prepared.eligible_ids)
        raise error
    candidate = source_by_id[selected_candidate_id]
    identity = prepared.state.get("identity", {}) if state_profile == "region_graph" else {}
    result = {
        "status": "GOAL",
        "candidate_id": selected_candidate_id,
        "kind": candidate["kind"],
        "target_id": candidate["target_id"],
        "episode_id": identity.get("episode_id", snapshot.get("episode_id")),
        "map_revision": identity.get("map_revision", snapshot.get("map", {}).get("revision")),
        "frame_id": prepared.state.get("representation", {}).get(
            "frame_id", snapshot.get("map", {}).get("frame_id")
        ),
        "goal_pose": candidate["goal_pose"],
        "confidence": confidence,
        "probability": selected_probability,
    }
    if state_profile == "region_graph":
        option = next(item for item in prepared.state["options"] if item["id"] == selected)
        result.update({
            "selected_option_id": selected,
            "action_type": option["action_type"],
            "region_id": option["region_id"],
            "request_id": identity["request_id"],
            "observation_id": identity["observation_id"],
            "region_graph_revision": identity["region_graph_revision"],
        })
    if return_trace:
        return result, {
            "state_profile": state_profile,
            "model_request": prepared.model_request,
            "eligible_candidate_ids": list(prepared.eligible_ids),
            "provider_metadata": provider_metadata,
        }
    return result
