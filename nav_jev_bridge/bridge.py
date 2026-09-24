"""Validate a navigation snapshot and select one planner-supplied candidate."""

from __future__ import annotations

import math
import json
import os
import urllib.request
from typing import Any, Protocol


STATE_PROFILES = ("candidates", "spatial", "full")


class DecisionProvider(Protocol):
    def choose(self, state: dict[str, Any], options: dict[str, str]) -> tuple[str, float, dict[str, float]]:
        """Return (candidate ID or hold, confidence, option probabilities)."""


class OpenRouterJevProvider:
    """Call Jev through OpenRouter's Decisions API, not Chat Completions."""

    ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
    MODEL = "typesafe/jev-1.13"

    def __init__(self, timeout_s: float = 4.0, model: str = MODEL, api_key: str | None = None):
        self.timeout_s = timeout_s
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")

    def choose(self, state: dict[str, Any], options: dict[str, str]) -> tuple[str, float, dict[str, float]]:
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")
        instructions = (
            "Find the object named in state.target_object. Choose one reachable candidate ID. "
            "Compare candidate semantic scores, information gain and path cost. "
            "Choose hold if none has adequate evidence. Never invent a position or a candidate."
        )
        if "local_grid" in state.get("map", {}):
            instructions += (
                " Also use observed target confidence and observation count, frontier semantic "
                "value and confidence, and the robot-centered occupancy/semantic grid. "
                "Do not treat unknown cells as free."
            )
        if "room_prior" in state or "target_subcategories" in state:
            instructions += (
                " Use the room prior, related objects, target subcategories and per-class "
                "object evidence as context. A room prior is not an observed object position. "
                "Prefer a well-supported target detection when evidence supports it."
            )
        payload = {
            "model": self.model,
            "state": state,
            "questions": {
                "next_goal": {
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": options,
                }
            },
        }
        request = urllib.request.Request(
            self.ENDPOINT,
            data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            body = json.load(response)
        answer = body["answers"]["next_goal"]
        return answer["choice"], float(answer["confidence"]), dict(answer["probabilities"])


class DemoProvider:
    """Offline smoke test only; this is a heuristic, not a Jev simulation."""

    def choose(self, state: dict[str, Any], options: dict[str, str]) -> tuple[str, float, dict[str, float]]:
        candidates = state["candidates"]
        if not candidates:
            return "hold", 1.0, {"hold": 1.0}
        best = max(
            candidates,
            key=lambda item: (
                float(item.get("semantic_score", 0))
                + float(item.get("information_gain", 0))
                - 0.05 * float(item["path"]["distance_m"])
            ),
        )
        return best["id"], 1.0, {best["id"]: 1.0}


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _valid_pose(pose: Any) -> bool:
    return isinstance(pose, dict) and all(_finite_number(pose.get(key)) for key in ("x", "y", "yaw"))


def validate_snapshot(snapshot: Any) -> None:
    """Check the boundary contract before any model call or goal dispatch."""
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != "1.0":
        raise ValueError("schema_version must be '1.0'")
    if not isinstance(snapshot.get("target_object"), str) or not snapshot["target_object"].strip():
        raise ValueError("target_object must be a nonempty string")
    if not isinstance(snapshot.get("episode_id"), str) or not snapshot["episode_id"]:
        raise ValueError("episode_id is required")
    if type(snapshot.get("timestamp_ms")) is not int or snapshot["timestamp_ms"] < 0:
        raise ValueError("timestamp_ms must be a nonnegative integer")
    if not isinstance(snapshot.get("map"), dict) or not isinstance(snapshot["map"].get("frame_id"), str) or not snapshot["map"]["frame_id"]:
        raise ValueError("map.frame_id is required")
    if type(snapshot["map"].get("revision")) is not int or snapshot["map"]["revision"] < 0:
        raise ValueError("map.revision must be a nonnegative integer")
    if not _valid_pose(snapshot.get("robot_pose")):
        raise ValueError("robot_pose requires finite x, y and yaw")
    for key in ("rooms", "objects", "semantic_matches", "frontiers", "candidates"):
        if not isinstance(snapshot.get(key), list):
            raise ValueError(f"{key} must be a list")
    if len(snapshot["candidates"]) > 254:
        raise ValueError("at most 254 candidates are supported, reserving one choice for hold")
    object_ids = {item.get("id") for item in snapshot["objects"] if isinstance(item, dict)}
    frontier_ids = {item.get("id") for item in snapshot["frontiers"] if isinstance(item, dict)}
    seen: set[str] = set()
    for item in snapshot["candidates"]:
        if not isinstance(item, dict):
            raise ValueError("each candidate must be an object")
        candidate_id = item.get("id")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id == "hold" or candidate_id in seen:
            raise ValueError("candidate IDs must be unique nonempty strings other than 'hold'")
        seen.add(candidate_id)
        if item.get("kind") not in ("explore_frontier", "approach_object"):
            raise ValueError(f"invalid kind for candidate {candidate_id}")
        if not isinstance(item.get("target_id"), str) or not item["target_id"]:
            raise ValueError(f"target_id is required for candidate {candidate_id}")
        valid_targets = frontier_ids if item["kind"] == "explore_frontier" else object_ids
        if item["target_id"] not in valid_targets:
            raise ValueError(f"target_id does not reference an available entity for candidate {candidate_id}")
        if not _valid_pose(item.get("goal_pose")):
            raise ValueError(f"invalid goal_pose for candidate {candidate_id}")
        path = item.get("path")
        if not isinstance(path, dict) or type(path.get("reachable")) is not bool or type(path.get("collision_free")) is not bool:
            raise ValueError(f"path flags must be booleans for candidate {candidate_id}")
        if not _finite_number(path.get("distance_m")) or path["distance_m"] < 0:
            raise ValueError(f"invalid path.distance_m for candidate {candidate_id}")
        for score in ("semantic_score", "information_gain"):
            value = item.get(score, 0)
            if not _finite_number(value) or not 0 <= value <= 1:
                raise ValueError(f"{score} must be between 0 and 1 for candidate {candidate_id}")


def build_model_state(snapshot: dict[str, Any], profile: str) -> dict[str, Any]:
    """Choose observed fields for an ablation; never add dataset ground truth."""
    if profile not in STATE_PROFILES:
        raise ValueError(f"unknown state profile: {profile}")
    if profile == "full":
        return snapshot

    state = {
        key: snapshot[key]
        for key in ("schema_version", "timestamp_ms", "episode_id", "target_object", "robot_pose", "candidates")
    }
    state["map"] = {
        key: value for key, value in snapshot["map"].items()
        if key != "local_grid"
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
        if "local_grid" in snapshot["map"]:
            state["map"]["local_grid"] = snapshot["map"]["local_grid"]
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


def decide(
    snapshot: dict[str, Any],
    provider: DecisionProvider,
    *,
    min_confidence: float = 0.4,
    min_probability: float = 0.5,
    state_profile: str = "full",
) -> dict[str, Any]:
    """Return a validated high-level goal or HOLD; never issue motor commands."""
    validate_snapshot(snapshot)
    if not 0 <= min_confidence <= 1 or not 0 <= min_probability <= 1:
        raise ValueError("decision thresholds must be between 0 and 1")
    if state_profile not in STATE_PROFILES:
        raise ValueError(f"unknown state profile: {state_profile}")

    eligible = [
        item for item in snapshot["candidates"]
        if item["path"]["reachable"] and item["path"]["collision_free"]
    ]
    if not eligible:
        return {"status": "HOLD", "reason": "no_reachable_collision_free_candidate"}

    state = build_model_state({**snapshot, "candidates": eligible}, state_profile)
    objects_by_id = {item.get("id"): item for item in state["objects"] if isinstance(item, dict)}
    frontiers_by_id = {item.get("id"): item for item in state["frontiers"] if isinstance(item, dict)}
    options = {}
    for item in eligible:
        target = (objects_by_id if item["kind"] == "approach_object" else frontiers_by_id)[item["target_id"]]
        evidence = (
            f"target_confidence={target.get('target_confidence', 'unknown')}; "
            f"target_observations={target.get('target_observations', 'unknown')}; "
            f"score_available={target.get('score_available', 'unknown')}; "
            f"confidence_tier={target.get('confidence_tier', 'unknown')}"
            if item["kind"] == "approach_object" else
            f"semantic_value={target.get('semantic_value', item.get('semantic_score', 0))}; "
            f"semantic_confidence={target.get('semantic_confidence', 'unknown')}; "
            f"cluster_cells={target.get('cluster_cells', 'unknown')}"
        )
        options[item["id"]] = (
            f"{item['kind']} target={item['target_id']}; "
            f"distance_m={item['path']['distance_m']}; "
            f"information_gain={item.get('information_gain', 0)}; {evidence}"
        )
    options["hold"] = "Wait for a new observation or replan; do not navigate."

    try:
        selected, confidence, probabilities = provider.choose(state, options)
    except Exception as exc:
        return {"status": "HOLD", "reason": "decision_provider_error", "error_type": type(exc).__name__}

    by_id = {item["id"]: item for item in eligible}
    if selected == "hold":
        return {"status": "HOLD", "reason": "provider_selected_hold", "confidence": confidence}
    if not isinstance(selected, str) or selected not in by_id or not _finite_number(confidence) or not 0 <= confidence <= 1:
        return {"status": "HOLD", "reason": "invalid_provider_answer"}
    if not isinstance(probabilities, dict):
        return {"status": "HOLD", "reason": "invalid_provider_probabilities"}
    selected_probability = probabilities.get(selected)
    if not _finite_number(selected_probability) or not 0 <= selected_probability <= 1:
        return {"status": "HOLD", "reason": "invalid_provider_probability"}
    if confidence < min_confidence or selected_probability < min_probability:
        return {
            "status": "HOLD",
            "reason": "uncertain_decision",
            "candidate_id": selected,
            "confidence": confidence,
            "probability": selected_probability,
        }

    candidate = by_id[selected]
    return {
        "status": "GOAL",
        "candidate_id": selected,
        "kind": candidate["kind"],
        "target_id": candidate["target_id"],
        "episode_id": snapshot["episode_id"],
        "map_revision": snapshot["map"]["revision"],
        "frame_id": snapshot["map"]["frame_id"],
        "goal_pose": candidate["goal_pose"],
        "confidence": confidence,
        "probability": selected_probability,
    }
