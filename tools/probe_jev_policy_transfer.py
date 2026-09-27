#!/usr/bin/env python3
"""Probe real Jev choices on one recorded state with improved policy evidence.

The source trace supplies real regions, objects, frontiers, routes and local
candidates. This tool changes only high-level evidence/confirmation fields and
never reads dataset ground truth.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
import urllib.request
from pathlib import Path
from typing import Any

from nav_jev_bridge.bridge import OpenRouterJevProvider, decide
from nav_jev_bridge.region_graph import SCHEMA_VERSION
from nav_jev_bridge.server import implementation_hash


DECISION_ORDER = [
    "high_confidence_object",
    "current_over_depth_object",
    "active_frontier_policy",
    "suspicious_object_if_no_active_path",
    "dormant_frontier_policy",
    "extreme_object_search",
]


def entities(state: dict[str, Any], plural: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for region in state.get("regions", []):
        rows.extend(region.get(plural, []))
    rows.extend(state.get("unassigned_entities", {}).get(plural, []))
    return rows


def frontier_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        float(row["semantic_score"])
        for row in rows
        if row.get("selectable") and isinstance(row.get("semantic_score"), (int, float))
        and math.isfinite(row["semantic_score"])
    ]
    mean = sum(values) / len(values) if values else 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values) if values else 0.0
    maximum = max(values) if values else 0.0
    ratio = maximum / mean if abs(mean) > 1e-12 else 1.0
    return {
        "total_count": len(rows),
        "reachable_count": sum(bool(row.get("selectable")) for row in rows),
        "mean": mean,
        "std_dev": math.sqrt(variance),
        "maximum": maximum,
        "max_to_mean": ratio,
        "zero_mean_ratio_definition": 1.0,
        "adaptive_max_to_mean_threshold": max(1.2, 0.95 * ratio),
    }


def refresh_policy(state: dict[str, Any]) -> None:
    object_by_id = {row["id"]: row for row in entities(state, "objects")}
    frontier_by_id = {row["id"]: row for row in entities(state, "frontiers")}
    stage_priority: dict[str, int] = {}
    for option in state["options"]:
        target_id = option["target_id"]
        if option["action_type"] == "approach_object":
            obj = object_by_id[target_id]
            high = bool(obj.get("high_confidence_eligible"))
            option["source_stage"] = (
                "high_confidence_object" if high else "suspicious_object_if_no_active_path"
            )
            option["source_priority"] = 1 if high else 4
            option["semantic_score"] = obj.get("target_confidence")
        else:
            frontier = frontier_by_id.get(target_id)
            if frontier is None:
                option["source_stage"] = "active_observation"
                option["source_priority"] = 3
                option["semantic_score"] = None
            else:
                dormant = frontier.get("state") == "dormant"
                option["source_stage"] = (
                    "dormant_frontier_policy" if dormant else "active_frontier_policy"
                )
                option["source_priority"] = 5 if dormant else 3
                option["semantic_score"] = frontier.get("semantic_score")
        option["safety_mode"] = "in_place" if option.get("exploration_kind") == "observe_rotation" else "normal"
        option["information_gain"] = None
        stage_priority[option["source_stage"]] = min(
            stage_priority.get(option["source_stage"], 99), option["source_priority"]
        )
    earliest = min(stage_priority.values())
    for option in state["options"]:
        option["original_policy_stage_active"] = option["source_priority"] == earliest

    active = [row for row in frontier_by_id.values() if row.get("state") != "dormant"]
    dormant = [row for row in frontier_by_id.values() if row.get("state") == "dormant"]
    active_stats = frontier_stats(active)
    high_subset = []
    for row in active:
        value = row.get("semantic_score")
        ratio = value / active_stats["mean"] if isinstance(value, (int, float)) and abs(active_stats["mean"]) > 1e-12 else None
        high_subset.append({
            "frontier_id": row["id"],
            "selected": ratio is not None and ratio >= active_stats["adaptive_max_to_mean_threshold"],
            "value_to_mean": ratio,
        })
    ordered_stages = sorted(stage_priority, key=stage_priority.get)
    state["jev_obj_policy"] = {
        "mode": "hybrid",
        "mode_id": 2,
        "decision_order": DECISION_ORDER,
        "semantic_thresholds": {
            "std_dev": 0.03,
            "max_to_mean": 1.2,
            "max_to_mean_percentage": 0.95,
        },
        "active_frontier_statistics": active_stats,
        "dormant_frontier_statistics": frontier_stats(dormant),
        "high_semantic_subset": high_subset,
        "hybrid_branch": (
            "semantic_tsp"
            if active_stats["std_dev"] > 0.03 and active_stats["max_to_mean"] > 1.2
            else "closest"
        ),
        "available_stages": ordered_stages,
        "earliest_available_stage": ordered_stages[0],
        "earliest_available_priority": earliest,
        "interpretation": (
            "The local planner reference gate is a strong default, not a hard filter. When the earliest "
            "active stage is a frontier stage, below-threshold historical object hypotheses "
            "are deferred and should be overridden only by fresh synchronized target evidence "
            "or when no active-stage option remains."
        ),
    }


def enrich(state: dict[str, Any]) -> None:
    state["schema_version"] = SCHEMA_VERSION
    params = state.get("selection_parameters", {})
    min_conf = float(params.get("min_target_confidence", 0.5))
    min_obs = int(params.get("min_target_observations", 2))
    identity_observation_id = state.get("identity", {}).get("observation_id")
    detection = state.get("perception", {}).get("target_detection", {})
    detection_observation_id = detection.get("observation_id")
    synchronized_detection = (
        isinstance(identity_observation_id, int)
        and detection_observation_id == identity_observation_id
        and int(detection.get("matched_boxes") or 0) > 0
        and int(detection.get("valid_masks") or 0) > 0
    )
    for obj in entities(state, "objects"):
        confidence = obj.get("target_confidence")
        observations = obj.get("target_observations")
        high = (
            isinstance(confidence, (int, float))
            and isinstance(observations, int)
            and confidence >= min_conf
            and observations >= min_obs
        )
        updated_current = (
            obj.get("updated_in_current_observation") is True
            and obj.get("last_observed_id") == identity_observation_id
            and obj.get("observations_since_seen") == 0
        )
        current_supported = updated_current and synchronized_detection
        selectable = bool(obj.get("selectable"))
        if current_supported and high:
            selection_status = "current_frame_high_confidence_target_with_verified_path"
        elif high:
            selection_status = "historical_high_confidence_target_with_verified_path"
        elif selectable:
            selection_status = "deferred_weak_target_hypothesis_with_verified_path"
        else:
            selection_status = "not_selectable"
        obj.update({
            "stored_best_label": obj.get("stored_best_label"),
            "computed_best_label": obj.get("computed_best_label"),
            "high_confidence_eligible": high,
            "relaxed_eligible": isinstance(confidence, (int, float)) and confidence > 0.01,
            "selection_status": selection_status,
            "post_arrival_confirmation_status": (
                "not_arrived_current_frame_supported"
                if current_supported
                else "not_arrived_historical_or_unsynchronized_only"
            ),
        })
    state["object_confirmation"] = {
        "status": "idle",
        "pending_candidate_id": None,
        "pending_target_id": None,
        "start_observation_id": None,
        "last_observation_id": None,
        "attempts": 0,
        "valid_fresh_hits": 0,
        "required_fresh_hits": 2,
        "max_attempts": 4,
        "last_outcome": "idle",
        "blocked_candidate_id": None,
        "blocked_target_id": None,
        "blocked_until_observation_id": None,
        "stop_rule": (
            "approach is not success; STOP requires fresh synchronized target masks and a "
            "fresh matching map object in multiple post-arrival observations"
        ),
    }
    state["coverage"]["high_level_evidence_complete"] = False
    state["coverage"]["decision_evidence_complete"] = True
    refresh_policy(state)


def remove_candidate(state: dict[str, Any], registry: list[dict[str, Any]], candidate_id: str) -> None:
    removed_targets = {
        option["target_id"] for option in state["options"]
        if option["candidate_id"] == candidate_id
    }
    state["options"] = [
        option for option in state["options"] if option["candidate_id"] != candidate_id
    ]
    registry[:] = [item for item in registry if item["id"] != candidate_id]
    for obj in entities(state, "objects"):
        if obj["id"] in removed_targets:
            obj["selectable"] = False
            obj["selection_status"] = "not_selectable"
            obj["not_selectable_reason"] = [
                "post_arrival_multiview_confirmation_failed_cooldown"
            ]
            obj["post_arrival_confirmation_status"] = "rejected_cooldown"
    count = len(state["options"])
    state["coverage"]["executable_candidates_total"] = count
    state["coverage"]["executable_candidates_exported"] = count


def cases(source: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    original_state = source["model_request"]["state"]
    original_registry = source["local_candidate_registry"]
    output = []

    state = copy.deepcopy(original_state)
    registry = copy.deepcopy(original_registry)
    enrich(state)
    output.append(("weak_stale_objects_with_reference_gate", {"region_graph": state, "candidates": registry}))

    state = copy.deepcopy(state)
    registry = copy.deepcopy(registry)
    remove_candidate(state, registry, "object-0-goal")
    state["object_confirmation"].update({
        "status": "rejected_cooldown",
        "last_outcome": "rejected_missing_fresh_multiview_evidence",
        "blocked_candidate_id": "object-0-goal",
        "blocked_target_id": "object-0",
        "blocked_until_observation_id": state["identity"]["observation_id"] + 12,
    })
    refresh_policy(state)
    output.append(("object0_rejected_after_arrival", {"region_graph": state, "candidates": registry}))

    state = copy.deepcopy(state)
    registry = copy.deepcopy(registry)
    object1 = next(row for row in entities(state, "objects") if row["id"] == "object-1")
    object1.update({
        "target_confidence": 0.92,
        "target_observations": 4,
        "updated_in_current_observation": True,
        "observations_since_seen": 0,
        "last_observed_id": state["identity"]["observation_id"],
        "high_confidence_eligible": True,
        "relaxed_eligible": True,
        "selection_status": "current_frame_high_confidence_target_with_verified_path",
        "post_arrival_confirmation_status": "not_arrived_current_frame_supported",
    })
    state["perception"]["target_detection"].update({
        "observation_id": state["identity"]["observation_id"],
        "matched_boxes": 1,
        "valid_masks": 1,
    })
    refresh_policy(state)
    output.append(("fresh_high_confidence_object1", {"region_graph": state, "candidates": registry}))
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--bridge-url", default="http://127.0.0.1:8766")
    parser.add_argument(
        "--run-tag",
        help="unique label included in request IDs; defaults to the current timestamp",
    )
    parser.add_argument(
        "--allow-compatible-hash", action="append", default=[],
        help="explicitly audited bridge implementation hash accepted in addition to current source",
    )
    parser.add_argument("--direct", action="store_true",
                        help="call OpenRouter directly using OPENROUTER_API_KEY")
    args = parser.parse_args()

    source = json.loads(args.trace.read_text(encoding="utf-8"))
    provider = OpenRouterJevProvider(timeout_s=args.timeout) if args.direct else None
    health = None
    if not args.direct:
        with urllib.request.urlopen(f"{args.bridge_url}/health", timeout=5.0) as response:
            health = json.load(response)
        expected_hash = implementation_hash()
        actual_hash = health.get("provider", {}).get("implementation_sha256")
        if health.get("provider", {}).get("name") != "openrouter":
            raise RuntimeError("bridge is not using the real OpenRouter provider")
        if health.get("state_profile") != "region_graph":
            raise RuntimeError("bridge is not using the region_graph profile")
        accepted_hashes = {expected_hash, *args.allow_compatible_hash}
        if actual_hash not in accepted_hashes:
            raise RuntimeError(
                f"bridge process is stale: expected one of {sorted(accepted_hashes)}, got {actual_hash}"
            )
    run_tag = args.run_tag or str(int(time.time() * 1000))
    results = []
    for index, (name, snapshot) in enumerate(cases(source), start=1):
        identity = snapshot["region_graph"]["identity"]
        identity["request_id"] = (
            f"{identity['request_id']}:policy-probe-{run_tag}-{index}"
        )
        if args.direct:
            decision, trace = decide(
                snapshot, provider, state_profile="region_graph", return_trace=True
            )
            metadata = trace.get("provider_metadata", {})
            raw = metadata.get("raw_response", {})
            answer = raw.get("answers", {}).get("next_goal", {}) if isinstance(raw, dict) else {}
            probabilities = answer.get("probabilities", {}) if isinstance(answer, dict) else {}
            top = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:5]
        else:
            request = urllib.request.Request(
                f"{args.bridge_url}/decide",
                data=json.dumps(snapshot, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=args.timeout + 5.0) as response:
                decision = json.load(response)
            metadata = {}
            top = []
        results.append({
            "case": name,
            "selected_option_id": decision["selected_option_id"],
            "candidate_id": decision["candidate_id"],
            "action_type": decision["action_type"],
            "target_id": decision["target_id"],
            "confidence": decision["confidence"],
            "probability": decision["probability"],
            "top_probabilities": top,
            "actual_model": metadata.get("actual_model"),
            "generation_id": metadata.get("generation_id"),
            "usage": metadata.get("usage"),
        })
    payload = {
        "source_trace": str(args.trace),
        "ground_truth_used": False,
        "provider": "openrouter",
        "bridge_health": health,
        "current_source_implementation_sha256": implementation_hash(),
        "explicitly_allowed_compatible_hashes": args.allow_compatible_hash,
        "requested_model": OpenRouterJevProvider.MODEL,
        "run_tag": run_tag,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
