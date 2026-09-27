"""Aggregate observed Jev request snapshots and decisions without exporting raw state."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Union


def summarize(traces: Path) -> dict:
    statuses: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    candidate_kinds: Counter[str] = Counter()
    image_scores: list[float] = []
    image_score_types: Counter[str] = Counter()
    current_view_validity: Counter[str] = Counter()
    current_view_backends: Counter[str] = Counter()
    current_view_fallbacks: Counter[str] = Counter()
    target_confidences: list[float] = []
    target_observations: list[int] = []
    repeat_counts: list[int] = []
    semantic_zero_candidates = 0
    candidate_total = 0
    profiles: Counter[str] = Counter()
    selected: Counter[str] = Counter()
    features: Counter[str] = Counter()
    counts: dict[str, list[int]] = defaultdict(list)
    per_episode: dict[str, Counter[str]] = defaultdict(Counter)
    latencies: list[float] = []
    corrupt = 0
    for path in sorted(traces.glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            state = row.get("snapshot") or row.get("model_request", {}).get("state")
            if not isinstance(state, dict):
                continue
            decision = row["decision"]
            if not isinstance(state, dict) or not isinstance(decision, dict):
                raise ValueError("trace must include snapshot and decision objects")
        except (OSError, ValueError, KeyError, TypeError):
            corrupt += 1
            continue
        profile = str(row.get("state_profile", "unknown"))
        profiles[profile] += 1
        status = str(decision.get("status", "unknown"))
        statuses[status] += 1
        if status == "HOLD":
            reasons[str(decision.get("reason", "unknown"))] += 1
        if status == "GOAL":
            kinds[str(decision.get("kind", "unknown"))] += 1
            selected[str(decision.get("candidate_id", "unknown"))] += 1
        for candidate in state.get("candidates", []):
            if isinstance(candidate, dict):
                candidate_kinds[str(candidate.get("kind", "unknown"))] += 1
                candidate_total += 1
                semantic_zero_candidates += candidate.get("semantic_score") == 0
        current_view = state.get("current_view", {})
        score = current_view.get("raw_score", current_view.get("image_text_match_score"))
        if isinstance(score, (int, float)):
            image_scores.append(float(score))
        image_score_types[str(current_view.get("score_type", "unknown"))] += 1
        current_view_validity[str(current_view.get("valid", "unknown"))] += 1
        current_view_backends[str(current_view.get("backend", "unknown"))] += 1
        current_view_fallbacks[str(current_view.get("fallback", "unknown"))] += 1
        for obj in state.get("objects", []):
            if not isinstance(obj, dict):
                continue
            confidence = obj.get("target_confidence")
            observations = obj.get("target_observations")
            if isinstance(confidence, (int, float)):
                target_confidences.append(float(confidence))
            if isinstance(observations, int):
                target_observations.append(observations)
        repeat_count = state.get("navigation_history", {}).get("consecutive_same_entity_selections")
        if isinstance(repeat_count, int):
            repeat_counts.append(repeat_count)
        episode_id = str(state.get("episode_id", "unknown"))
        per_episode[episode_id][status] += 1
        for name in ("candidates", "objects", "frontiers", "semantic_matches", "rooms"):
            values = state.get(name)
            counts[name].append(len(values) if isinstance(values, list) else 0)
        for field, present in {
            "local_grid": "local_grid" in state.get("map", {}),
            "current_view": "current_view" in state,
            "room_prior": bool(state.get("room_prior")),
            "related_objects": bool(state.get("related_objects")),
            "target_subcategories": bool(state.get("target_subcategories")),
            "observed_objects": bool(state.get("objects")),
            "frontiers": bool(state.get("frontiers")),
            "reachable_candidates": any(
                bool(c.get("path", {}).get("reachable")) and
                bool(c.get("path", {}).get("collision_free"))
                for c in state.get("candidates", []) if isinstance(c, dict)
            ),
        }.items():
            features[field] += int(present)
        latency = row.get("latency_ms")
        if isinstance(latency, (int, float)) and latency >= 0:
            latencies.append(float(latency))

    def describe(values: list[Union[float, int]]) -> dict[str, Union[float, int]]:
        return {
            "count": len(values),
            "min": min(values),
            "median": statistics.median(values),
            "mean": statistics.mean(values),
            "max": max(values),
        } if values else {}

    return {
        "trace_files": sum(profiles.values()), "unreadable_files": corrupt,
        "episodes_with_traces": len(per_episode), "profile_counts": dict(profiles),
        "decision_statuses": dict(statuses), "hold_reasons": dict(reasons),
        "goal_kinds": dict(kinds), "selected_candidate_ids": dict(selected),
        "candidate_kinds": dict(candidate_kinds),
        "zero_semantic_score_candidates": semantic_zero_candidates,
        "candidate_total": candidate_total,
        "image_text_match_scores": describe(image_scores),
        "image_text_match_score_types": dict(image_score_types),
        "current_view_validity": dict(current_view_validity),
        "current_view_backends": dict(current_view_backends),
        "current_view_fallbacks": dict(current_view_fallbacks),
        "object_target_confidences": describe(target_confidences),
        "object_target_observations": describe(target_observations),
        "consecutive_same_entity_selections": describe(repeat_counts),
        "feature_presence": dict(features),
        "entity_counts": {name: describe(values) for name, values in counts.items()},
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "per_episode_decisions": {key: dict(value) for key, value in per_episode.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(args.traces)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
