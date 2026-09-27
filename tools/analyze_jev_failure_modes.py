"""Analyze Jev trace evidence conflicts and repeated-selection loops.

This tool is descriptive. The score bands are reported as analysis bands, not
as calibrated success thresholds or dataset ground truth.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Union


def _describe(values: list[float]) -> dict[str, Union[float, int]]:
    if not values:
        return {}
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "max": max(values),
    }


def _band(score: Any) -> str:
    if not isinstance(score, (int, float)):
        return "unavailable"
    if score < 0.15:
        return "below_0.15"
    if score < 0.25:
        return "0.15_to_0.25"
    return "at_least_0.25"


def analyze(trace_dir: Path) -> dict[str, Any]:
    kinds: Counter[str] = Counter()
    score_bands: Counter[str] = Counter()
    selected_ids: Counter[str] = Counter()
    current_scores: list[float] = []
    selected_confidences: list[float] = []
    selected_observations: list[float] = []
    selected_distances: list[float] = []
    repeats: list[float] = []
    episode_counts: Counter[str] = Counter()
    episode_kinds: dict[str, Counter[str]] = defaultdict(Counter)
    weak_view_high_map = []
    repeated_selection = []
    repeated_no_progress = []
    stale_selected_object = []
    unreadable = 0

    for path in sorted(trace_dir.glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            state = row.get("model_request", {}).get("state") or row.get("snapshot")
            decision = row.get("decision", {})
            if not isinstance(state, dict) or not isinstance(decision, dict):
                raise ValueError("missing state or decision")
        except (OSError, ValueError, TypeError):
            unreadable += 1
            continue

        episode = str(state.get("episode_id", "unknown"))
        candidate_id = str(decision.get("candidate_id", "unknown"))
        kind = str(decision.get("kind", "unknown"))
        candidates = {
            item.get("id"): item for item in state.get("candidates", [])
            if isinstance(item, dict) and item.get("id") is not None
        }
        objects = {
            item.get("id"): item for item in state.get("objects", [])
            if isinstance(item, dict) and item.get("id") is not None
        }
        selected = candidates.get(candidate_id, {})
        history = state.get("navigation_history", {})
        current_view = state.get("current_view", {})
        score = current_view.get("raw_score", current_view.get("image_text_match_score"))
        repeats_count = history.get("consecutive_same_entity_selections")
        progress = history.get("last_goal_execution", {}).get("last_step_progress_m")

        kinds[kind] += 1
        score_bands[_band(score)] += 1
        selected_ids[candidate_id] += 1
        episode_counts[episode] += 1
        episode_kinds[episode][kind] += 1
        if isinstance(score, (int, float)):
            current_scores.append(float(score))
        if isinstance(repeats_count, int):
            repeats.append(float(repeats_count))
        distance = selected.get("path", {}).get("distance_m") if isinstance(selected, dict) else None
        if isinstance(distance, (int, float)):
            selected_distances.append(float(distance))

        if kind != "approach_object" or not isinstance(selected, dict):
            continue
        target_id = selected.get("target_id")
        obj = objects.get(target_id, {})
        confidence = obj.get("target_confidence")
        observations = obj.get("target_observations")
        if isinstance(confidence, (int, float)):
            selected_confidences.append(float(confidence))
        if isinstance(observations, (int, float)):
            selected_observations.append(float(observations))

        sample = {
            "trace": path.name,
            "episode_id": episode,
            "observation_id": state.get("observation_id"),
            "candidate_id": candidate_id,
            "target_id": target_id,
            "current_view_score": score,
            "target_confidence": confidence,
            "target_observations": observations,
            "repeat_count": repeats_count,
            "last_step_progress_m": progress,
        }
        if isinstance(score, (int, float)) and score < 0.15 and isinstance(confidence, (int, float)) and confidence >= 0.5:
            weak_view_high_map.append(sample)
        if isinstance(repeats_count, int) and repeats_count >= 3:
            repeated_selection.append(sample)
            if isinstance(progress, (int, float)) and progress <= 0.01:
                repeated_no_progress.append(sample)
        provenance_observation = obj.get("provenance", {}).get("observation_id")
        if provenance_observation is not None and provenance_observation != state.get("observation_id"):
            stale_selected_object.append(sample)

    return {
        "trace_files": sum(episode_counts.values()),
        "unreadable_files": unreadable,
        "episodes_with_traces": len(episode_counts),
        "goal_kinds": dict(kinds),
        "current_view_score_bands": dict(score_bands),
        "current_view_scores": _describe(current_scores),
        "selected_candidate_ids": dict(selected_ids),
        "selected_object_confidences": _describe(selected_confidences),
        "selected_object_observations": _describe(selected_observations),
        "selected_path_distances_m": _describe(selected_distances),
        "consecutive_same_entity_selections": _describe(repeats),
        "weak_view_high_map_conflict_count": len(weak_view_high_map),
        "repeated_object_selection_count": len(repeated_selection),
        "repeated_object_no_progress_count": len(repeated_no_progress),
        "selected_object_not_from_current_observation_count": len(stale_selected_object),
        "example_conflicts": weak_view_high_map[:10],
        "example_repeated_no_progress": repeated_no_progress[:10],
        "per_episode": {
            episode: {"decisions": count, "goal_kinds": dict(episode_kinds[episode])}
            for episode, count in episode_counts.items()
        },
        "analysis_note": (
            "Score bands and conflict rules are descriptive diagnostics, not calibrated success "
            "thresholds. They use only runtime evidence sent to the decision provider."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.traces)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
