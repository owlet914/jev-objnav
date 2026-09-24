"""Compare Jev information profiles on the same saved pilot snapshots.

This is an open-loop decision comparison, not a navigation success metric.
Only run it on pilot traces; do not tune against reserved val_unseen traces.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from nav_jev_bridge.bridge import (
    STATE_PROFILES,
    DemoProvider,
    OpenRouterJevProvider,
    decide,
)


def replay(trace_dir: Path, provider, max_snapshots: int, thresholds: list[float]) -> dict:
    snapshots = []
    for path in sorted(trace_dir.glob("*.json")):
        trace = json.loads(path.read_text(encoding="utf-8"))
        if "snapshot" in trace:
            snapshots.append(trace["snapshot"])
        if len(snapshots) >= max_snapshots:
            break
    result = {"snapshots": len(snapshots), "profiles": {}}
    for profile in STATE_PROFILES:
        decisions = []
        latencies = []
        for snapshot in snapshots:
            started = time.perf_counter()
            decision = decide(
                snapshot, provider,
                min_confidence=0.0, min_probability=0.0,
                state_profile=profile,
            )
            latencies.append((time.perf_counter() - started) * 1000)
            decisions.append(decision)
        result["profiles"][profile] = {
            "goal_choices": sum(item["status"] == "GOAL" for item in decisions),
            "hold_choices": sum(item["status"] == "HOLD" for item in decisions),
            "median_latency_ms": statistics.median(latencies) if latencies else None,
            "accepted_at_threshold": {
                str(threshold): sum(
                    item["status"] == "GOAL"
                    and item["confidence"] >= threshold
                    and item["probability"] >= 0.5
                    for item in decisions
                )
                for threshold in thresholds
            },
            "candidate_ids": [item.get("candidate_id") for item in decisions],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--provider", choices=("demo", "openrouter"), default="demo")
    parser.add_argument("--max-snapshots", type=int, default=20)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.4, 0.6])
    parser.add_argument("--output", type=Path, default=Path("examples/ovon_replay_summary.json"))
    args = parser.parse_args()
    if args.max_snapshots <= 0 or any(not 0 <= value <= 1 for value in args.thresholds):
        parser.error("max-snapshots must be positive and thresholds must be within [0, 1]")
    if args.provider == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
        parser.error("OPENROUTER_API_KEY is required for OpenRouter replay")
    provider = OpenRouterJevProvider() if args.provider == "openrouter" else DemoProvider()
    result = replay(args.traces, provider, args.max_snapshots, args.thresholds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
