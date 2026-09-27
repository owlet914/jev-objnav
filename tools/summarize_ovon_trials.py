"""Summarize OVON pilot/dev Jev Obj records and Jev decision traces.

Each child of --runs-root is one configuration and may contain record.txt,
continue.txt and traces/*.json. Missing trials are reported as incomplete;
there is no fabricated success rate.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import re
import statistics
from pathlib import Path


def parse_run(directory: Path, expected_episodes: int = 24) -> dict:
    result: dict = {"name": directory.name, "complete": False}
    record = directory / "record.txt"
    if record.is_file():
        blocks = record.read_text(encoding="utf-8").split("Scene ID: ")[1:]
        outcomes = []
        for block in blocks:
            match = re.search(r"^success or not:\s*(.+)$", block, re.MULTILINE)
            if match:
                outcomes.append(match.group(1).strip())
        result["episodes_recorded"] = len(outcomes)
        result["successes"] = outcomes.count("success")
        result["failure_reasons"] = dict(Counter(outcome for outcome in outcomes if outcome != "success"))
        if outcomes:
            result["success_rate"] = result["successes"] / len(outcomes)
    continuation = directory / "continue.txt"
    if continuation.is_file():
        content = continuation.read_text(encoding="utf-8")
        match = re.search(r"Total SPL\s+\|\s+([\d.]+)", content)
        if match and result.get("episodes_recorded"):
            result["spl"] = float(match.group(1)) / result["episodes_recorded"]
    traces = []
    for path in sorted((directory / "traces").glob("*.json")):
        try:
            traces.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    if traces:
        holds = sum(trace.get("decision", {}).get("status") == "HOLD" for trace in traces)
        latencies = [trace["latency_ms"] for trace in traces if isinstance(trace.get("latency_ms"), (int, float))]
        result["jev_decisions"] = len(traces)
        result["non_goal_decision_rate"] = holds / len(traces)
        if latencies:
            result["median_jev_latency_ms"] = statistics.median(latencies)
    result["complete"] = result.get("episodes_recorded", 0) == expected_episodes
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("jev_obj/videos/ovon_trials"))
    parser.add_argument("--expected-episodes", type=int, default=24)
    parser.add_argument("--output", type=Path, default=Path("examples/ovon_trial_summary.json"))
    args = parser.parse_args()
    runs = [
        parse_run(path, args.expected_episodes)
        for path in sorted(args.runs_root.iterdir()) if path.is_dir()
    ] if args.runs_root.is_dir() else []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(runs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(runs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
