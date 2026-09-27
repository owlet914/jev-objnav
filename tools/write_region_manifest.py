"""Write a secret-free manifest for one region-graph evaluation run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
from region_run_fingerprint import source_fingerprint, pilot_manifest
from preflight_jev_region import validate_health


ROOT = Path(__file__).resolve().parents[1]


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "pilot10", "pilot24"), required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--smoke-episode-index", type=int, default=0)
    parser.add_argument("--health-url", default="http://127.0.0.1:8766/health")
    args = parser.parse_args()

    with urlopen(args.health_url, timeout=5) as response:
        health = json.load(response)
    validate_health(health)
    planner = ROOT / "jev_obj/devel/lib/exploration_manager/exploration_node"
    if not planner.is_file(): raise RuntimeError("planner binary missing")
    status = git("status", "--porcelain=v1")
    fingerprint = source_fingerprint(ROOT)
    dataset = pilot_manifest(ROOT)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "expected_episode_count": {"smoke": 1, "pilot10": 10, "pilot24": 24}[args.mode],
        "smoke_episode_index": args.smoke_episode_index if args.mode == "smoke" else None,
        "dataset": "ovon_small",
        "split": "pilot",
        "uses_val_unseen": False,
        "command": args.command,
        "git_head": git("rev-parse", "HEAD").decode().strip(),
        "working_tree_dirty": bool(status),
        "working_tree_fingerprint_sha256": fingerprint["sha256"],
        "source_fingerprint": fingerprint,
        "dataset_manifest": dataset,
        "run_id": args.run_dir.name,
        "serializer_version": "json-utf8-v2",
        "planner_binary_sha256": hashlib.sha256(planner.read_bytes()).hexdigest(),
        "timeouts": {"upstream_s":30,"ros_http_s":45},
        "token_targets": {"typical_input": [6000,12000], "stress_input":24000},
        "configuration_status": "expected; effective config and ordered episodes verified by evaluator",
        "bridge_health": health,
        "region_contract": {
            "schema_version": "jev-region-graph/1.1",
            "region_tile_size_m": 3.0,
            "summary_is_lossy": True,
            "local_execution_uses_full_precision": True,
        },
        "perception": {
            "required_services": ["grounding_dino", "blip2_itm", "mobile_sam", "yolov7"],
            "fallback_allowed": False,
            "preflight_report": "docs/JEV_REGION_GRAPH_VALIDATION.md",
        },
    }
    args.run_dir.mkdir(parents=True, exist_ok=False)
    (args.run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
