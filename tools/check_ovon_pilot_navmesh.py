"""Check pilot/dev OVON starts and scoring viewpoints against HM3D navmeshes.

This uses ground-truth viewpoints only for offline dataset validation. It never
exports them to the online Jev snapshot and does not measure policy success.
The reserved val_unseen split is not read.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import habitat_sim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", type=Path,
        default=Path("jev_obj/data/datasets/objectnav/ovon_small"),
    )
    parser.add_argument(
        "--scenes", type=Path, default=Path("jev_obj/data/scene_datasets")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("examples/ovon_navmesh_check.json")
    )
    args = parser.parse_args()
    report = {}
    for split in ("pilot", "dev"):
        stats = {
            "episodes": 0,
            "start_navigable": 0,
            "viewpoint_path_found": 0,
            "scenes_loaded": 0,
            "offmesh_starts": [],
        }
        for shard in sorted((args.datasets / split / "content").glob("*.json.gz")):
            with gzip.open(shard, "rt", encoding="utf-8") as stream:
                data = json.load(stream)
            scene = args.scenes / data["episodes"][0]["scene_id"]
            finder = habitat_sim.PathFinder()
            if not finder.load_nav_mesh(str(scene.with_suffix(".navmesh"))):
                raise RuntimeError(f"cannot load navmesh for {scene}")
            stats["scenes_loaded"] += 1
            for episode in data["episodes"]:
                stats["episodes"] += 1
                start = episode["start_position"]
                if finder.is_navigable(start):
                    stats["start_navigable"] += 1
                else:
                    snapped = finder.snap_point(start)
                    stats["offmesh_starts"].append({
                        "scene": shard.stem,
                        "episode_id": episode["info"].get("ovon_episode_id"),
                        "snap_distance_m": math.dist(start, snapped),
                    })
                key = f"{Path(episode['scene_id']).name}_{episode['object_category']}"
                goals = data["goals_by_category"][key]
                viewpoints = (
                    view["agent_state"]["position"]
                    for goal in goals for view in goal["view_points"]
                )
                for position in viewpoints:
                    path = habitat_sim.ShortestPath()
                    path.requested_start = start
                    path.requested_end = position
                    if finder.find_path(path):
                        stats["viewpoint_path_found"] += 1
                        break
        report[split] = stats
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
