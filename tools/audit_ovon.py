"""Audit OVON validation episodes and required HM3D scene assets.

This reads labels only. It never exports ground-truth object goals to Jev.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


SPLITS = ("val_seen", "val_seen_synonyms", "val_unseen")


def audit_split(root: Path, scenes_root: Path, split: str) -> dict:
    split_dir = root / split
    shards = sorted((split_dir / "content").glob("*.json.gz"))
    if not shards or not list(split_dir.glob("*.json.gz")):
        raise FileNotFoundError(f"incomplete OVON split: {split_dir}")

    episode_count = 0
    categories: set[str] = set()
    scene_ids: set[str] = set()
    missing_goal_keys = 0
    for shard in shards:
        with gzip.open(shard, "rt", encoding="utf-8") as stream:
            data = json.load(stream)
        goal_keys = data.get("goals_by_category", {})
        for episode in data.get("episodes", []):
            episode_count += 1
            category = episode["object_category"]
            scene_id = episode["scene_id"]
            categories.add(category)
            scene_ids.add(scene_id)
            goal_key = f"{Path(scene_id).name}_{category}"
            if goal_key not in goal_keys:
                missing_goal_keys += 1

    missing_scenes = sorted(
        scene_id for scene_id in scene_ids if not (scenes_root / scene_id).is_file()
    )
    return {
        "shards": len(shards),
        "episodes": episode_count,
        "categories": len(categories),
        "scenes": len(scene_ids),
        "missing_goal_keys": missing_goal_keys,
        "missing_scene_meshes": len(missing_scenes),
        "missing_scene_examples": missing_scenes[:5],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes-root", type=Path,
        default=Path("jev_obj/data/datasets/ovon/hm3d_ovon"),
    )
    parser.add_argument(
        "--scenes-root", type=Path,
        default=Path("jev_obj/data/scene_datasets"),
    )
    parser.add_argument("--output", type=Path, default=Path("examples/ovon_audit.json"))
    parser.add_argument("--splits", nargs="+", default=SPLITS)
    args = parser.parse_args()
    result = {
        "source": str(args.episodes_root),
        "scene_root": str(args.scenes_root),
        "scene_dataset_config_present": (
            args.scenes_root / "hm3d/hm3d_annotated_basis.scene_dataset_config.json"
        ).is_file(),
        "splits": {
            split: audit_split(args.episodes_root, args.scenes_root, split)
            for split in args.splits
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
