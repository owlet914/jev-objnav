"""Convert OVON validation episode metadata for Habitat-Lab 0.3.1 ObjectNav-v1.

The conversion keeps the ground-truth goals in Habitat's dataset for scoring.
The Jev bridge never reads them. HM3D meshes and Habitat are still required.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


SPLITS = ("val_seen", "val_seen_synonyms", "val_unseen")


def load_gzip(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def write_gzip(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))


def convert_split(source: Path, destination: Path, split: str) -> dict:
    shards = sorted((source / split / "content").glob("*.json.gz"))
    if not shards:
        raise FileNotFoundError(f"OVON content shards missing: {source / split}")

    categories: set[str] = set()
    total = 0
    for shard in shards:
        data = load_gzip(shard)
        for episode in data["episodes"]:
            categories.add(episode["object_category"])
            total += 1
    category_ids = {name: index for index, name in enumerate(sorted(categories))}

    for shard in shards:
        data = load_gzip(shard)
        for episode in data["episodes"]:
            info = episode.setdefault("info", {})
            info["ovon_episode_id"] = episode["episode_id"]
            # OVON-v1 has this extra episode field; ObjectNav-v1 does not.
            # The category names are task metadata, not ground-truth positions.
            info["children_object_categories"] = episode.pop(
                "children_object_categories", []
            )
        # Habitat-Lab's ObjectGoalSensor needs a nonempty task-category map.
        # These IDs are for the observation sensor only; actual success uses
        # each goal's stored viewpoints, retained in goals_by_category.
        data["category_to_task_category_id"] = category_ids
        data["category_to_scene_annotation_category_id"] = category_ids
        write_gzip(destination / split / "content" / shard.name, data)

    stub = {
        "episodes": [],
        "category_to_task_category_id": category_ids,
        "category_to_scene_annotation_category_id": category_ids,
    }
    write_gzip(destination / split / f"{split}.json.gz", stub)
    return {"shards": len(shards), "episodes": total, "categories": len(categories)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("jev_obj/data/datasets/ovon/hm3d_ovon")
    )
    parser.add_argument(
        "--destination", type=Path,
        default=Path("jev_obj/data/datasets/objectnav/ovon_compat"),
    )
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=SPLITS)
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve():
        parser.error("source and destination must differ")
    result = {
        split: convert_split(args.source, args.destination, split) for split in args.splits
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
