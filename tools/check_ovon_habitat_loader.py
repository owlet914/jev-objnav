"""Load only pilot/dev episodes using Habitat-Lab's actual ObjectNav-v1 loader.

The reserved val_unseen split is intentionally not read. Ground-truth goals
stay inside Habitat's dataset for scoring and are not returned in this report.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from habitat.datasets.object_nav.object_nav_dataset import ObjectNavDatasetV1


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
        "--output", type=Path, default=Path("examples/ovon_loader_check.json")
    )
    args = parser.parse_args()
    report = {}
    for split in ("pilot", "dev"):
        dataset = ObjectNavDatasetV1()
        files = [args.datasets / split / f"{split}.json.gz"] + sorted(
            (args.datasets / split / "content").glob("*.json.gz")
        )
        for path in files:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                dataset.from_json(stream.read(), scenes_dir=str(args.scenes.resolve()))
        report[split] = {
            "episodes": len(dataset.episodes),
            "scenes": len({episode.scene_id for episode in dataset.episodes}),
            "categories": len({episode.object_category for episode in dataset.episodes}),
            "all_goals_loaded": all(bool(episode.goals) for episode in dataset.episodes),
            "all_meshes_exist": all(Path(episode.scene_id).is_file() for episode in dataset.episodes),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
