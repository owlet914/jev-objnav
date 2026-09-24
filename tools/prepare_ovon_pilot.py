"""Make fixed small OVON development sets and extract only their HM3D scenes.

Only val_seen is read. The val_unseen split is reserved for final evaluation.
Ground-truth goals remain in Habitat episode files for scoring, never in Jev input.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from pathlib import Path, PurePosixPath

from tools.prepare_ovon_habitat031 import load_gzip, write_gzip


SCENES = {
    "pilot": ("MHPLjHsuG27", "QaLdnwvtxbs"),
    "dev": ("svBbv1Pavdk", "Nfvxx8J5NCo"),
}
EPISODES_PER_SCENE = 12
SEED = 20260924


def make_small_splits(source: Path, destination: Path) -> dict:
    root = load_gzip(source / "val_seen" / "val_seen.json.gz")
    manifest = {
        "source_split": "val_seen",
        "final_test_split": "val_unseen",
        "seed": SEED,
        "episodes_per_scene": EPISODES_PER_SCENE,
        "splits": {},
    }
    for split, scene_names in SCENES.items():
        selected = []
        for scene_name in scene_names:
            shard = source / "val_seen" / "content" / f"{scene_name}.json.gz"
            data = load_gzip(shard)
            episodes = sorted(data["episodes"], key=lambda ep: str(ep["episode_id"]))
            if len(episodes) < EPISODES_PER_SCENE:
                raise ValueError(f"not enough episodes in {shard}")
            rng = random.Random(f"{SEED}:{scene_name}")
            chosen = sorted(
                rng.sample(episodes, EPISODES_PER_SCENE),
                key=lambda ep: str(ep["episode_id"]),
            )
            goal_keys = {
                f"{Path(ep['scene_id']).name}_{ep['object_category']}" for ep in chosen
            }
            missing = goal_keys - data["goals_by_category"].keys()
            if missing:
                raise ValueError(f"missing scoring goals in {shard}: {sorted(missing)}")
            data["episodes"] = chosen
            data["goals_by_category"] = {
                key: data["goals_by_category"][key] for key in sorted(goal_keys)
            }
            write_gzip(destination / split / "content" / shard.name, data)
            selected.append({
                "scene": scene_name,
                "episode_ids": [ep["info"]["ovon_episode_id"] for ep in chosen],
                "categories": sorted({ep["object_category"] for ep in chosen}),
            })
        write_gzip(destination / split / f"{split}.json.gz", root)
        manifest["splits"][split] = selected
    return manifest


def extract_selected_scenes(archive: Path, destination: Path) -> dict[str, int]:
    wanted = {scene for scenes in SCENES.values() for scene in scenes}
    counts = {scene: 0 for scene in wanted}
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            if member.is_dir():
                continue
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts or path.parts[0] != "hm3d":
                continue
            is_config = member.filename.endswith("scene_dataset_config.json") and (
                len(path.parts) == 2 or (len(path.parts) == 3 and path.parts[1] == "val")
            )
            is_scene = (
                len(path.parts) == 4
                and path.parts[1] == "val"
                and path.parts[2].split("-", 1)[-1] in wanted
                and path.suffix in (".glb", ".navmesh", ".txt")
            )
            if not (is_config or is_scene):
                continue
            output = destination.joinpath(*path.parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as stream, output.open("wb") as target:
                shutil.copyfileobj(stream, target)
            if is_scene:
                counts[path.parts[2].split("-", 1)[-1]] += 1
    for scene, count in counts.items():
        if count < 4:
            raise ValueError(f"incomplete scene assets for {scene}: {count} files")
    config = destination / "hm3d/hm3d_annotated_basis.scene_dataset_config.json"
    if not config.is_file():
        raise ValueError("HM3D annotated scene configuration is missing")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("hm3d.zip"))
    parser.add_argument(
        "--source", type=Path,
        default=Path("jev_obj/data/datasets/objectnav/ovon_compat"),
    )
    parser.add_argument(
        "--dataset-output", type=Path,
        default=Path("jev_obj/data/datasets/objectnav/ovon_small"),
    )
    parser.add_argument(
        "--scene-output", type=Path, default=Path("jev_obj/data/scene_datasets")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("examples/ovon_experiment_split.json")
    )
    args = parser.parse_args()
    manifest = make_small_splits(args.source, args.dataset_output)
    manifest["scene_assets"] = extract_selected_scenes(args.archive, args.scene_output)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"episode_counts": {
        split: sum(len(scene["episode_ids"]) for scene in scenes)
        for split, scenes in manifest["splits"].items()
    }, "scene_assets": manifest["scene_assets"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
