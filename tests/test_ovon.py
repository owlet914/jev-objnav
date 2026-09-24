import tempfile
import unittest
import zipfile
from pathlib import Path

from tools.extract_ovon_val import extract_validation_archive
from tools.prepare_ovon_habitat031 import convert_split, load_gzip, write_gzip


class OvonConversionTests(unittest.TestCase):
    def test_validation_extraction_excludes_train_and_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "ovon.zip"
            destination = Path(tmp) / "extracted"
            with zipfile.ZipFile(archive, "w") as source:
                for split in ("val_seen", "val_seen_synonyms", "val_unseen"):
                    source.writestr(f"hm3d_ovon/{split}/{split}.json.gz", b"stub")
                    source.writestr(f"hm3d_ovon/{split}/content/scene.json.gz", b"shard")
                source.writestr("hm3d_ovon/train/content/scene.json.gz", b"train")
                source.writestr("hm3d_ovon/val_seen/../../escaped.json.gz", b"unsafe")

            self.assertEqual(
                extract_validation_archive(archive, destination),
                {"val_seen": 2, "val_seen_synonyms": 2, "val_unseen": 2},
            )
            self.assertFalse((destination / "hm3d_ovon" / "train").exists())
            self.assertFalse((destination / "escaped.json.gz").exists())

    def test_keeps_scoring_goals_and_removes_unsupported_episode_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            destination = Path(tmp) / "destination"
            split = "val_unseen"
            shard = source / split / "content" / "scene.json.gz"
            original = {
                "goals_by_category": {
                    "scene.basis.glb_mug": [{
                        "object_category": "mug",
                        "object_id": "mug_1",
                        "position": [1, 0, 2],
                        "view_points": [],
                    }]
                },
                "episodes": [{
                    "episode_id": "42",
                    "scene_id": "hm3d/val/scene.basis.glb",
                    "object_category": "mug",
                    "children_object_categories": ["coffee mug"],
                    "goals": [],
                    "info": {},
                }],
            }
            write_gzip(shard, original)

            self.assertEqual(convert_split(source, destination, split)["episodes"], 1)
            converted = load_gzip(destination / split / "content" / shard.name)
            episode = converted["episodes"][0]
            self.assertNotIn("children_object_categories", episode)
            self.assertEqual(episode["info"]["children_object_categories"], ["coffee mug"])
            self.assertEqual(episode["info"]["ovon_episode_id"], "42")
            self.assertEqual(converted["goals_by_category"], original["goals_by_category"])
            self.assertEqual(converted["category_to_task_category_id"], {"mug": 0})


if __name__ == "__main__":
    unittest.main()
