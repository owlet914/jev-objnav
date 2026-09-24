"""Extract only OVON validation metadata from the supplied archive.

The train split is large and is not needed for an evaluation pilot. This
extractor does not fetch HM3D meshes or any ground-truth scene geometry.
"""

from __future__ import annotations

import argparse
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath


SPLITS = ("val_seen", "val_seen_synonyms", "val_unseen")
MAX_FILE_BYTES = 128 * 1024 * 1024


def extract_validation_archive(
    archive: Path, destination: Path, splits: tuple[str, ...] = SPLITS
) -> dict[str, int]:
    counts = {split: 0 for split in splits}
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            if member.is_dir():
                continue
            path = PurePosixPath(member.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or len(path.parts) not in (3, 4)
                or path.parts[0] != "hm3d_ovon"
                or path.parts[1] not in splits
                or path.suffixes[-2:] != [".json", ".gz"]
                or (len(path.parts) == 4 and path.parts[2] != "content")
            ):
                continue
            mode = (member.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                raise ValueError(f"symbolic link in archive: {member.filename}")
            if member.file_size > MAX_FILE_BYTES:
                raise ValueError(f"validation shard too large: {member.filename}")
            output = destination.joinpath(*path.parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as stream, output.open("wb") as target:
                shutil.copyfileobj(stream, target)
            counts[path.parts[1]] += 1
    for split in splits:
        split_dir = destination / "hm3d_ovon" / split
        if not list(split_dir.glob("*.json.gz")) or not list(
            (split_dir / "content").glob("*.json.gz")
        ):
            raise ValueError(f"validation split incomplete: {split}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("hm3d_ovon.zip"))
    parser.add_argument(
        "--destination", type=Path, default=Path("jev_obj/data/datasets/ovon")
    )
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=SPLITS)
    args = parser.parse_args()
    print(extract_validation_archive(args.archive, args.destination, tuple(args.splits)))


if __name__ == "__main__":
    main()
