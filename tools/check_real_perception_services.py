"""Run one real inference through every local perception service.

The command fails if a detector reports fallback/unavailable provenance or if
an expected response cannot be decoded. It is a preflight check, not an
accuracy benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import cv2


ROOT = Path(__file__).resolve().parents[1]
JEV_OBJ = ROOT / "jev_obj"
sys.path.insert(0, str(JEV_OBJ))

from vlm.detector.grounding_dino import GroundingDINOClient  # noqa: E402
from vlm.detector.yolov7 import YOLOv7Client  # noqa: E402
from vlm.itm.blip2itm import BLIP2ITMClient  # noqa: E402
from vlm.segmentor.sam import MobileSAMClient  # noqa: E402


def timed(call: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    value = call()
    return value, round((time.perf_counter() - started) * 1000, 3)


def require_real_detection(name: str, detection: Any, backend: str) -> None:
    if detection.available is not True:
        raise RuntimeError(f"{name} unavailable")
    if detection.fallback is not False:
        raise RuntimeError(f"{name} used fallback")
    if detection.backend != backend:
        raise RuntimeError(f"{name} backend={detection.backend!r}, expected {backend!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", type=Path, default=JEV_OBJ / "examples" / "ovon_episode_step.png"
    )
    args = parser.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)
    height, width = image.shape[:2]

    gdino, gdino_ms = timed(
        lambda: GroundingDINOClient().predict(
            image, "chair . table . sofa .", box_threshold=0.25, text_threshold=0.20
        )
    )
    require_real_detection("GroundingDINO", gdino, "grounding_dino")

    (itm_score, itm_metadata), itm_ms = timed(
        lambda: BLIP2ITMClient().cosine_with_metadata(image, "a chair")
    )
    if itm_metadata.get("available") is not True or itm_metadata.get("fallback") is not False:
        raise RuntimeError(f"BLIP2 invalid provenance: {itm_metadata}")
    if itm_metadata.get("backend") != "blip2_itm":
        raise RuntimeError(f"BLIP2 backend={itm_metadata.get('backend')!r}")

    margin_x = max(1, width // 4)
    margin_y = max(1, height // 4)
    bbox = [margin_x, margin_y, width - margin_x, height - margin_y]
    mask, sam_ms = timed(lambda: MobileSAMClient().segment_bbox(image, bbox))
    if mask.shape != (height, width):
        raise RuntimeError(f"MobileSAM mask shape={mask.shape}, expected {(height, width)}")

    yolo, yolo_ms = timed(lambda: YOLOv7Client().predict(image))
    require_real_detection("YOLOv7", yolo, "yolov7")

    print(json.dumps({
        "image": str(args.image),
        "image_shape": list(image.shape),
        "services": {
            "grounding_dino": {
                "available": gdino.available,
                "fallback": gdino.fallback,
                "backend": gdino.backend,
                "detections": gdino.num_detections,
                "phrases": gdino.phrases,
                "latency_ms": gdino_ms,
            },
            "blip2_itm": {
                **itm_metadata,
                "raw_score": itm_score,
                "latency_ms": itm_ms,
            },
            "mobile_sam": {
                "mask_shape": list(mask.shape),
                "foreground_pixels": int(mask.sum()),
                "latency_ms": sam_ms,
            },
            "yolov7": {
                "available": yolo.available,
                "fallback": yolo.fallback,
                "backend": yolo.backend,
                "detections": yolo.num_detections,
                "phrases": yolo.phrases,
                "latency_ms": yolo_ms,
            },
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
