"""Exercise the real GroundingDINO HTTP service sequentially and concurrently."""

from __future__ import annotations

import argparse
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import requests


def build_payload(image_path: Path, caption: str) -> dict:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError(f"could not encode {image_path}")
    return {
        "image": base64.b64encode(encoded.tobytes()).decode("ascii"),
        "caption": caption,
        "box_threshold": 0.25,
        "text_threshold": 0.20,
    }


def request_once(url: str, payload: dict, timeout_s: float) -> dict:
    started = time.perf_counter()
    response = requests.post(url, json=payload, timeout=timeout_s)
    response.raise_for_status()
    body = response.json()
    if body.get("available") is not True or body.get("fallback") is not False:
        raise RuntimeError(f"invalid provenance: {body}")
    if body.get("backend") != "grounding_dino":
        raise RuntimeError(f"unexpected backend: {body.get('backend')!r}")
    for key in ("boxes", "logits", "phrases"):
        if key not in body:
            raise RuntimeError(f"missing response field: {key}")
    return {
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "detections": len(body["phrases"]),
        "phrases": body["phrases"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:12181/gdino")
    parser.add_argument("--image", type=Path, default=Path("examples/ovon_episode_step.png"))
    parser.add_argument("--caption", default="table . chair . sofa .")
    parser.add_argument("--sequential", type=int, default=4)
    parser.add_argument("--concurrent", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    payload = build_payload(args.image, args.caption)
    sequential = [request_once(args.url, payload, args.timeout) for _ in range(args.sequential)]
    concurrent = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(request_once, args.url, payload, args.timeout)
            for _ in range(args.concurrent)
        ]
        for future in as_completed(futures):
            concurrent.append(future.result())

    all_results = sequential + concurrent
    print(json.dumps({
        "service": "real_grounding_dino",
        "sequential_requests": len(sequential),
        "concurrent_requests": len(concurrent),
        "workers": args.workers,
        "successful_requests": len(all_results),
        "latency_ms": {
            "min": min(item["latency_ms"] for item in all_results),
            "max": max(item["latency_ms"] for item in all_results),
            "mean": round(sum(item["latency_ms"] for item in all_results) / len(all_results), 3),
        },
        "detections_per_request": [item["detections"] for item in all_results],
        "sample_phrases": all_results[0]["phrases"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
