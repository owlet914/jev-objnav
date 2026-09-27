"""Read-only production-code probes for the independent information-parity review.

Only detector/segmentor/HTTP external boundaries are mocked. No Jev call, ROS
navigation, dataset reads, or application-source edits are performed.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "jev_obj"))
import numpy as np
from nav_jev_bridge.bridge import SnapshotError, prepare_decision
from nav_jev_bridge.server import make_handler
from vlm.detector.detections import ObjectDetections
from vlm.itm.blip2itm import BLIP2ITMClient
from vlm.utils import get_object_utils as detection_module


def run():
    results = {}
    snapshot = json.loads((ROOT / "examples/jev_parity/cpp_snapshot.json").read_text(encoding="utf-8"))
    prepared = prepare_decision(snapshot, "full")
    class RecordingProvider:
        def choose_request(self, payload):
            self.payload = payload
            choice = next(iter(payload["questions"]["next_goal"]["criteria"]))
            return choice, 1.0, {choice: 1.0}
    provider = RecordingProvider()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(provider, 0.4, 0.5))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    started = time.perf_counter()
    try:
        body = json.dumps(snapshot, allow_nan=False).encode("utf-8")
        request = Request(f"http://127.0.0.1:{server.server_port}/decide", data=body,
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=5) as response:
            decision = json.load(response)
        assert provider.payload["state"] == snapshot
        assert provider.payload["questions"]["next_goal"]["criteria"] == prepared.options
        results["production_http_transport"] = {
            "status": decision["status"], "request_bytes": len(body),
            "state_equals_fixture": True, "criteria_equals_prepared_options": True,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "provider": "recording mock, not Jev",
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    results["fixture_completeness"] = {
        "accepted_by_production_prepare_decision": True,
        "claims_complete": prepared.state["coverage"]["high_level_evidence_complete"],
        "missing_top_level_evidence": [k for k in ("original_policy", "search_branches", "perception", "navigation_history") if k not in prepared.state],
        "frontiers_without_raw_neighborhood": sum("semantic_features" not in f for f in snapshot["frontiers"]),
        "objects_without_normal_path_search": sum("normal_path_search" not in o for o in snapshot["objects"]),
    }
    contradictory = copy.deepcopy(snapshot)
    contradictory["map"]["observation_id"] = 999
    mismatch_accepted = True
    mismatch_error = None
    try:
        prepare_decision(contradictory, "full")
    except SnapshotError as exc:
        mismatch_accepted = False
        mismatch_error = exc.code
    results["mismatched_map_observation"] = {
        "top_observation_id": contradictory["observation_id"],
        "map_observation_id": contradictory["map"]["observation_id"],
        "accepted_by_production_prepare_decision": mismatch_accepted,
        "error_code": mismatch_error,
    }
    legacy = ObjectDetections.from_json({"boxes": [], "logits": [], "phrases": []})
    with patch("vlm.itm.blip2itm.send_request", return_value={"response": 0.5}):
        score, metadata = BLIP2ITMClient().cosine_with_metadata(np.zeros((2, 2, 3), dtype=np.uint8), "target")
    results["missing_metadata_promoted_to_valid"] = {
        "detector_available": legacy.available, "detector_backend": legacy.backend,
        "itm_score": score, "itm_metadata": metadata,
    }
    cfg = SimpleNamespace(
        yolo=SimpleNamespace(agnostic_nms=False, confidence_threshold_yolo=0.2, iou_threshold_yolo=0.5),
        groundingDINO=SimpleNamespace(confidence_threshold_dino=0.2, text_threshold=0.2),
    )
    yolo = ObjectDetections.from_json({
        "boxes": [[0.1, 0.1, 0.5, 0.5]], "logits": [0.9], "phrases": ["chair"],
        "available": True, "fallback": False, "backend": "yolov7",
    })
    dino = ObjectDetections.from_json({
        "boxes": [], "logits": [], "phrases": [],
        "available": False, "fallback": True, "backend": "grounding_dino_fallback",
    })
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    with patch.object(detection_module.yolov7_detector, "predict", return_value=yolo), \
         patch.object(detection_module.dino_detector, "predict", return_value=dino), \
         patch.object(detection_module.sam_segmentor, "segment_bbox", return_value=np.ones((480, 640), dtype=np.uint8)):
        *_, status = detection_module.get_object("chair", image, cfg, ["cabinet"], return_metadata=True)
    results["real_target_related_fallback"] = status
    with patch.object(detection_module.yolov7_detector, "predict", return_value=yolo), \
         patch.object(detection_module.dino_detector, "predict", side_effect=TimeoutError("synthetic related-detector timeout")), \
         patch.object(detection_module.sam_segmentor, "segment_bbox", return_value=np.ones((480, 640), dtype=np.uint8)):
        *_, timeout_status = detection_module.get_object(
            "chair", image, cfg, ["cabinet"], return_metadata=True
        )
        results["related_timeout"] = {
            "raises": False,
            "target_results_returned": timeout_status["target_match_count"] > 0,
            "target_available": timeout_status["target_available"],
            "related_available": timeout_status["related_available"],
            "detectors": timeout_status["detectors"],
        }
    sources = [
        "nav_jev_bridge/bridge.py", "jev_obj/src/planner/exploration_manager/src/jev_integration.cpp",
        "jev_obj/src/planner/plan_env/src/map_ros.cpp", "jev_obj/vlm/utils/get_object_utils.py",
        "jev_obj/vlm/detector/detections.py", "jev_obj/vlm/itm/blip2itm.py",
        "tools/jev_cpp_contract_fixture.cpp",
    ]
    results["reviewed_source_sha256"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources
    }
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
