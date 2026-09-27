import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jev_obj"))

from vlm.detector.detections import ObjectDetections
from vlm.itm.blip2itm import BLIP2ITMClient
from vlm.utils import get_object_utils


class PerceptionMetadataTests(unittest.TestCase):
    def test_fallback_detection_is_marked_unavailable(self):
        detections = ObjectDetections.from_json(
            {
                "boxes": [],
                "logits": [],
                "phrases": [],
                "available": False,
                "fallback": True,
                "backend": "grounding_dino_fallback",
            }
        )

        self.assertFalse(detections.available)
        self.assertTrue(detections.fallback)
        self.assertEqual(detections.backend, "grounding_dino_fallback")

    def test_itm_fallback_metadata_is_not_lost(self):
        response = {
            "response": 0.5,
            "available": False,
            "fallback": True,
            "backend": "itm_fixed_fallback",
        }
        with patch("vlm.itm.blip2itm.send_request", return_value=response):
            score, metadata = BLIP2ITMClient().cosine_with_metadata(
                np.zeros((2, 2, 3), dtype=np.uint8), "target"
            )

        self.assertEqual(score, 0.5)
        self.assertFalse(metadata["available"])
        self.assertTrue(metadata["fallback"])
        self.assertEqual(metadata["backend"], "itm_fixed_fallback")

    def test_missing_metadata_is_unverified(self):
        detections = ObjectDetections.from_json(
            {"boxes": [], "logits": [], "phrases": []}
        )
        self.assertFalse(detections.available)
        self.assertEqual(detections.backend, "unknown_unverified")
        with patch("vlm.itm.blip2itm.send_request", return_value={"response": 0.5}):
            _, metadata = BLIP2ITMClient().cosine_with_metadata(
                np.zeros((2, 2, 3), dtype=np.uint8), "target"
            )
        self.assertFalse(metadata["available"])
        self.assertFalse(metadata["metadata_verified"])
        self.assertEqual(metadata["backend"], "unknown_unverified")

    def test_related_detector_timeout_preserves_target_result(self):
        cfg = SimpleNamespace(
            yolo=SimpleNamespace(
                agnostic_nms=False, confidence_threshold_yolo=0.2,
                iou_threshold_yolo=0.5,
            ),
            groundingDINO=SimpleNamespace(
                confidence_threshold_dino=0.2, text_threshold=0.2,
            ),
        )
        target = ObjectDetections.from_json({
            "boxes": [[0.1, 0.1, 0.5, 0.5]], "logits": [0.9],
            "phrases": ["chair"], "available": True, "fallback": False,
            "backend": "yolov7",
        })
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch.object(get_object_utils.yolov7_detector, "predict", return_value=target), \
             patch.object(get_object_utils.dino_detector, "predict", side_effect=TimeoutError()), \
             patch.object(
                 get_object_utils.sam_segmentor, "segment_bbox",
                 return_value=np.ones((480, 640), dtype=np.uint8),
             ):
            _, scores, masks, labels, status = get_object_utils.get_object(
                "chair", image, cfg, ["cabinet"], return_metadata=True
            )
        self.assertEqual(len(scores), 1)
        self.assertEqual(len(masks), 1)
        self.assertEqual(labels, [0])
        self.assertTrue(status["target_available"])
        self.assertFalse(status["related_available"])
        self.assertEqual(status["label_detection_valid"], [True, False])
        self.assertEqual(status["detectors"][1]["error"], "TimeoutError")

    def test_empty_segmentation_is_zero_evidence_not_service_unavailable(self):
        cfg = SimpleNamespace(
            yolo=SimpleNamespace(
                agnostic_nms=False, confidence_threshold_yolo=0.2,
                iou_threshold_yolo=0.5,
            ),
            groundingDINO=SimpleNamespace(
                confidence_threshold_dino=0.2, text_threshold=0.2,
            ),
        )
        target = ObjectDetections.from_json({
            "boxes": [[0.1, 0.1, 0.5, 0.5]], "logits": [0.9],
            "phrases": ["kitchen appliance"], "available": True,
            "fallback": False, "backend": "grounding_dino",
        })
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch.object(get_object_utils.dino_detector, "predict", return_value=target), \
             patch.object(
                 get_object_utils.sam_segmentor, "segment_bbox",
                 return_value=np.zeros((480, 640), dtype=np.uint8),
             ):
            _, scores, masks, labels, status = get_object_utils.get_object(
                "kitchen appliance", image, cfg, [], return_metadata=True
            )
        self.assertEqual((scores, masks, labels), ([], [], []))
        self.assertTrue(status["target_available"])
        self.assertTrue(status["related_available"])
        self.assertEqual(status["label_detection_valid"], [False])
        self.assertEqual(status["valid_mask_count"], 0)


if __name__ == "__main__":
    unittest.main()
