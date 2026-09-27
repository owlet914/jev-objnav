import cv2
import json
import numpy as np
from vlm.coco_classes import COCO_CLASSES
from vlm.detector.yolov7 import YOLOv7Client
from vlm.segmentor.sam import MobileSAMClient
from vlm.detector.grounding_dino import GroundingDINOClient
from vlm.itm.blip2itm import BLIP2ITMClient
from vlm.utils.get_itm_message import get_itm_message

yolov7_detector = YOLOv7Client(port=12184)
blip2_itm = BLIP2ITMClient(port=12182)
sam_segmentor = MobileSAMClient(port=12183)
dino_detector = GroundingDINOClient(port=12181)


def get_segmentation(segmented_img, idx, detections, img, label, score, color):
    object_mask = np.zeros((480, 640), dtype=np.uint8)
    bbox_denorm = detections.boxes[idx] * np.array(
        [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
    )
    x1, y1, x2, y2 = [int(v) for v in bbox_denorm]
    bbox_area = (x2 - x1) * (y2 - y1)
    img_area = img.shape[0] * img.shape[1]

    if bbox_area / img_area < 0.99:
        object_mask = sam_segmentor.segment_bbox(img, bbox_denorm.tolist())
        contours, _ = cv2.findContours(
            object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            cv2.drawContours(segmented_img, [contour], 0, color, 4)

        cv2.rectangle(
            segmented_img,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        label_text = f"{label} ({score:.2f})"
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2
        )
        label_x = x1
        label_y = y1 - text_height
        cv2.rectangle(
            segmented_img,
            (label_x, label_y - 30),
            (label_x + text_width, label_y + text_height),
            color,
            2,
        )
        cv2.putText(
            segmented_img,
            label_text,
            (label_x, label_y),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (255, 255, 255),
            1,
        )

    return segmented_img, object_mask

def get_object(right_label, img, cfg, similar_answer, return_metadata=False):
    score_list = []
    object_masks_list = []
    segmented_img = img.copy()
    label_list = []
    coco_label = []
    dino_label = []
    perception = {
        "available": False,
        "fallback": False,
        "target_available": False,
        "related_available": False,
        "detectors": [],
    }

    def record_detector(name, detections, requested_classes, error=None):
        status = {
            "name": name,
            "backend": str(getattr(detections, "backend", "unavailable")),
            "available": bool(getattr(detections, "available", False)),
            "fallback": bool(getattr(detections, "fallback", False)),
            "requested_classes": list(requested_classes),
            "target_coverage": any(item in right_label_list for item in requested_classes),
            "raw_detection_count": len(detections.logits) if detections is not None else 0,
            "target_match_count": 0,
            "related_match_count": 0,
            "valid_mask_count": 0,
            "segmentation_error_count": 0,
            "invalid_classes": [],
            "error": error,
        }
        perception["detectors"].append(status)
        perception["available"] = perception["available"] or (
            status["available"] and not status["fallback"]
        )
        perception["fallback"] = perception["fallback"] or status["fallback"]
        return status

    def process_detections(detections, status):
        for idx in range(len(detections.logits)):
            label_detected = detections.phrases[idx]
            if label_detected in right_label_list:
                label_index = 0
                color = (255, 0, 0)
                count_key = "target_match_count"
            elif label_detected in all_answer:
                label_index = list(all_answer).index(label_detected) - len(right_label_list) + 1
                if label_index <= 0:
                    continue
                color = (0, 255, 0)
                count_key = "related_match_count"
            else:
                continue
            score = detections.logits[idx].item()
            try:
                segmented, object_mask = get_segmentation(
                    segmented_img, idx, detections, img, label_detected, score, color=color
                )
            except Exception as exc:
                status["segmentation_error_count"] += 1
                status["invalid_classes"].append(label_detected)
                status["error"] = status["error"] or f"segmentation:{type(exc).__name__}"
                continue
            segmented_img[:] = segmented
            if not np.any(object_mask):
                status["invalid_classes"].append(label_detected)
                continue
            score_list.append(score)
            object_masks_list.append(object_mask)
            label_list.append(label_index)
            status[count_key] += 1
            status["valid_mask_count"] += int(bool(np.any(object_mask)))
    right_label_list = list(map(str.strip, right_label.split('|')))
    # print(f"right_label_list: {right_label_list}")
    all_answer = right_label_list + similar_answer
    for label in all_answer:
        if label in COCO_CLASSES:
            coco_label.append(label)
        else:
            dino_label.append(label)

    if any(item in dino_label for item in right_label_list):
        dino_label = all_answer
        coco_label = []
        for label in right_label_list:
            if label in COCO_CLASSES:
                coco_label.append(label)

    if coco_label:
        try:
            detections = yolov7_detector.predict(
                img, agnostic_nms=cfg.yolo.agnostic_nms,
                conf_thres=cfg.yolo.confidence_threshold_yolo,
                iou_thres=cfg.yolo.iou_threshold_yolo,
            )
            status = record_detector("yolov7", detections, coco_label)
            process_detections(detections, status)
        except Exception as exc:
            record_detector("yolov7", None, coco_label, type(exc).__name__)

    if dino_label:
        caption = ' '.join(f'{item}.  ' for item in dino_label)
        try:
            detections = dino_detector.predict(
                img, caption=caption,
                box_threshold=cfg.groundingDINO.confidence_threshold_dino,
                text_threshold=cfg.groundingDINO.text_threshold,
            )
            status = record_detector("grounding_dino", detections, dino_label)
            process_detections(detections, status)
        except Exception as exc:
            record_detector("grounding_dino", None, dino_label, type(exc).__name__)

    real_detectors = [
        item for item in perception["detectors"]
        if item["available"] and not item["fallback"]
    ]
    def class_served(label):
        return any(
            label in item["requested_classes"]
            for item in real_detectors
        )

    def class_evidence_valid(label):
        return class_served(label) and not any(
            label in item["invalid_classes"]
            for item in real_detectors
        )

    perception["target_available"] = bool(right_label_list) and all(
        class_served(label) for label in right_label_list
    )
    related_requested = any(
        item not in right_label_list for item in all_answer
    )
    perception["related_available"] = (
        all(class_served(label) for label in similar_answer)
        if related_requested else True
    )
    perception["label_detection_valid"] = [
        bool(right_label_list) and all(class_evidence_valid(label) for label in right_label_list)
    ] + [
        class_evidence_valid(label) for label in similar_answer
    ]
    perception["target_match_count"] = sum(
        item["target_match_count"] for item in perception["detectors"]
    )
    perception["related_match_count"] = sum(
        item["related_match_count"] for item in perception["detectors"]
    )
    perception["valid_mask_count"] = sum(
        item["valid_mask_count"] for item in perception["detectors"]
    )

    result = (segmented_img, score_list, object_masks_list, label_list)
    return (*result, perception) if return_metadata else result

def get_object_with_itm(label, img, cfg):
    score_list = []
    object_masks_list = []
    cosine_list = []
    itm_score_list = []
    segmented_img = img.copy()
    if label in COCO_CLASSES:
        detections = yolov7_detector.predict(img, agnostic_nms=cfg.yolo.agnostic_nms,
                                             conf_thres=cfg.yolo.confidence_threshold_yolo, iou_thres=cfg.yolo.iou_threshold_yolo)
        for idx in range(len(detections.logits)):
            label_detected = detections.phrases[idx]
            score = detections.logits[idx].item()
            if detections.phrases[idx] == label:
                segmented_img, object_mask = get_segmentation(
                    segmented_img, idx, detections, img, label_detected, score, color=(255, 0, 0)
                )
                img_detected = crop_and_expand_box(img, detections, idx)
                # cv2.imshow(f"img_detected{idx}", img_detected)
                cosine, itm_score = get_itm_message(img_detected, label)
                print(f"cosine: {cosine:.3f}, itm_score: {itm_score:.3f}")
                score_list.append(score)
                object_masks_list.append(object_mask)
                cosine_list.append(cosine)
                itm_score_list.append(itm_score)

    else:
        detections = dino_detector.predict(img, caption=label, 
                                           box_threshold=cfg.groundingDINO.confidence_threshold_dino, text_threshold=cfg.groundingDINO.text_threshold)
        for idx in range(len(detections.logits)):
            label_detected = detections.phrases[idx]
            score = detections.logits[idx].item()
            if score > cfg.groundingDINO.confidence_threshold_dino:
                segmented_img, object_mask = get_segmentation(
                    segmented_img, idx, detections, img, label_detected, score, color=(255, 0, 0)
                )
                score_list.append(score)
                object_masks_list.append(object_mask)
                img_detected = crop_and_expand_box(img, detections, idx)
                # cv2.imshow(f"img_detected{idx}", img_detected)
                cosine, itm_score = get_itm_message(img_detected, label)
                print(f"cosine: {cosine}, itm_score: {itm_score}")
                cosine_list.append(cosine)
                itm_score_list.append(itm_score)
    
    return segmented_img, score_list, object_masks_list, cosine_list, itm_score_list


def crop_and_expand_box(img, detections, idx, expand_pixels=0.4):
    # Get bounding box coordinates in [x_min, y_min, x_max, y_max] format
    x_min, y_min, x_max, y_max = detections.boxes[idx]
    x_min = int(x_min * img.shape[1])
    y_min = int(y_min * img.shape[0])
    x_max = int(x_max * img.shape[1])
    y_max = int(y_max * img.shape[0])

    # Expand the box outward; clamp to image boundaries
    x_min = max(int(x_min*(1-expand_pixels)), 0)
    y_min = max(int(y_min*(1-expand_pixels)), 0)
    x_max = min(int(x_max*(1+expand_pixels)), img.shape[1] - 1)
    y_max = min(int(y_max*(1+expand_pixels)), img.shape[0] - 1)

    # Crop the image to keep only the box region
    img_detected = img[y_min:y_max+1, x_min:x_max+1]

    return img_detected
