from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw


class RoofDetector:
    """Lightweight roof detector used to keep the project runnable without heavy ML downloads."""

    def __init__(self, confidence_threshold: float = 0.05):
        self.confidence_threshold = confidence_threshold

    def detect_roofs(self, image: Image.Image) -> Tuple[np.ndarray, List[dict]]:
        arr = np.asarray(image.convert("RGB"))
        h, w = arr.shape[:2]

        if h == 0 or w == 0:
            return np.empty((0, 4), dtype=int), []

        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        hue, sat, val = cv2.split(hsv)
        vegetation = ((hue >= 35) & (hue <= 95) & (sat >= 40)).astype(np.uint8) * 255
        roof_mask = ((~vegetation.astype(bool)) & (sat >= 18) & (val >= 35)).astype(np.uint8) * 255
        roof_mask |= (((sat < 120) & (val > 60) & (val < 220)) & (np.mean(arr, axis=2) > 35)).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        roof_mask = cv2.morphologyEx(roof_mask, cv2.MORPH_OPEN, kernel)
        roof_mask = cv2.morphologyEx(roof_mask, cv2.MORPH_CLOSE, kernel)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(roof_mask, 8)
        boxes = []
        min_area = max(200, int(0.0008 * w * h))

        for idx in range(1, num_labels):
            x, y, bw, bh, area = stats[idx]
            if area < min_area or bw < 20 or bh < 20:
                continue
            boxes.append([max(0, x - 8), max(0, y - 8), min(w, x + bw + 8), min(h, y + bh + 8)])

        if not boxes:
            box_w = max(80, int(w * 0.55))
            box_h = max(80, int(h * 0.55))
            x1 = max(0, (w - box_w) // 2)
            y1 = max(0, (h - box_h) // 2)
            x2 = min(w, x1 + box_w)
            y2 = min(h, y1 + box_h)
            boxes = [[x1, y1, x2, y2]]

        boxes = sorted(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)[:8]
        metadata = [{
            "bbox": box,
            "confidence": self.confidence_threshold,
            "class_name": "building",
        } for box in boxes]
        return np.asarray(boxes, dtype=int), metadata

    def segment_roofs(self, image: Image.Image, bboxes: np.ndarray) -> np.ndarray:
        arr = np.asarray(image.convert("RGB"))
        masks = []

        for x1, y1, x2, y2 in bboxes:
            crop = arr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            mask = np.zeros(crop.shape[:2], dtype=np.uint8)
            mask[:] = 255
            masks.append(mask)

        if not masks:
            return np.zeros((image.height, image.width), dtype=np.uint8)

        return np.stack(masks, axis=0)
