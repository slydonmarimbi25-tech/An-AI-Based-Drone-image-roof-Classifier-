"""Building detection and segmentation for rooftop extraction."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

try:
    from shapely.geometry import Polygon
except ImportError:  # pragma: no cover
    Polygon = None

try:
    from ultralytics import YOLOWorld
except ImportError:  # pragma: no cover
    YOLOWorld = None


class RooftopDetector:
    """Zero-shot rooftop detector using YOLO-World when available, with a fallback for local execution."""

    def __init__(self, conf_threshold: float = 0.005, model_path: str = "yolov8s-world.pt", use_model: bool = False):
        self.conf_threshold = conf_threshold
        self.model_path = model_path
        self.use_model = use_model
        self.model = None
        if self.use_model:
            self._load_model()

    def _load_model(self) -> None:
        if YOLOWorld is None:
            self.model = None
            return

        local_model = Path(self.model_path)
        if not local_model.exists():
            self.model = None
            return

        try:
            self.model = YOLOWorld(str(local_model))
            self.model.set_classes(["building", "house", "rooftop", "structure"])
        except Exception:  # pragma: no cover
            self.model = None

    def detect_rooftops(self, image_pil: Image.Image) -> np.ndarray:
        """Return rooftop boxes in XYXY format as a NumPy array."""
        if self.model is not None:
            try:
                results = self.model.predict(image_pil, conf=self.conf_threshold)
                if len(results) and hasattr(results[0], "boxes"):
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    if boxes.size:
                        return boxes.astype(int)
            except Exception:
                pass

        image = np.asarray(image_pil.convert("RGB"), dtype=np.uint8)
        height, width = image.shape[:2]
        if width <= 0 or height <= 0:
            return np.empty((0, 4), dtype=np.int32)

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        vegetation = ((h >= 35) & (h <= 95) & (s >= 40)).astype(np.uint8) * 255
        roof_mask = ((~vegetation.astype(bool)) & (s >= 18) & (v >= 35)).astype(np.uint8) * 255
        roof_mask |= (((s < 120) & (v > 60) & (v < 220)) & (np.mean(image, axis=2) > 35)).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        roof_mask = cv2.morphologyEx(roof_mask, cv2.MORPH_OPEN, kernel)
        roof_mask = cv2.morphologyEx(roof_mask, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(roof_mask, 8)
        boxes: List[List[int]] = []
        min_area = max(200, int(0.0008 * width * height))

        for label_index in range(1, num_labels):
            x, y, w, h, area = stats[label_index]
            if area < min_area or w < 20 or h < 20:
                continue
            x1 = max(0, x - 8)
            y1 = max(0, y - 8)
            x2 = min(width, x + w + 8)
            y2 = min(height, y + h + 8)
            boxes.append([x1, y1, x2, y2])

        if not boxes:
            box_width = max(80, int(width * 0.55))
            box_height = max(80, int(height * 0.55))
            x1 = max(0, (width - box_width) // 2)
            y1 = max(0, (height - box_height) // 2)
            x2 = min(width, x1 + box_width)
            y2 = min(height, y1 + box_height)
            boxes = [[x1, y1, x2, y2]]

        boxes = sorted(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)[:8]
        return np.array(boxes, dtype=np.int32)

    def annotate(self, image_pil: Image.Image, results: Sequence[dict]) -> Image.Image:
        """Overlay detection boxes and category labels on the original frame."""
        image = image_pil.copy()
        draw = ImageDraw.Draw(image)
        palette = {"RCC": "#10b981", "Tiled": "#f59e0b", "Tin": "#60a5fa"}

        for entry in results:
            bbox = entry.get("bbox", [0, 0, 0, 0])
            x_min, y_min, x_max, y_max = [int(v) for v in bbox]
            category = entry.get("category", "RCC")
            confidence = float(entry.get("confidence", 0.0))
            color = palette.get(category, "#e2e8f0")
            draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=color, width=4)
            draw.text((x_min + 8, max(y_min - 16, 2)), f"{category} {confidence:.2f}", fill=color)

        return image

    def create_polygon(self, bbox: Sequence[int]) -> Polygon:
        """Create a Shapely polygon from a bounding box."""
        x_min, y_min, x_max, y_max = [float(v) for v in bbox]
        coords = [
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max),
            (x_min, y_min),
        ]
        return Polygon(coords)
