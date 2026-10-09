from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

from PIL import Image, ImageDraw

from config.settings import OUTPUT_DIR
from core.classifier import RoofClassifier
from core.detector import RoofDetector
from core.gis_engine import GISEngine


class RooftopAnalysisPipeline:
    def __init__(self):
        self.detector = RoofDetector()
        self.classifier = RoofClassifier()
        self.gis = GISEngine()

    def run(self, image_path: str) -> Tuple[Image.Image, Any, dict]:
        image = Image.open(image_path).convert("RGB")
        boxes, detection_meta = self.detector.detect_roofs(image)

        results: List[dict] = []
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)

        palette = {
            "RCC": (34, 197, 94),
            "Tiled": (249, 115, 22),
            "Tin": (59, 130, 246),
        }

        for idx, bbox in enumerate(boxes, start=1):
            x1, y1, x2, y2 = [int(v) for v in bbox]
            crop = image.crop((x1, y1, x2, y2))
            classification = self.classifier.classify_crop(crop)
            label = classification["label"]
            color = palette.get(label, (255, 255, 255))

            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
            draw.text((x1 + 6, max(0, y1 - 18)), f"{idx}:{label}", fill=color)

            geometry = self.gis.pixel_to_polygon([x1, y1, x2, y2], image.size)
            results.append({
                "roof_id": idx,
                "roof_type": label,
                "confidence": classification["confidence"],
                "bbox": [x1, y1, x2, y2],
                "geometry": geometry,
            })

        output_path = OUTPUT_DIR / "rooftop_results.geojson"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gdf = self.gis.export_geojson(results, str(output_path))

        metadata = {
            "image_path": image_path,
            "processed": True,
            "detections": len(results),
            "output_path": str(output_path),
            "detections_meta": detection_meta,
        }
        return annotated, gdf, metadata
