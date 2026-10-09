"""Zero-shot rooftop classification using CLIP-like prompts and a fallback heuristic."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from PIL import Image

try:
    import torch
    from transformers import CLIPModel, CLIPProcessor
except (ImportError, OSError, RuntimeError):  # pragma: no cover
    torch = None
    CLIPModel = None
    CLIPProcessor = None


class RooftopClassifier:
    """Classify rooftop crops into RCC, Tiled, or Tin using CLIP if available."""

    def __init__(self) -> None:
        self.class_names = ["RCC", "Tiled", "Tin"]
        self.candidate_labels = [
            "a concrete flat RCC roof",
            "a terracotta or blue tiled roof",
            "a corrugated tin metal roof",
        ]
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self) -> None:
        if CLIPModel is None or CLIPProcessor is None:
            return
        try:
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model.eval()
        except Exception:  # pragma: no cover
            self.model = None
            self.processor = None

    def _heuristic_prediction(self, crop: Image.Image) -> Dict[str, float]:
        arr = np.asarray(crop.convert("RGB"), dtype=np.float32)
        if arr.size == 0:
            category = "RCC"
            confidence = 0.55
        else:
            mean = arr.mean(axis=(0, 1))
            r, g, b = mean
            brightness = float(mean.mean())
            red_bias = r - max(g, b)

            if red_bias > 18 and r > 80:
                category = "Tiled"
                confidence = 0.88
            elif brightness > 150 and abs(r - g) < 28 and abs(g - b) < 28:
                category = "Tin"
                confidence = 0.82
            else:
                category = "RCC"
                confidence = 0.74

        probs = {name: 0.1 for name in self.class_names}
        probs[category] = float(confidence)
        for other in [name for name in self.class_names if name != category]:
            probs[other] = float((1.0 - confidence) / (len(self.class_names) - 1))
        return {"category": category, "confidence": float(confidence), "probs": probs}

    def classify_rooftops(self, image_pil: Image.Image, boxes: Sequence[Sequence[int]]) -> List[Dict[str, object]]:
        """Classify each rooftop crop from the detected boxes."""
        results: List[Dict[str, object]] = []
        width, height = image_pil.size

        for idx, box in enumerate(boxes, start=1):
            x_min, y_min, x_max, y_max = [int(v) for v in box]
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(width, x_max)
            y_max = min(height, y_max)

            crop = image_pil.crop((x_min, y_min, x_max, y_max))
            if self.model is not None and self.processor is not None and torch is not None:
                try:
                    inputs = self.processor(
                        text=self.candidate_labels,
                        images=crop,
                        return_tensors="pt",
                        padding=True,
                    )
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        logits = outputs.logits_per_image.softmax(dim=1)
                        probs = logits[0].cpu().numpy()
                    top_idx = int(np.argmax(probs))
                    category = self.class_names[top_idx]
                    confidence = float(probs[top_idx])
                except Exception:
                    heuristic = self._heuristic_prediction(crop)
                    category = heuristic["category"]
                    confidence = heuristic["confidence"]
            else:
                heuristic = self._heuristic_prediction(crop)
                category = heuristic["category"]
                confidence = heuristic["confidence"]

            results.append(
                {
                    "roof_id": idx,
                    "category": category,
                    "confidence": round(confidence, 3),
                    "bbox": [x_min, y_min, x_max, y_max],
                }
            )

        return results
