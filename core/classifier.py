from __future__ import annotations

from typing import Any, Dict

import numpy as np
from PIL import Image

from config.settings import CLASS_NAMES


class RoofClassifier:
    """Deterministic classifier based on colour and texture cues for the prototype pipeline."""

    def __init__(self):
        self.class_names = CLASS_NAMES

    def _rgb_profile(self, image: Image.Image) -> Dict[str, float]:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        mean = arr.mean(axis=(0, 1))
        std = arr.std(axis=(0, 1))
        return {
            "r": float(mean[0]),
            "g": float(mean[1]),
            "b": float(mean[2]),
            "saturation": float(np.std(arr[:, :, 0] - arr[:, :, 1])) if arr.size else 0.0,
            "brightness": float(mean.mean()),
            "contrast": float(std.mean()),
        }

    def classify_crop(self, crop_image: Image.Image) -> Dict[str, Any]:
        profile = self._rgb_profile(crop_image)
        r, g, b = profile["r"], profile["g"], profile["b"]

        if r > max(g, b) + 10:
            label = "Tiled"
            confidence = 0.82
        elif abs(r - g) < 10 and abs(g - b) < 10 and profile["brightness"] > 120:
            label = "Tin"
            confidence = 0.74
        else:
            label = "RCC"
            confidence = 0.76

        probabilities = {name: 0.1 for name in self.class_names}
        probabilities[label] = round(confidence, 3)
        for other in [name for name in self.class_names if name != label]:
            probabilities[other] = round((1.0 - confidence) / (len(self.class_names) - 1), 3)

        return {
            "label": label,
            "confidence": round(confidence, 3),
            "probabilities": probabilities,
            "features": profile,
        }
