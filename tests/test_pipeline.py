from pathlib import Path

from PIL import Image
import numpy as np

from core.pipeline import RooftopAnalysisPipeline


def test_pipeline_processes_generated_image(tmp_path):
    image_path = tmp_path / "roof_sample.png"
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:, :, 0] = 120
    img[:, :, 1] = 120
    img[:, :, 2] = 120
    img[50:150, 50:150] = [180, 160, 150]
    Image.fromarray(img).save(image_path)

    pipeline = RooftopAnalysisPipeline()
    annotated, gdf, metadata = pipeline.run(str(image_path))

    assert annotated is not None
    assert len(gdf) >= 1
    assert metadata["image_path"] == str(image_path)
    assert metadata["processed"] is True


def test_detector_fallback_is_image_specific():
    from src.detector import RooftopDetector

    detector = RooftopDetector(conf_threshold=0.005)
    small = Image.new("RGB", (320, 240), color=(200, 200, 200))
    large = Image.new("RGB", (1600, 1000), color=(180, 180, 180))

    small_boxes = detector.detect_rooftops(small)
    large_boxes = detector.detect_rooftops(large)

    assert len(small_boxes) == 1
    assert len(large_boxes) == 1
    assert small_boxes[0][2] <= small.width
    assert large_boxes[0][2] <= large.width
    assert not np.array_equal(small_boxes, large_boxes)
