from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"

CLASS_NAMES = ["RCC", "Tiled", "Tin"]
TEXT_PROMPTS = {
    "RCC": "a flat concrete RCC grey roof on a building",
    "Tiled": "a terracotta or ceramic red or blue tiled sloped roof",
    "Tin": "a corrugated galvanized metal tin sheet roof",
}
DEFAULT_CONFIDENCE_THRESHOLD = 0.05
YOLO_WORLD_CLASSES = ["building", "house", "rooftop", "structure"]
DEFAULT_CRS = "EPSG:4326"
ALTERNATE_CRS = "EPSG:3857"
