"""Core computer vision and geospatial utilities for rooftop extraction and classification."""

try:
    from .detector import RooftopDetector
except Exception:  # pragma: no cover
    RooftopDetector = None

try:
    from .classifier import RooftopClassifier
except Exception:  # pragma: no cover
    RooftopClassifier = None

try:
    from .gis_export import export_geojson
except Exception:  # pragma: no cover
    export_geojson = None

__all__ = ["RooftopDetector", "RooftopClassifier", "export_geojson"]
