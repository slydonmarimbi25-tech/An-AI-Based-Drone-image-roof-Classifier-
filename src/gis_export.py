"""Geospatial export utilities for rooftop analysis results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover
    gpd = None

try:
    from shapely.geometry import Polygon
except ImportError:  # pragma: no cover
    Polygon = None


def export_geojson(results: Iterable[dict], output_path: str = "rooftops_classified.geojson"):
    """Convert rooftop results to GeoJSON using GeoPandas when available."""
    features: List[dict] = []

    for item in results:
        bbox = item.get("bbox", [0, 0, 0, 0])
        x_min, y_min, x_max, y_max = [float(v) for v in bbox]
        if Polygon is not None:
            polygon = Polygon(
                [
                    (x_min, y_min),
                    (x_max, y_min),
                    (x_max, y_max),
                    (x_min, y_max),
                    (x_min, y_min),
                ]
            )
        else:
            polygon = [
                [
                    (x_min, y_min),
                    (x_max, y_min),
                    (x_max, y_max),
                    (x_min, y_max),
                    (x_min, y_min),
                ]
            ]

        features.append(
            {
                "roof_id": item.get("roof_id", 0),
                "category": item.get("category", "Unknown"),
                "confidence": float(item.get("confidence", 0.0)),
                "geometry": polygon,
            }
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if gpd is not None and Polygon is not None:
        gdf = gpd.GeoDataFrame(features, geometry="geometry", crs="EPSG:4326")
        gdf.to_file(output, driver="GeoJSON")
        return gdf

    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "roof_id": item["roof_id"],
                    "category": item["category"],
                    "confidence": item["confidence"],
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": item["geometry"] if isinstance(item["geometry"], list) else [[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]],
                },
            }
            for item in features
        ],
    }
    output.write_text(json.dumps(payload), encoding="utf-8")
    return payload
