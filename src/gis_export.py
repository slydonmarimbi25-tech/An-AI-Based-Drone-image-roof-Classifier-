"""Geospatial export utilities for rooftop analysis results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence

import geopandas as gpd
from shapely.geometry import Polygon


def export_geojson(results: Iterable[dict], output_path: str = "rooftops_classified.geojson") -> gpd.GeoDataFrame:
    """Convert rooftop results to GeoJSON using GeoPandas."""
    features: List[dict] = []

    for item in results:
        bbox = item.get("bbox", [0, 0, 0, 0])
        x_min, y_min, x_max, y_max = [float(v) for v in bbox]
        polygon = Polygon(
            [
                (x_min, y_min),
                (x_max, y_min),
                (x_max, y_max),
                (x_min, y_max),
                (x_min, y_min),
            ]
        )
        features.append(
            {
                "roof_id": item.get("roof_id", 0),
                "category": item.get("category", "Unknown"),
                "confidence": float(item.get("confidence", 0.0)),
                "geometry": polygon,
            }
        )

    gdf = gpd.GeoDataFrame(features, geometry="geometry", crs="EPSG:4326")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output, driver="GeoJSON")
    return gdf
