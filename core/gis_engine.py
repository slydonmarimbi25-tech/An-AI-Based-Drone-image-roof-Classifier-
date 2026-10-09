from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List

try:
    from shapely.geometry import Polygon
except ImportError:  # pragma: no cover
    Polygon = None

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover
    gpd = None


class GISEngine:
    def pixel_to_polygon(self, bbox: List[int], image_shape: tuple[int, int]):
        x1, y1, x2, y2 = map(int, bbox)
        coords = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
        if Polygon is not None:
            return Polygon(coords)
        return coords

    def export_geojson(self, results: Iterable[dict], output_path: str, crs: str = "EPSG:4326"):
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        features = []
        for item in results:
            geom = item.get("geometry")
            if Polygon is not None and hasattr(geom, "geom_type"):
                geometry = json.loads(json.dumps(geom.__geo_interface__))
            elif isinstance(geom, list):
                geometry = {
                    "type": "Polygon",
                    "coordinates": [geom],
                }
            else:
                geometry = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}

            features.append({
                "type": "Feature",
                "properties": {
                    "roof_id": item.get("roof_id"),
                    "roof_type": item.get("roof_type"),
                    "confidence": item.get("confidence"),
                    "bbox": item.get("bbox"),
                },
                "geometry": geometry,
            })

        payload = {"type": "FeatureCollection", "features": features}

        if gpd is not None:
            try:
                gdf = gpd.GeoDataFrame.from_features(payload["features"], crs=crs)
                gdf.to_file(output, driver="GeoJSON")
                return gdf
            except Exception:
                pass

        with output.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        return results
