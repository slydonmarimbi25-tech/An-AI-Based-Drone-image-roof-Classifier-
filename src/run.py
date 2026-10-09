from __future__ import annotations

import argparse

from core.pipeline import RooftopAnalysisPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Rooftop analysis CLI")
    parser.add_argument("--input", required=True, help="Image path to process")
    parser.add_argument("--output", default="outputs/rooftop_results.geojson", help="Output GeoJSON location")
    args = parser.parse_args()

    pipeline = RooftopAnalysisPipeline()
    annotated, gdf, metadata = pipeline.run(args.input)
    print(f"Processed {metadata['detections']} rooftops")
    print(f"Result written to: {metadata['output_path']}")


if __name__ == "__main__":
    main()
