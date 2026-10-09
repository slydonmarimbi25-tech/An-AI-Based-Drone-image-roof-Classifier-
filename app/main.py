from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from core.pipeline import RooftopAnalysisPipeline


st.set_page_config(page_title="Roof Classifier Demo", layout="wide")
st.title("AI Rooftop Classification")

uploaded_file = st.sidebar.file_uploader("Upload drone image", type=["png", "jpg", "jpeg", "tif", "tiff"])
confidence = st.sidebar.slider("Detection confidence", 0.01, 0.20, 0.05, step=0.01)
run_button = st.sidebar.button("Run analysis")

if uploaded_file is not None:
    temp_dir = Path("outputs/uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)
    image_path = temp_dir / uploaded_file.name
    image_path.write_bytes(uploaded_file.getvalue())
    st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

if run_button and uploaded_file is not None:
    pipeline = RooftopAnalysisPipeline()
    pipeline.detector.confidence_threshold = float(confidence)
    annotated, gdf, metadata = pipeline.run(str(image_path))

    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("Annotated result")
        st.image(annotated, use_container_width=True)
    with right_col:
        st.subheader("Summary")
        roofs = gdf if isinstance(gdf, list) else []
        counts = {}
        for item in roofs:
            counts[item.get("roof_type", "Unknown")] = counts.get(item.get("roof_type", "Unknown"), 0) + 1
        st.metric("Total roofs", len(roofs))
        st.write(counts)
        st.write(metadata)

    st.download_button(
        "Download GeoJSON",
        data=(gdf if isinstance(gdf, str) else ""),
        file_name="rooftop_results.geojson",
        mime="application/geo+json",
    )
else:
    st.info("Upload an image and click Run analysis to process it.")
