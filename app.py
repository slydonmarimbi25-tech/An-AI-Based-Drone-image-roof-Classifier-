from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.detector import RooftopDetector
from src.classifier import RooftopClassifier
from src.gis_export import export_geojson


DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


st.set_page_config(page_title="Drone Roof Classification", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #0b1020 0%, #101827 100%);
        color: #e5eefb;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95);
    }
    .metric-card {
        background: rgba(15, 118, 110, 0.12);
        border: 1px solid rgba(94, 234, 212, 0.18);
        border-radius: 0.8rem;
        padding: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_sample_image() -> Image.Image:
    sample_path = DATA_DIR / "sample_drone.jpg"
    if not sample_path.exists():
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (1600, 1000), color=(160, 170, 180))
        for x in range(160, 1100, 220):
            for y in range(100, 800, 220):
                box = (x, y, x + 210, y + 210)
                img.paste((70, 80, 90), box)
        img.save(sample_path)
    return Image.open(sample_path).convert("RGB")


@st.cache_data
def analyze_image(image_bytes: bytes, conf_threshold: float):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    detector = RooftopDetector(conf_threshold=conf_threshold)
    boxes = detector.detect_rooftops(image)
    results = RooftopClassifier().classify_rooftops(image, boxes)
    geojson_path = OUTPUT_DIR / "rooftops_classified.geojson"
    geojson_path.parent.mkdir(parents=True, exist_ok=True)
    gdf = export_geojson(results, str(geojson_path))
    annotated = detector.annotate(image, results)
    return annotated, pd.DataFrame(results), gdf


st.title("AI Rooftop Classification")
st.caption("Drone imagery feature extraction and roof type classification for RCC, Tiled, and Tin roofs.")

with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload Drone Image", type=["jpg", "jpeg", "png", "tif", "tiff"])
    conf_threshold = st.slider("Detection confidence", min_value=0.001, max_value=0.10, value=0.005, step=0.001)
    category_filter = st.selectbox("Show category", ["All", "RCC", "Tiled", "Tin"])
    run_button = st.button("Run analysis")

if run_button or uploaded_file is not None:
    if uploaded_file is not None:
        input_bytes = uploaded_file.getvalue()
        image_name = uploaded_file.name
    else:
        sample_image = load_sample_image()
        img_buffer = io.BytesIO()
        sample_image.save(img_buffer, format="PNG")
        input_bytes = img_buffer.getvalue()
        image_name = "sample_drone.jpg"

    with st.spinner("Analyzing rooftop features..."):
        annotated_image, df, gdf = analyze_image(input_bytes, conf_threshold)

    if category_filter != "All":
        df = df[df["category"] == category_filter]

    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        st.subheader("Annotated Image")
        st.image(annotated_image, width="stretch", caption=image_name)

    with col_right:
        st.subheader("Spatial Analytics")
        total_roofs = len(df)
        counts = df["category"].value_counts().to_dict() if not df.empty else {}
        avg_conf = df["confidence"].mean() if not df.empty else 0.0

        metric_cols = st.columns(3)
        metric_cols[0].metric("Total Roofs", total_roofs)
        metric_cols[1].metric("RCC", counts.get("RCC", 0))
        metric_cols[2].metric("Avg Confidence", f"{avg_conf:.2%}")

        st.markdown("### Category Distribution")
        st.bar_chart(df["category"].value_counts()) if not df.empty else st.info("No rooftops detected for the selected filter.")

    st.subheader("Extracted Features")
    st.dataframe(df, width="stretch")

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    geojson_bytes = gdf.to_json()

    st.download_button("Download GeoJSON", geojson_bytes, file_name="rooftops_classified.geojson", mime="application/geo+json")
    st.download_button("Download CSV", csv_buffer.getvalue(), file_name="roof_summary.csv", mime="text/csv")
else:
    sample_image = load_sample_image()
    st.subheader("Sample Preview")
    st.image(sample_image, width="stretch")
    st.info("Upload a drone image or use the sample to run the rooftop analysis workflow.")


if __name__ == "__main__":
    app = None
