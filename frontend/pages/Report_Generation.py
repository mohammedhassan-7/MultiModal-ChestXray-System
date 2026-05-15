"""Report Generation mode — upload an X-ray and get a structured report."""
import sys
from pathlib import Path

# Make frontend/ importable when Streamlit runs this page
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
import streamlit as st
from api_client import generate_report

st.title("🧾 Report Generation")
st.caption("Upload a chest X-ray → MedGemma 1.5 produces a structured radiology report.")

st.sidebar.header("Generator")
st.sidebar.markdown("**MedGemma 1.5 4B IT** _(only supported generator for report-gen)_")
st.sidebar.caption("Single model mode — see QA tab to compare retrievers.")

uploaded = st.file_uploader(
    "Chest X-ray (JPG / PNG)", type=["jpg", "jpeg", "png"],
    help="Use any JPG from data/images/ for a quick demo.",
)

if not uploaded:
    st.info("⬆️ Upload an X-ray to begin.")
    st.stop()

image_bytes = uploaded.getvalue()
left, right = st.columns([1, 1])
with left:
    st.image(image_bytes, caption=uploaded.name, width='stretch')

with right:
    if st.button("Generate report", type="primary", width='stretch'):
        try:
            with st.spinner("MedGemma is reading the X-ray… (~10–30 s on first run)"):
                result = generate_report(image_bytes, filename=uploaded.name)
            st.subheader("📋 Report")
            st.write(result["report"])
            st.caption(f"Model: `{result['model']}`")
        except requests.RequestException as e:
            st.error(f"API call failed: {e}")
            st.caption("Is the FastAPI backend running? `uvicorn api.main:app`")
    else:
        st.caption("Click **Generate report** when ready.")
