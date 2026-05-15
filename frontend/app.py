"""Streamlit entry — sidebar navigation with icons. Run: streamlit run frontend/app.py"""
import sys
from pathlib import Path

# Make /frontend importable so pages can `from api_client import ...`
sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests
import streamlit as st
from api_client import API_URL

st.set_page_config(
    page_title="Chest X-Ray Intelligence",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "DSAI 413 – Assignment 2  ·  MedGemma + ColPali / BiomedCLIP RAG",
    },
)


def _backend_status() -> tuple[bool, list[str]]:
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        if r.ok:
            return True, r.json().get("loaded", [])
    except requests.RequestException:
        pass
    return False, []


def home():
    st.title("🩻 Chest X-Ray Intelligence System")
    st.caption("DSAI 413 – Assignment 2  ·  MedGemma 1.5 + ColPali / BiomedCLIP retrieval")

    online, loaded = _backend_status()
    cols = st.columns([1, 3])
    with cols[0]:
        if online:
            st.metric("Backend", "🟢 online")
        else:
            st.metric("Backend", "🔴 offline")
    with cols[1]:
        if online:
            st.info(f"Endpoint: `{API_URL}`  ·  models loaded: " +
                    (", ".join(loaded) if loaded else "_none yet (loaded on first call)_"))
        else:
            st.error(f"Cannot reach `{API_URL}`. Start with `uvicorn api.main:app`.")

    st.write("---")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("🧾 Report Generation")
        st.write(
            "Upload a chest X-ray. **MedGemma 1.5 4B** writes a structured radiology report "
            "with **Findings** and **Impression** sections."
        )
        st.markdown(
            "- Generator: **MedGemma 1.5 4B IT** (medical-domain VLM)\n"
            "- Image-only input — no question needed\n"
            "- Typical latency: ~10–30 s on first call"
        )

    with c2:
        st.subheader("💬 QA Mode (RAG)")
        st.write(
            "Ask a clinical question about an X-ray. **BiomedCLIP** or **ColPali** retrieves "
            "similar reports; **MedGemma** answers, grounded in that context."
        )
        st.markdown(
            "- Default: **BiomedCLIP RAG** (best on our eval)\n"
            "- Switch retriever in the sidebar of the QA page\n"
            "- Toggle RAG off for ungrounded baseline"
        )

    st.write("---")

    st.subheader("📊 Model comparison")
    st.markdown(
        """
| Mode | BLEU-4 | ROUGE-L | Token-F1 |
|---|---:|---:|---:|
| MedGemma alone (no RAG) | 0.067 | 0.262 | 0.246 |
| MedGemma + BiomedCLIP RAG | 0.122 | **0.386** | **0.387** |
| **MedGemma + ColPali RAG** ⭐ | **0.139** | 0.378 | 0.379 |
| MedGemma report generation | 0.029 | 0.205 | 0.241 |
        """
    )
    st.caption(
        "Eval on 50 random QA pairs + 20 random reports. "
        "RAG ~2× answer quality. ColPali and BiomedCLIP are essentially tied — "
        "ColPali wins BLEU (precise phrasing), BiomedCLIP wins Token-F1 (medical-term coverage)."
    )

    st.write("---")
    st.caption("← Pick a mode from the sidebar to begin")


pages = {
    "Overview": [st.Page(home, title="Home", icon="🏠", default=True)],
    "Modes": [
        st.Page("pages/Report_Generation.py", title="Report Generation", icon="🧾"),
        st.Page("pages/QA_Mode.py",            title="QA Mode",            icon="💬"),
    ],
}
st.navigation(pages).run()
