"""QA Mode — ask a clinical question grounded by retrieved reports."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
import streamlit as st
from api_client import answer_question

st.title("💬 QA Mode (RAG)")
st.caption("Ask a clinical question about a chest X-ray. Retrieval grounds the answer in similar reports.")

# Three model presets — defaulting to the best from our eval
PRESETS = {
    "MedGemma + ColPali RAG (best BLEU)":      {"use_rag": True,  "retriever": "colpali"},
    "MedGemma + BiomedCLIP RAG (best F1)":     {"use_rag": True,  "retriever": "clip"},
    "MedGemma alone (no RAG)":                 {"use_rag": False, "retriever": "clip"},
}

st.sidebar.header("Model")
preset_name = st.sidebar.selectbox("Select pipeline", list(PRESETS.keys()), index=0)
preset = PRESETS[preset_name]

st.sidebar.markdown("---")
st.sidebar.markdown("**Generator**: MedGemma 1.5 4B IT")
st.sidebar.markdown(f"**RAG**: {'on' if preset['use_rag'] else 'off'}")
if preset["use_rag"]:
    st.sidebar.markdown(f"**Retriever**: `{preset['retriever']}`")
st.sidebar.caption("Defaults reflect our eval ranking (see landing page).")

col_input, col_output = st.columns([1, 1])

with col_input:
    uploaded = st.file_uploader(
        "Chest X-ray (JPG / PNG)", type=["jpg", "jpeg", "png"],
    )
    question = st.text_input(
        "Question",
        placeholder="e.g. Is there evidence of pneumothorax?",
        help="Ask a single, specific clinical question. Image-grounded works best.",
    )
    submit = st.button("Ask", type="primary", width='stretch',
                        disabled=not (uploaded and question))
    if uploaded:
        st.image(uploaded.getvalue(), caption=uploaded.name, width='stretch')

with col_output:
    if not submit:
        st.info("Upload an X-ray, type a question, and click **Ask**.")
        st.stop()

    try:
        with st.spinner("Retrieving + answering…"):
            result = answer_question(
                uploaded.getvalue(), question,
                use_rag=preset["use_rag"],
                retriever_name=preset["retriever"],
                filename=uploaded.name,
            )
    except requests.RequestException as e:
        st.error(f"API call failed: {e}")
        st.caption("Is the FastAPI backend running? `uvicorn api.main:app`")
        st.stop()

    st.subheader("🩺 Answer")
    st.write(result["answer"])

    if result.get("retrieved"):
        st.subheader(f"📚 Retrieved evidence  ·  retriever: `{result['retriever']}`")
        for r in result["retrieved"]:
            with st.expander(f"Study **{r['study_id']}**  —  score {r['score']:.3f}"):
                st.write(r["report"])
    elif preset["use_rag"]:
        st.warning("No evidence returned by the retriever.")
    else:
        st.caption("No retrieval was performed (no-RAG mode).")
