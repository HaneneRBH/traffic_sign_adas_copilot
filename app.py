"""
Phase 4 - Streamlit demo for the Traffic-Sign ADAS Copilot.

Wraps the perception -> RAG bridge in a small web UI: drop in (or pick) a sign
image and see the predicted class, its confidence, the top-k alternatives, and a
grounded explanation. The detected speed limit is surfaced as a preview of what
the Phase 3 RL policy will consume.

Run from the project root:

    streamlit run app.py

CPU only. The model and the KB load once (cached); no index or Ollama is required
(explanation falls back to the deterministic template). Toggle the LLM in the
sidebar if you have Ollama running.
"""

import os
import sys
import tempfile

import streamlit as st

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "src", "pipeline"))
sys.path.insert(0, os.path.join(_ROOT, "src", "perception"))

from predict_and_explain import load_model, predict, explain, Retriever, DEFAULT_CKPT  # noqa: E402
from class_names import speed_limit_kmh  # noqa: E402

TEST_DIR = os.path.join(_ROOT, "data", "Test")


# --- cached heavy resources (load once per session) -----------------------

@st.cache_resource(show_spinner="Loading perception model...")
def get_model(ckpt_path):
    return load_model(ckpt_path)


@st.cache_resource(show_spinner="Loading knowledge base...")
def get_retriever():
    return Retriever()


def list_sample_images(n=24):
    if not os.path.isdir(TEST_DIR):
        return []
    files = sorted(f for f in os.listdir(TEST_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg", ".ppm")))
    return files[:n]


# --- UI -------------------------------------------------------------------

st.set_page_config(page_title="Traffic-Sign ADAS Copilot", page_icon="🚦", layout="wide")
st.title("🚦 Traffic-Sign ADAS Copilot")
st.caption("Perception (fine-tuned MobileNetV3) → grounded explanation (RAG). CPU only.")

with st.sidebar:
    st.header("Settings")
    ckpt_path = st.text_input("Checkpoint", value=DEFAULT_CKPT)
    use_llm = st.checkbox("Use local LLM (Ollama)", value=False,
                          help="If unchecked, the explanation uses the deterministic "
                               "template built straight from the rule base (no LLM needed).")
    topk = st.slider("Top-k alternatives", 1, 5, 3)
    st.divider()
    st.markdown(
        "**Pipeline status**\n\n"
        "1. Perception ✅\n"
        "2. Explanation (RAG) ✅\n"
        "3. Speed decision (RL) ⬜\n"
        "4. This demo ✅"
    )

# Image source: upload, or pick a sample from data/Test.
col_src1, col_src2 = st.columns(2)
with col_src1:
    uploaded = st.file_uploader("Upload a sign image", type=["png", "jpg", "jpeg", "ppm"])
with col_src2:
    samples = list_sample_images()
    sample = st.selectbox("...or pick a test image", ["(none)"] + samples)

image_path = None
if uploaded is not None:
    suffix = os.path.splitext(uploaded.name)[1] or ".png"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.close()
    image_path = tmp.name
elif sample and sample != "(none)":
    image_path = os.path.join(TEST_DIR, sample)

if image_path is None:
    st.info("Upload an image or pick a test image to run the pipeline.")
    st.stop()

# --- run the pipeline -----------------------------------------------------

try:
    model, img_size = get_model(ckpt_path)
except FileNotFoundError:
    st.error(f"Checkpoint not found: {ckpt_path}")
    st.stop()

retriever = get_retriever()
class_id, confidence, top = predict(model, img_size, image_path, topk=topk)
exp = explain(class_id, use_llm=use_llm, retriever=retriever)

# --- display --------------------------------------------------------------

left, right = st.columns([1, 1.6])

with left:
    st.image(image_path, caption=os.path.basename(image_path), width=240)

with right:
    st.metric(label="Predicted sign", value=exp.get("name", f"class {class_id}"),
              delta=f"{confidence * 100:.1f}% confidence")

    kmh = speed_limit_kmh(class_id)
    if kmh is not None:
        st.success(f"Detected speed limit: **{kmh} km/h** "
                   f"— this is what the Phase 3 RL policy will turn into a target speed.")

    badge = "LLM (Ollama)" if exp.get("source") == "llm" else "template (no LLM)"
    st.markdown(f"**Explanation** &nbsp; :gray[via {badge}]")
    st.info(exp.get("explanation", ""))

st.divider()
st.subheader("Top-k alternatives")
labels = [retriever.by_class_id(c)["name"] if retriever.by_class_id(c) else f"class {c}"
          for c, _ in top]
probs = {labels[i]: top[i][1] for i in range(len(top))}
st.bar_chart(probs, horizontal=True)
