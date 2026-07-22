"""Streamlit demo (`streamlit run app.py`): photo in, breed out, showing the crop the classifier
actually sees so the prediction is explainable.
"""

import glob
import os
import tempfile
from pathlib import Path

import streamlit as st

from bovid import config
from bovid.logging_conf import setup_logging
from bovid.predict import load_model, predict
from bovid.utils import bgr_to_pil

setup_logging()

st.set_page_config(page_title="Indian Bovine Breed Classifier", page_icon="🐄", layout="wide")


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def example_images(n=6):
    """A few held-out test photos so the demo can be tried without hunting for an image."""
    test_dir = os.path.join(config.RAW_DATA_DIR, "test")
    return sorted(glob.glob(os.path.join(test_dir, "*.jpg")))[:n]


def render_result(result):
    left, right = st.columns([1, 1])
    with left:
        st.image(result.image_path, caption="Input photo", use_container_width=True)
    with right:
        st.image(bgr_to_pil(result.crop_bgr),
                 caption=f"What the classifier sees — crop via **{result.crop_method}**",
                 use_container_width=True)

    st.subheader("Predictions")
    for breed, confidence in result.predictions:
        st.write(f"**{breed.replace('_', ' ')}** — {confidence:.1%}")
        st.progress(confidence)

    if result.is_confident:
        st.success(f"Confident: top prediction is above the {config.CONFIDENCE_THRESHOLD:.0%} threshold.")
    else:
        st.warning(
            f"**Uncertain** — top-1 confidence is below {config.CONFIDENCE_THRESHOLD:.0%}. "
            "Treat this as a suggestion and consider the other candidates above. "
            "(This threshold was chosen from measured calibration, not guessed.)"
        )


with st.sidebar:
    st.header("About")
    st.write(
        "Two-stage pipeline: a detector crops the animal, then a YOLOv8s classifier "
        "predicts the breed. Both stages use identical preprocessing at training and "
        "inference time."
    )
    st.subheader("Measured accuracy")
    st.metric("Top-1", "53.5% ± 4.5%")
    st.metric("Top-3", "78.2% ± 2.9%")
    st.caption(
        "5-fold grouped cross-validation over all 795 images — every image is held out in "
        "one fold. Grouped so that augmented copies of the same photo never straddle train "
        "and validation (a naive split leaks and reports a misleading ~93%)."
    )
    st.subheader("Known limitations")
    st.write(
        "- ~32 training images per breed — data scarcity is the main accuracy ceiling.\n"
        "- Visually similar zebu breeds are genuinely hard to separate "
        "(e.g. Gir/Banni, Kankrej/Kangayam).\n"
        "- Assistive tool, **not** an authoritative breed determination."
    )

st.title("🐄 Indian Bovine Breed Classifier")
st.caption("Upload a photo of a cow or buffalo, or try one of the examples below.")

uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

st.write("**…or try an example:**")
examples = example_images()
cols = st.columns(len(examples)) if examples else []
chosen_example = None
for col, path in zip(cols, examples, strict=True):
    with col:
        st.image(path, use_container_width=True)
        if st.button("Use this", key=path):
            chosen_example = path

result = None
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(suffix=Path(uploaded_file.name).suffix, delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    # Narrow: only a corrupt-upload OSError gets a friendly message; other errors propagate.
    try:
        result = predict(tmp_path, model=get_model())
    except OSError as e:
        st.error(f"That file could not be read as an image. ({e})")
    finally:
        os.unlink(tmp_path)
elif chosen_example:
    result = predict(chosen_example, model=get_model())

if result:
    render_result(result)
