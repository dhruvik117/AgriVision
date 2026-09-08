import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AgriVision",
    page_icon="🌿",
    layout="centered"
)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

MODEL_PATH = "model/AgriVision_best.keras"
CLASS_PATH = "class_names.json"

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_classes():
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

model = load_model()
class_names = load_classes()

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🌿 AgriVision")
st.subheader("AI-Powered Crop Disease Screening")

st.info(
    "Upload a clear image of a crop leaf. "
    "AgriVision analyzes the image and provides a screening result."
)

# --------------------------------------------------
# UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "📷 Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded image",
        use_container_width=True
    )

    if st.button("🔍 Analyze Leaf", use_container_width=True):

        with st.spinner("Analyzing image..."):

            # IMPORTANT:
            # This model expects RAW 0–255 pixel values.
            img = image.resize((224, 224))
            arr = np.array(img).astype("float32")
            arr = np.expand_dims(arr, axis=0)

            # Prediction
            predictions = model.predict(arr, verbose=0)[0]

            top_indices = np.argsort(predictions)[::-1][:3]

            top_idx = top_indices[0]
            predicted_class = class_names[top_idx]
            score = float(predictions[top_idx])

        # --------------------------------------------------
        # RESULT
        # --------------------------------------------------

        st.success("Analysis complete")

        st.markdown("### 🌱 Result")

        st.write(
            f"**Detected condition:** "
            f"{predicted_class.replace('_', ' ')}"
        )

        st.write(
            f"**Model score:** {score * 100:.2f}%"
        )

        # --------------------------------------------------
        # TOP 3
        # --------------------------------------------------

        st.markdown("### 📊 Top 3 Model Results")

        for rank, idx in enumerate(top_indices, start=1):

            st.write(
                f"**{rank}.** "
                f"{class_names[idx].replace('_', ' ')} — "
                f"{predictions[idx] * 100:.2f}%"
            )

        # --------------------------------------------------
        # DISCLAIMER
        # --------------------------------------------------

        st.warning(
            "AgriVision is a screening and educational tool. "
            "AI predictions can be incorrect and should be verified "
            "with a qualified agricultural expert before treatment decisions."
        )

# --------------------------------------------------
# ABOUT
# --------------------------------------------------

with st.expander("ℹ️ About AgriVision"):

    st.write(
        "AgriVision is an AI-based crop leaf disease screening "
        "prototype developed for educational and agricultural awareness."
    )

    st.write(
        "The model was trained to recognize 15 crop-health categories "
        "covering pepper, potato, and tomato leaves."
    )
