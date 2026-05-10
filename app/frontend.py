import requests
import streamlit as st
from PIL import Image


# ---------------------------------
# Config
# ---------------------------------

API_URL = "http://127.0.0.1:8000/predict"


# ---------------------------------
# Page Setup
# ---------------------------------

st.set_page_config(
    page_title="CV MLOps App",
    page_icon="📷",
    layout="centered"
)


# ---------------------------------
# Header
# ---------------------------------

st.title("📷 Data-Centric CV MLOps App")

st.markdown(
    """
Upload an image and click **Predict** to get model inference.
"""
)

st.divider()


# ---------------------------------
# Upload Section
# ---------------------------------

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)


# ---------------------------------
# Image Preview
# ---------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

else:

    st.info(
        "Please upload an image to continue."
    )


# ---------------------------------
# Prediction Button
# ---------------------------------

predict_clicked = st.button(
    "🔍 Predict",
    use_container_width=True
)


# ---------------------------------
# Prediction Logic
# ---------------------------------

if predict_clicked:

    if uploaded_file is None:

        st.warning(
            "Upload an image first."
        )

    else:

        with st.spinner(
            "Generating prediction..."
        ):

            try:

                response = requests.post(
                    API_URL,
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type
                        )
                    }
                )

                prediction = response.json()

                st.success(
                    f"Prediction: "
                    f"{prediction['prediction']}"
                )

                if "confidence" in prediction:

                    st.metric(
                        "Confidence Score",
                        f"{prediction['confidence']:.4f}"
                    )

            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )


# ---------------------------------
# Footer
# ---------------------------------

st.divider()

st.caption(
    "Built using FastAPI, Streamlit, PyTorch, MLflow, DVC, and Evidently AI"
)