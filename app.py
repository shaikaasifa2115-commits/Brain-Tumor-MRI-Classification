
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    page_icon="🧠",
    layout="centered"
)

MODEL_PATH = "best_mobilenetv2.keras"
IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


st.title("🧠 Brain Tumor MRI Classification")
st.write(
    "Upload a brain MRI image for experimental classification "
    "using MobileNetV2."
)

st.warning(
    "Educational demonstration only. This tool is not intended "
    "for medical diagnosis."
)

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a brain MRI image",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(
            image,
            caption="Uploaded MRI Image",
            use_container_width=True
        )

        resized_image = image.resize(IMAGE_SIZE)
        image_array = np.array(resized_image, dtype=np.float32)

        image_array = (
            tf.keras.applications.mobilenet_v2.preprocess_input(
                image_array
            )
        )

        image_array = np.expand_dims(image_array, axis=0)

        with st.spinner("Analyzing image..."):
            predictions = model.predict(
                image_array, verbose=0
            )[0]

        predicted_index = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(predictions[predicted_index]) * 100

        st.subheader("Prediction Results")
        st.success(
            f"Predicted class: {predicted_class.replace('_', ' ').title()}"
        )
        st.metric("Model confidence score", f"{confidence:.2f}%")

        st.subheader("Scores for all classes")
        for name, score in zip(CLASS_NAMES, predictions):
            st.write(
                f"**{name.replace('_', ' ').title()}**: "
                f"{float(score) * 100:.2f}%"
            )

    except Exception as error:
        st.error(f"Unable to process the image: {error}")
