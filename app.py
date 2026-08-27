import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Scene Classifier",
    page_icon="🌍",
    layout="centered"
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = "intel_scene_cnn.keras"
PKL_PATH = "intel_scene_cnn_info.pkl"


# ============================================================
# LOAD CNN MODEL
# ============================================================

@st.cache_resource
def load_cnn_model():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model


# ============================================================
# LOAD PKL INFORMATION
# ============================================================

@st.cache_resource
def load_model_info():

    with open(
        PKL_PATH,
        "rb"
    ) as file:

        info = pickle.load(file)

    return info


# ============================================================
# LOAD MODEL AND INFORMATION
# ============================================================

try:

    model = load_cnn_model()

    model_info = load_model_info()

    CLASS_NAMES = model_info["class_names"]

    IMAGE_SIZE = tuple(
        model_info["image_size"]
    )

except Exception as e:

    st.error(
        "Unable to load the CNN model or PKL file."
    )

    st.error(str(e))

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("🌍 AI-Based Natural Scene Classification")

st.write(
    "Upload an image and the custom CNN model "
    "will classify the scene into one of six categories."
)


# ============================================================
# INFORMATION
# ============================================================

st.info(
    """
    **CNN Classes**

    🏢 Buildings  
    🌲 Forest  
    ❄️ Glacier  
    ⛰️ Mountain  
    🌊 Sea  
    🛣️ Street
    """
)


# ============================================================
# IMAGE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(

    "Upload an image",

    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image):

    # Resize
    image = image.resize(
        IMAGE_SIZE
    )

    # Convert to array
    image_array = np.array(
        image
    )

    # Make sure image has 3 channels
    if image_array.shape[-1] == 4:

        image_array = image_array[:, :, :3]

    # Normalize
    image_array = (
        image_array.astype(
            np.float32
        ) / 255.0
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # CNN prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    # Get predicted class
    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        predictions[
            predicted_index
        ]
    )

    return (
        predicted_class,
        confidence,
        predictions
    )


# ============================================================
# DISPLAY AND PREDICT IMAGE
# ============================================================

if uploaded_file is not None:

    # Open image
    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # Display image
    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Input Image",
        use_container_width=True
    )


    # Predict button
    if st.button(
        "🔍 Classify Image",
        use_container_width=True
    ):

        with st.spinner(
            "CNN is analyzing the image..."
        ):

            predicted_class, confidence, predictions = (
                predict_image(image)
            )


        # ====================================================
        # RESULT
        # ====================================================

        st.success(
            f"Prediction: {predicted_class.title()}"
        )


        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )


        # ====================================================
        # ALL CLASS PROBABILITIES
        # ====================================================

        st.subheader(
            "Class Probabilities"
        )


        probability_data = {}

        for i, class_name in enumerate(
            CLASS_NAMES
        ):

            probability_data[
                class_name.title()
            ] = float(
                predictions[i]
            )


        # Display progress bars
        for class_name, probability in (
            probability_data.items()
        ):

            st.write(
                f"**{class_name}** — "
                f"{probability * 100:.2f}%"
            )

            st.progress(
                probability
            )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "About the Model"
    )

    st.write(
        "Custom Convolutional Neural Network"
    )

    st.write(
        "**Dataset:** Intel Image Classification"
    )

    st.write(
        "**Number of Classes:** 6"
    )

    st.write(
        "**Image Size:** 150 × 150"
    )

    st.write(
        "**Model:** CNN"
    )

    st.write(
        "**Framework:** TensorFlow / Keras"
    )

    st.divider()

    st.write(
        "This application uses a CNN trained "
        "to classify natural and urban scenes."
    )