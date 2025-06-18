import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 🔹 Load the YOLOv5 model trained on cassava diseases
model = YOLO('best.pt')

# 🔹 Define class names manually (if needed)
model.model.names = [
    'Cassava Mosaic Disease', 
    'Cassava Brown Streak Disease', 
    'Cassava Green Mite', 
    'Healthy'
]

# 🔹 Streamlit page configuration
st.set_page_config(
    page_title="Cassava Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 🔹 Page title and instructions
st.title("Cassava Disease Detection App")
st.write("Upload a cassava leaf image to detect whether it's healthy or infected.")

# 🔹 Sidebar - confidence threshold slider
st.sidebar.header("Detection Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.5, 
    step=0.05
)

# 🔹 Image uploader
uploaded_file = st.file_uploader("Upload a cassava leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Run detection
    results = model.predict(image, conf=confidence_threshold)

    # Visualize results
    for r in results:
        img_with_boxes = r.plot()  # returns numpy array with boxes
        st.image(img_with_boxes, caption="Detection Result", use_container_width=True)

        st.subheader("Detection Summary")
        if len(r.boxes) == 0:
            st.info("No disease or object detected above the confidence threshold.")
        else:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.model.names[cls]
                st.write(f"🔍 **{class_name}** — Confidence: `{conf:.2f}`")
else:
    st.warning("Please upload a cassava leaf image to start detection.")
