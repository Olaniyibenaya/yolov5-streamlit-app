import streamlit as st
import torch
from PIL import Image
import numpy as np
import json
import pathlib
from streamlit_lottie import st_lottie
from io import BytesIO

# Optional: load Lottie animation
lottie_animation = load_lottiefile("Animation.json")  # Replace with your own file if needed

# For Colab path compatibility if running locally
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load your custom YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

# Page configuration
st.set_page_config(page_title="Custom Object Detection App", page_icon="🎯", layout="wide")

# Show animation
st_lottie(lottie_animation, height=300, width=1000)

# App title
st.title("Custom Object Detection")
st.write("Upload an image and this app will detect objects based on your trained model.")

# Sidebar: Confidence slider
st.sidebar.header("Detection Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")

    # Set model confidence
    model.conf = confidence_threshold
    results = model(image)

    # Draw boxes on image
    img_with_boxes = np.array(results.render()[0])
    img_with_boxes = Image.fromarray(img_with_boxes)

    # Display result
    st.image(img_with_boxes, caption=f'{len(results.xyxy[0])} objects detected', use_column_width=True)

    # Show class names
    class_names = model.names
    detected_classes = [class_names[int(cls)] for cls in results.pred[0][:, -1]]
    st.write("Detected Classes:", detected_classes)

    # Download button
    buf = BytesIO()
    img_with_boxes.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(label="Download Image", data=byte_im, file_name="output.png", mime="image/png")
