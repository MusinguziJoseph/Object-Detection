import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import torch

def load_model():
    """Load the YOLO model"""
    model = YOLO('best1.pt') 
    return model

def process_image(image, model):
    """
    Process a single image with the YOLO model
    """
    results = model.predict(image)
    return results[0].plot()  # Getting the first result's plotted image

def process_video_frame(frame, model):
    """
    Process a video frame with the YOLO model
    """
    results = model.predict(frame)
    return results[0].plot()

def main():
    st.title("YOLO Object Detection App")
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Model loading
    @st.cache_resource
    def load_model_cache():
        return load_model()
    
    try:
        model = load_model_cache()
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        return
    
    # Confidence threshold slider
    confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    model.conf = confidence
    
    # Source selection
    source_radio = st.sidebar.radio(
        "Select Source", 
        ["Upload Image", "Upload Video", "Webcam"]
    )
    
    if source_radio == "Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_file is not None:
            # Convert uploaded file to image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                st.subheader("Detected Objects")
                processed_img = process_image(image, model)
                st.image(processed_img, caption="Processed Image", use_column_width=True)
    
    elif source_radio == "Upload Video":
        uploaded_file = st.file_uploader("Choose a video...", type=['mp4', 'avi', 'mov'])
        
        if uploaded_file is not None:
            # Save uploaded video to temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frame = process_video_frame(frame, model)
                stframe.image(processed_frame)
            
            cap.release()
    
    elif source_radio == "Webcam":
        
        camera_devices = [0]  
        selected_camera = st.sidebar.selectbox("Select Camera", camera_devices)
        
        run = st.checkbox('Start Webcam')
        
        # Webcam feed control loop
        if run:
            cap = cv2.VideoCapture(selected_camera)
            stframe = st.empty()

            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access webcam")
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frame = process_video_frame(frame, model)
                stframe.image(processed_frame)

                
                run = st.checkbox('Start Webcam', value=True)  
            
            cap.release()
            st.write("Webcam feed stopped.")
        else:
            st.write("Webcam is not running.")

if __name__ == '__main__':
    main()
