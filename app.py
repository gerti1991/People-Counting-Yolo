import streamlit as st
import cv2
import numpy as np
from models.model_streamlit import process_video, model
from live_camera_app import run_live_counter

# Streamlit interface
st.set_page_config(
    page_title="People Counter",
    page_icon="👥",
    layout="wide"
)

st.title("👥 People Counting System")
st.markdown("Advanced people counting using YOLO - Support for videos and live cameras")

# Sidebar for mode selection
st.sidebar.title("Select Mode")
mode = st.sidebar.radio(
    "Choose counting mode:",
    ["📹 Video File Processing", "🎥 Live Camera Counting"]
)

if mode == "📹 Video File Processing":
    st.header("Video File Processing")
    video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])

    if video_file is not None:
        # Save uploaded file to a temporary location
        video_path = 'data/uploaded_video.mp4'
        with open(video_path, 'wb') as f:
            f.write(video_file.read())
        
        output_path = 'results/output_video.mp4'
        
        batch_size = st.slider("Batch Size (for processing speed)", 1, 8, 4)
        
        if st.button("Process Video"):
            with st.spinner('Processing video...'):
                processed_video_path = process_video(video_path, output_path, model, batch_size)
            
            st.success('Video processed successfully!')
            st.video(processed_video_path)
            
            # Download link
            with open(processed_video_path, 'rb') as file:
                st.download_button(
                    label="Download Processed Video",
                    data=file.read(),
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )

elif mode == "🎥 Live Camera Counting":
    run_live_counter()