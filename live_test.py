import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Live People Counter", 
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 Live People Counter - Direct Test")
st.markdown("Real-time people counting using Camera 0")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolov9c.pt")

model = load_model()

# Camera controls
col1, col2 = st.columns(2)
with col1:
    start_button = st.button("🟢 Start Camera", type="primary")
with col2:
    stop_button = st.button("🔴 Stop Camera")

# Initialize session state
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

if start_button:
    st.session_state.camera_active = True

if stop_button:
    st.session_state.camera_active = False

# Camera feed
if st.session_state.camera_active:
    frame_placeholder = st.empty()
    info_placeholder = st.empty()
    
    cap = cv2.VideoCapture(0)
    
    if cap.isOpened():
        while st.session_state.camera_active:
            ret, frame = cap.read()
            if ret:
                # Run YOLO detection
                results = model(frame, classes=[0], conf=0.5, verbose=False)
                people_count = 0
                
                # Draw detections
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            people_count += 1
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            
                            # Draw bounding box
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            
                            # Draw label
                            label = f"Person: {conf:.2f}"
                            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Add count overlay
                cv2.rectangle(frame, (10, 10), (250, 60), (0, 0, 0), -1)
                cv2.putText(frame, f"People Count: {people_count}", (20, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Time: {datetime.now().strftime('%H:%M:%S')}", (20, 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Display frame
                frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                # Display info
                info_placeholder.metric("People Detected", people_count)
                
                time.sleep(0.1)
            else:
                st.error("Failed to read from camera")
                break
        
        cap.release()
    else:
        st.error("Could not open camera. Make sure camera is connected and not in use by another application.")
else:
    st.info("Click 'Start Camera' to begin live people counting")
    st.markdown("""
    ### Instructions:
    1. Click the **🟢 Start Camera** button above
    2. Allow camera access if prompted
    3. You should see live video feed with people detection
    4. Green boxes will appear around detected people
    5. Click **🔴 Stop Camera** to stop the feed
    """)
