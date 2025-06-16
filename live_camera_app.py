import streamlit as st
import cv2
import numpy as np
from models.model_streamlit import model
import time
from datetime import datetime

def get_camera_sources():
    """Get available camera sources"""
    sources = {}
    
    # Check for USB/built-in cameras (0-5)
    for i in range(6):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            sources[f"Camera {i}"] = i
            cap.release()
    
    # Add IP camera option
    sources["IP Camera (Custom URL)"] = "ip_camera"
    sources["RTSP Stream (Custom URL)"] = "rtsp_stream"
    
    return sources

def run_live_counter():
    st.title("🎥 Live Camera People Counter")
    st.markdown("Real-time people counting using any camera source")
    
    # Sidebar for camera settings
    st.sidebar.header("Camera Settings")
    
    # Get available cameras
    camera_sources = get_camera_sources()
    
    if not camera_sources:
        st.error("No cameras detected! Please connect a camera and refresh.")
        return
    
    # Camera selection
    selected_camera = st.sidebar.selectbox(
        "Select Camera Source:",
        options=list(camera_sources.keys())
    )
    
    camera_source = camera_sources[selected_camera]
    
    # Custom URL input for IP cameras
    if camera_source in ["ip_camera", "rtsp_stream"]:
        if camera_source == "ip_camera":
            default_url = "http://192.168.1.100:8080/video"
            st.sidebar.info("Example: http://192.168.1.100:8080/video")
        else:
            default_url = "rtsp://username:password@192.168.1.100:554/stream"
            st.sidebar.info("Example: rtsp://admin:password@192.168.1.100:554/stream")
        
        camera_url = st.sidebar.text_input(
            "Camera URL:",
            value=default_url
        )
        camera_source = camera_url
    
    # Detection settings
    confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.1)
    
    # Message settings
    st.sidebar.header("Notification Settings")
    enable_messages = st.sidebar.checkbox("Enable Console Messages", value=True)
    message_interval = st.sidebar.slider("Message Interval (seconds)", 1, 30, 5)
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_button = st.button("🟢 Start Camera", type="primary")
    with col2:
        stop_button = st.button("🔴 Stop Camera")
    with col3:
        screenshot_button = st.button("📸 Screenshot")
    
    # Initialize session state
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False
    if 'total_count' not in st.session_state:
        st.session_state.total_count = 0
    if 'last_message_time' not in st.session_state:
        st.session_state.last_message_time = 0
    
    # Camera control
    if start_button:
        st.session_state.camera_running = True
        st.success(f"Starting camera: {selected_camera}")
    
    if stop_button:
        st.session_state.camera_running = False
        st.info("Camera stopped")
    
    # Main camera display
    if st.session_state.camera_running:
        # Create placeholders
        frame_placeholder = st.empty()
        info_placeholder = st.empty()
        message_placeholder = st.empty()
        
        try:
            # Initialize camera
            cap = cv2.VideoCapture(camera_source)
            
            if not cap.isOpened():
                st.error(f"Failed to open camera: {camera_source}")
                st.session_state.camera_running = False
                return
            
            # Set camera properties
            if isinstance(camera_source, int):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Main loop
            frame_count = 0
            start_time = time.time()
            
            while st.session_state.camera_running:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from camera")
                    break
                
                frame_count += 1
                
                # Detect people
                results = model(frame, classes=[0], conf=confidence, verbose=False)
                people_count = 0
                
                # Process detections
                for result in results:
                    for box in result.boxes:
                        people_count += 1
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        
                        # Draw centroid
                        centroid_x = int((x1 + x2) / 2)
                        centroid_y = int((y1 + y2) / 2)
                        cv2.circle(frame, (centroid_x, centroid_y), 5, (255, 0, 0), -1)
                        
                        # Add label
                        label = f"Person: {conf:.2f}"
                        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Update total count
                if people_count > st.session_state.total_count:
                    st.session_state.total_count = people_count
                
                # Calculate FPS
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                
                # Add overlay information
                height, width = frame.shape[:2]
                cv2.rectangle(frame, (10, 10), (300, 100), (0, 0, 0), -1)
                cv2.putText(frame, f"People: {people_count}", (20, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Total: {st.session_state.total_count}", (20, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"FPS: {fps:.1f}", (20, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Convert BGR to RGB for Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Display frame
                frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                # Display info
                with info_placeholder.container():
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Current Count", people_count)
                    col2.metric("Max Count", st.session_state.total_count)
                    col3.metric("FPS", f"{fps:.1f}")
                    col4.metric("Camera", selected_camera)
                
                # Send messages
                if enable_messages and people_count > 0:
                    current_time = time.time()
                    if current_time - st.session_state.last_message_time >= message_interval:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        message = f"[{timestamp}] 👥 {people_count} people detected"
                        message_placeholder.success(message)
                        st.session_state.last_message_time = current_time
                
                # Screenshot functionality
                if screenshot_button:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"results/live_screenshot_{timestamp}_count{people_count}.jpg"
                    cv2.imwrite(filename, frame)
                    st.success(f"Screenshot saved: {filename}")
                
                # Small delay to prevent overwhelming
                time.sleep(0.1)
            
            cap.release()
            
        except Exception as e:
            st.error(f"Camera error: {str(e)}")
            st.session_state.camera_running = False
    
    else:
        st.info("Click 'Start Camera' to begin live people counting")
        st.markdown("""
        ### Supported Camera Types:
        - 🖥️ **Built-in webcam** (Laptop cameras)
        - 🔌 **USB cameras** (External webcams)
        - 🌐 **IP cameras** (Network cameras with HTTP stream)
        - 📡 **RTSP streams** (Professional security cameras)
        
        ### Features:
        - ✅ Real-time people detection and counting
        - ✅ Live FPS monitoring
        - ✅ Automatic total count tracking
        - ✅ Customizable confidence threshold
        - ✅ Screenshot capture
        - ✅ Console message notifications
        """)

if __name__ == "__main__":
    run_live_counter()
