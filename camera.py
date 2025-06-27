import streamlit as st
import cv2
import numpy as np
import time
import os
from datetime import datetime
from dotenv import load_dotenv
from models.model import model

# Load environment variables
load_dotenv()

def get_camera_sources():
    """Get all available camera sources from environment variables"""
    camera_sources = {}
    
    # USB/Built-in cameras
    for key, value in os.environ.items():
        if key.startswith('USB_CAMERA_'):
            try:
                camera_id = int(value)
                # Test if camera is accessible
                cap = cv2.VideoCapture(camera_id)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        camera_sources[f"📷 USB Camera {key.split('_')[-1]}"] = camera_id
                cap.release()
            except (ValueError, Exception):
                pass
    
    # IP Cameras
    for key, value in os.environ.items():
        if key.startswith('IP_CAMERA_'):
            camera_name = f"🌐 IP Camera {key.split('_')[-1]}"
            camera_sources[camera_name] = value
    
    # RTSP Streams
    for key, value in os.environ.items():
        if key.startswith('RTSP_CAMERA_'):
            camera_name = f"📡 RTSP Camera {key.split('_')[-1]}"
            camera_sources[camera_name] = value
    
    # Phone Cameras
    for key, value in os.environ.items():
        if key.startswith('PHONE_CAMERA_'):
            camera_name = f"📱 Phone Camera {key.split('_')[-1]}"
            camera_sources[camera_name] = value
    
    # ONVIF Cameras
    for key, value in os.environ.items():
        if key.startswith('ONVIF_CAMERA_'):
            camera_name = f"🔒 ONVIF Camera {key.split('_')[-1]}"
            camera_sources[camera_name] = value
    
    # Custom Cameras
    for key, value in os.environ.items():
        if key.startswith('CUSTOM_CAMERA_'):
            camera_name = f"⚙️ Custom Camera {key.split('_')[-1]}"
            camera_sources[camera_name] = value
    
    return camera_sources

def test_camera_source(source):
    """Test if a camera source is accessible"""
    try:
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            return ret and frame is not None
        return False
    except Exception:
        return False

def initialize_camera(source, timeout=10):
    """Initialize camera with enhanced error handling and timeout"""
    try:
        # Convert string numbers to int for USB cameras
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        
        st.info(f"🔄 Connecting to camera: {source}")
        
        # Create camera capture with timeout handling
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            return None, f"Cannot open camera source: {source}"
        
        # Set camera properties from environment
        width = int(os.getenv('CAMERA_WIDTH', 640))
        height = int(os.getenv('CAMERA_HEIGHT', 480))
        fps = int(os.getenv('CAMERA_FPS', 30))
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, int(os.getenv('FRAME_BUFFER_SIZE', 1)))
        
        # Test frame capture
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return None, f"Cannot read frames from camera: {source}"
        
        st.success(f"✅ Camera connected: {frame.shape[1]}x{frame.shape[0]}")
        return cap, None
        
    except Exception as e:
        return None, f"Camera initialization error: {str(e)}"

def run_live_counter():
    """Enhanced live camera counting with flexible camera sources"""
    
    st.header("🎥 Live Camera - Universal Camera Support")
    st.markdown("Real-time people counting with **unique person tracking** - supports USB, IP, RTSP, Phone cameras and more!")
    
    # Camera source selection
    st.subheader("📷 Camera Source Selection")
    
    # Get available cameras from .env
    camera_sources = get_camera_sources()
    
    if not camera_sources:
        st.error("❌ No camera sources configured!")
        st.info("💡 Configure your cameras in the `.env` file. See `.env.example` for examples.")
        
        with st.expander("� Quick Setup Guide"):
            st.markdown("""
            **1. Edit the `.env` file to add your cameras:**
            ```
            # USB Cameras
            USB_CAMERA_0=0
            USB_CAMERA_1=1
            
            # IP Cameras  
            IP_CAMERA_1=http://192.168.1.100:8080/video
            
            # RTSP Cameras
            RTSP_CAMERA_1=rtsp://admin:password@192.168.1.200:554/stream1
            
            # Phone Cameras (IP Webcam app)
            PHONE_CAMERA_1=http://192.168.1.50:8080/video
            ```
            
            **2. Restart the application**
            
            **3. Your cameras will appear in the dropdown**
            """)
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_camera_name = st.selectbox(
            "Select Camera Source:",
            options=list(camera_sources.keys()),
            help="Choose your camera source from configured options"
        )
        selected_camera = camera_sources[selected_camera_name]
    
    with col2:
        # Test camera button
        if st.button("🔍 Test Camera", help="Test if selected camera is accessible"):
            with st.spinner("Testing camera connection..."):
                if test_camera_source(selected_camera):
                    st.success("✅ Camera is accessible!")
                else:
                    st.error("❌ Cannot access camera. Check your configuration.")
    
    # Manual camera input
    with st.expander("⚙️ Manual Camera Input"):
        manual_source = st.text_input(
            "Custom Camera Source:",
            placeholder="Enter camera URL or index (e.g., 0, http://192.168.1.100:8080/video)",
            help="Manually enter a camera source if not in your .env file"
        )
        if manual_source:
            if st.button("🔍 Test Manual Source"):
                with st.spinner("Testing manual camera source..."):
                    # Try to convert to int if it's a number
                    try:
                        test_source = int(manual_source)
                    except ValueError:
                        test_source = manual_source
                    
                    if test_camera_source(test_source):
                        st.success("✅ Manual camera source is accessible!")
                        selected_camera = test_source
                        selected_camera_name = f"⚙️ Manual: {manual_source}"
                    else:
                        st.error("❌ Cannot access manual camera source.")
    
    # Camera settings
    st.subheader("⚙️ Detection Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        confidence = st.slider(
            "Detection Confidence:",
            0.1, 1.0, 
            float(os.getenv('DEFAULT_CONFIDENCE', 0.5)), 0.1,
            help="Higher values = fewer false positives"
        )
    
    with col2:
        tracking_distance = st.slider(
            "Tracking Distance:",
            30, 150, 
            int(os.getenv('DEFAULT_TRACKING_DISTANCE', 80)), 10,
            help="Max distance for tracking same person"
        )
    
    with col3:
        camera_timeout = st.slider(
            "Connection Timeout:",
            5, 30, 
            int(os.getenv('CAMERA_TIMEOUT', 10)), 1,
            help="Timeout for camera connection (seconds)"
        )
    
    # Control buttons
    st.subheader("🎮 Camera Controls")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        start_button = st.button("🟢 Start Counting", type="primary", use_container_width=True)
    with col2:
        stop_button = st.button("🔴 Stop Camera", use_container_width=True)
    with col3:
        reset_button = st.button("🔄 Reset Count", use_container_width=True)
    with col4:
        screenshot_button = st.button("📸 Screenshot", use_container_width=True)
    
    # Initialize session state for unique tracking
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False
    if 'unique_people_count' not in st.session_state:
        st.session_state.unique_people_count = 0
    if 'tracked_people' not in st.session_state:
        st.session_state.tracked_people = []
    if 'next_person_id' not in st.session_state:
        st.session_state.next_person_id = 1
    if 'max_people_in_frame' not in st.session_state:
        st.session_state.max_people_in_frame = 0
    if 'total_detections' not in st.session_state:
        st.session_state.total_detections = 0
    
    # Camera control
    if start_button:
        st.session_state.camera_running = True
        st.success("🎬 Camera started - Unique people tracking active!")
    
    if stop_button:
        st.session_state.camera_running = False
        st.info("📹 Camera stopped")
    
    if reset_button:
        st.session_state.unique_people_count = 0
        st.session_state.tracked_people = []
        st.session_state.next_person_id = 1
        st.session_state.max_people_in_frame = 0
        st.session_state.total_detections = 0
        st.success("🔄 Count reset - Starting fresh!")
    
    # Live metrics
    st.subheader("📊 Live Statistics")
    metrics_cols = st.columns(4)
    
    with metrics_cols[0]:
        unique_metric = st.empty()
    with metrics_cols[1]:
        current_metric = st.empty()
    with metrics_cols[2]:
        max_metric = st.empty()
    with metrics_cols[3]:
        fps_metric = st.empty()
    
    # Video display and status
    video_placeholder = st.empty()
    status_placeholder = st.empty()
      # Run camera if active
    if st.session_state.camera_running:
        run_unique_tracking_camera(
            selected_camera, selected_camera_name, confidence, tracking_distance,
            video_placeholder, status_placeholder,
            unique_metric, current_metric, max_metric, fps_metric,
            screenshot_button, camera_timeout
        )

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def is_new_person(centroid, tracked_people, threshold, current_frame):
    """Check if centroid represents a new person or update existing"""
    for person in tracked_people:
        if calculate_distance(centroid, person['last_centroid']) < threshold:
            # Update existing person
            person['last_centroid'] = centroid
            person['last_seen_frame'] = current_frame
            return False, person['id']
    return True, None

def cleanup_old_tracks(tracked_people, current_frame, max_gap=150):
    """Remove people who haven't been seen for a while (5 seconds at 30fps)"""
    return [p for p in tracked_people if current_frame - p['last_seen_frame'] < max_gap]

def run_unique_tracking_camera(camera_source, camera_name, confidence, tracking_distance, 
                              video_placeholder, status_placeholder,
                              unique_metric, current_metric, max_metric, fps_metric,
                              screenshot_button, timeout=10):
    """Run camera with unique person tracking and enhanced source support"""
    
    # Initialize camera with enhanced error handling
    cap, error_msg = initialize_camera(camera_source, timeout)
    if cap is None:
        st.error(f"❌ {error_msg}")
        status_placeholder.error(f"Camera connection failed: {camera_name}")
        return
    
    status_placeholder.success(f"🎬 Camera active - Tracking unique people from {camera_name}")
    
    frame_count = 0
    start_time = time.time()
    connection_errors = 0
    max_connection_errors = 5
    
    # Main camera loop
    while st.session_state.camera_running:
        ret, frame = cap.read()
        if not ret:
            connection_errors += 1
            if connection_errors >= max_connection_errors:
                st.error(f"❌ Too many connection errors. Camera disconnected: {camera_name}")
                break
            else:
                st.warning(f"⚠️ Frame read error ({connection_errors}/{max_connection_errors})")
                time.sleep(0.1)
                continue
        
        # Reset connection error counter on successful read
        connection_errors = 0
        frame_count += 1
        
        # Calculate FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Run YOLO detection
        results = model(frame, classes=[0], conf=confidence, verbose=False)
        
        current_people_count = 0
        current_centroids = []
        
        # Process detections
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    current_people_count += 1
                    st.session_state.total_detections += 1
                    
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    # Calculate centroid
                    centroid_x = int((x1 + x2) / 2)
                    centroid_y = int((y1 + y2) / 2)
                    centroid = (centroid_x, centroid_y)
                    current_centroids.append(centroid)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # Draw centroid
                    cv2.circle(frame, centroid, 5, (255, 0, 0), -1)
                    
                    # Add confidence label
                    label = f"Person: {conf:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Track unique people
        for centroid in current_centroids:
            is_new, person_id = is_new_person(centroid, st.session_state.tracked_people, tracking_distance, frame_count)
            if is_new:
                # New person detected
                new_person = {
                    'id': st.session_state.next_person_id,
                    'first_seen_frame': frame_count,
                    'last_seen_frame': frame_count,
                    'last_centroid': centroid,
                    'first_seen_time': time.time()
                }
                st.session_state.tracked_people.append(new_person)
                st.session_state.unique_people_count += 1
                st.session_state.next_person_id += 1
        
        # Cleanup old tracks
        st.session_state.tracked_people = cleanup_old_tracks(st.session_state.tracked_people, frame_count)
        
        # Update statistics
        st.session_state.max_people_in_frame = max(st.session_state.max_people_in_frame, current_people_count)
        
        # Add overlay information
        overlay_height = 140
        cv2.rectangle(frame, (10, 10), (400, overlay_height), (0, 0, 0), -1)
        
        # Display statistics on frame
        cv2.putText(frame, f"Current people: {current_people_count}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Unique people: {st.session_state.unique_people_count}", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Max in frame: {st.session_state.max_people_in_frame}", (20, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw tracking IDs for active people
        for person in st.session_state.tracked_people:
            if frame_count - person['last_seen_frame'] < 30:  # Show ID for recent people
                cv2.putText(frame, f"ID:{person['id']}", 
                           (person['last_centroid'][0] + 10, person['last_centroid'][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # Update metrics
        unique_metric.metric("🚶‍♂️ Unique People", st.session_state.unique_people_count, 
                           help="Total unique individuals detected")
        current_metric.metric("👥 Current Frame", current_people_count,
                            help="People currently visible")
        max_metric.metric("🔝 Max in Frame", st.session_state.max_people_in_frame,
                        help="Maximum people seen simultaneously")
        fps_metric.metric("📊 FPS", f"{fps:.1f}",
                        help="Frames per second")
        
        # Display video
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
          # Handle screenshot
        if screenshot_button:
            save_unique_screenshot(frame, st.session_state.unique_people_count, current_people_count, camera_name)
        
        # Small delay to prevent overwhelming
        time.sleep(0.03)  # ~30 FPS max
    
    # Cleanup
    cap.release()
    status_placeholder.info(f"📹 Camera stopped - Final count: {st.session_state.unique_people_count} unique people")

def save_unique_screenshot(frame, unique_count, current_count, camera_name="Unknown"):
    """Save screenshot with unique counting info and camera source"""
    import os
    
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Clean camera name for filename
    clean_camera_name = "".join(c for c in camera_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
    clean_camera_name = clean_camera_name.replace(' ', '_')
    
    filename = f"results/unique_count_{timestamp}_{unique_count}unique_{current_count}current_{clean_camera_name}.jpg"
    
    cv2.imwrite(filename, frame)
    st.success(f"📸 Screenshot saved from {camera_name}: {unique_count} unique people, {current_count} currently visible")
    
    # Show download button
    with open(filename, 'rb') as file:
        st.download_button(
            label="📥 Download Screenshot",
            data=file.read(),
            file_name=f"unique_count_{unique_count}_people_{clean_camera_name}.jpg",
            mime="image/jpeg"
        )
