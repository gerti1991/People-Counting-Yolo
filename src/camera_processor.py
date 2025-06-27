#!/usr/bin/env python3
"""
Camera Processing Module
Handles live camera functionality for real-time people counting
"""

import cv2
import streamlit as st
import numpy as np
from ultralytics import YOLO
import time

def get_model():
    """Load and return YOLO model"""
    return YOLO("yolov9c.pt")

def get_available_cameras(max_cameras=5):
    """Find available cameras"""
    available_cameras = []
    for i in range(max_cameras):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(i)
                cap.release()
        except:
            continue
    return available_cameras

def run_live_counter():
    """Main live camera counting interface"""
    
    st.markdown("### 🎥 Camera Configuration")
    
    # Camera selection
    available_cameras = get_available_cameras()
    
    if not available_cameras:
        st.error("❌ No cameras detected")
        st.info("💡 Please check your camera connection and try again")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        camera_index = st.selectbox(
            "Select Camera:",
            available_cameras,
            format_func=lambda x: f"Camera {x}",
            help="Choose your camera device"
        )
    
    with col2:
        confidence = st.slider(
            "Detection Confidence:",
            0.1, 1.0, 0.5, 0.1,
            help="Higher values = fewer false positives"
        )
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_button = st.button("🎬 Start Counting", type="primary", use_container_width=True)
    with col2:
        stop_button = st.button("⏹️ Stop", use_container_width=True)
    with col3:
        snapshot_button = st.button("📸 Snapshot", use_container_width=True)
    
    # Live metrics
    st.markdown("### 📊 Live Statistics")
    metrics_cols = st.columns(4)
    
    with metrics_cols[0]:
        people_metric = st.empty()
    with metrics_cols[1]:
        fps_metric = st.empty()
    with metrics_cols[2]:
        total_metric = st.empty()
    with metrics_cols[3]:
        max_metric = st.empty()
    
    # Video display
    video_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Session state for camera control
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False
    if 'total_people_seen' not in st.session_state:
        st.session_state.total_people_seen = 0
    if 'max_people_frame' not in st.session_state:
        st.session_state.max_people_frame = 0
    
    # Start camera
    if start_button:
        st.session_state.camera_running = True
        st.session_state.total_people_seen = 0
        st.session_state.max_people_frame = 0
    
    # Stop camera
    if stop_button:
        st.session_state.camera_running = False
        status_placeholder.success("📹 Camera stopped")
    
    # Camera loop
    if st.session_state.camera_running:
        run_camera_loop(
            camera_index, confidence, video_placeholder, status_placeholder,
            people_metric, fps_metric, total_metric, max_metric, snapshot_button
        )

def run_camera_loop(camera_index, confidence, video_placeholder, status_placeholder,
                   people_metric, fps_metric, total_metric, max_metric, snapshot_button):
    """Run the camera processing loop"""
    
    # Load model
    model = get_model()
    
    # Initialize camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        st.error(f"❌ Cannot access camera {camera_index}")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    status_placeholder.success("🎬 Camera started - Live counting active")
    
    frame_count = 0
    start_time = time.time()
    
    # Main processing loop
    while st.session_state.camera_running:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to read from camera")
            break
        
        frame_count += 1
        
        # Calculate FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Run detection
        results = model(frame, classes=[0], conf=confidence, verbose=False)
        current_people_count = 0
        
        # Process detections
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    current_people_count += 1
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # Add label
                    label = f"Person: {conf:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Update statistics
        st.session_state.total_people_seen += current_people_count
        st.session_state.max_people_frame = max(st.session_state.max_people_frame, current_people_count)
        
        # Add overlay
        overlay_text = f"People: {current_people_count} | FPS: {fps:.1f}"
        cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Update metrics
        people_metric.metric("👥 Current People", current_people_count)
        fps_metric.metric("📊 FPS", f"{fps:.1f}")
        total_metric.metric("📈 Total Detected", st.session_state.total_people_seen)
        max_metric.metric("🔝 Max in Frame", st.session_state.max_people_frame)
        
        # Display video
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # Handle snapshot
        if snapshot_button:
            save_snapshot(frame, current_people_count)
        
        # Small delay to prevent overwhelming
        time.sleep(0.1)
    
    # Cleanup
    cap.release()
    status_placeholder.info("📹 Camera session ended")

def save_snapshot(frame, people_count):
    """Save snapshot with timestamp"""
    import os
    from datetime import datetime
    
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/snapshot_{timestamp}_{people_count}people.jpg"
    
    cv2.imwrite(filename, frame)
    st.success(f"📸 Snapshot saved: {filename}")
    
    # Show download button
    with open(filename, 'rb') as file:
        st.download_button(
            label="📥 Download Snapshot",
            data=file.read(),
            file_name=f"snapshot_{people_count}_people.jpg",
            mime="image/jpeg"
        )
