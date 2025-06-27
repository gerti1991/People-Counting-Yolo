import streamlit as st
import cv2
import numpy as np
import time
import os
from datetime import datetime
from dotenv import load_dotenv
from models.model import model
from face_recognition_system import FaceRecognitionSystem
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    st.warning("⚠️ face_recognition library not available. Install with: pip install face_recognition")

# Load environment variables
load_dotenv()

class IntegratedPeopleTracker:
    """Integrated system combining YOLO detection with face recognition"""
    
    def __init__(self):
        self.face_system = FaceRecognitionSystem()
        self.tracked_people = []
        self.next_person_id = 1
        self.unique_people_count = 0
        self.recognized_people = {}  # Store recognized people with their info
        self.frame_count = 0
        self.fps = 0.0
        self.last_time = time.time()
        
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def extract_face_from_bbox(self, frame, bbox):
        """Extract face region from YOLO bounding box"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Expand bounding box to focus on head area (face is typically in upper portion)
            height = y2 - y1
            face_region_height = int(height * 0.4)  # Top 40% of person bbox
            
            face_y1 = max(0, y1)
            face_y2 = min(frame.shape[0], y1 + face_region_height)
            face_x1 = max(0, x1)
            face_x2 = min(frame.shape[1], x2)
            
            face_region = frame[face_y1:face_y2, face_x1:face_x2]
            
            if face_region.size > 0:
                return face_region, (face_x1, face_y1, face_x2, face_y2)
            return None, None
            
        except Exception as e:
            return None, None
    
    def process_frame_with_face_recognition(self, frame, confidence=0.5, tracking_distance=80):
        """Process frame with both YOLO detection and face recognition"""
        frame_count = getattr(self, 'frame_count', 0) + 1
        self.frame_count = frame_count
        
        # Run YOLO detection for people
        results = model(frame, classes=[0], conf=confidence, verbose=False)
        
        current_detections = []
        
        # Process each detection
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    # Calculate centroid for tracking
                    centroid_x = int((x1 + x2) / 2)
                    centroid_y = int((y1 + y2) / 2)
                    centroid = (centroid_x, centroid_y)
                    
                    # Extract face region for recognition
                    face_region, face_coords = self.extract_face_from_bbox(frame, [x1, y1, x2, y2])
                    
                    # Try face recognition
                    face_name = "Unknown"
                    face_confidence = 0.0
                    
                    if face_region is not None and face_region.size > 1000:  # Minimum face size
                        recognized_name, person_id, face_conf, face_location = self.face_system.recognize_face(
                            face_region, tolerance=0.6
                        )
                        if recognized_name and recognized_name != "Unknown":
                            face_name = recognized_name
                            face_confidence = face_conf
                    
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'centroid': centroid,
                        'confidence': conf,
                        'face_name': face_name,
                        'face_confidence': face_confidence,
                        'face_coords': face_coords,
                        'frame_count': frame_count
                    }
                    
                    current_detections.append(detection)
        
        # Update tracking with face information
        self.update_tracking_with_faces(current_detections, tracking_distance, frame_count)
        
        return current_detections
    
    def update_tracking_with_faces(self, detections, tracking_distance, frame_count):
        """Update person tracking incorporating face recognition"""
        
        # Match detections with existing tracks
        for detection in detections:
            matched = False
            centroid = detection['centroid']
            face_name = detection['face_name']
            
            # Try to match with existing tracks
            for person in self.tracked_people:
                if self.calculate_distance(centroid, person['last_centroid']) < tracking_distance:
                    # Update existing person
                    person['last_centroid'] = centroid
                    person['last_seen_frame'] = frame_count
                    person['detection_count'] += 1
                    
                    # Update face information if recognized
                    if face_name != "Unknown":
                        if person.get('face_name') == "Unknown" or person.get('face_name') is None:
                            person['face_name'] = face_name
                            person['face_confidence'] = detection['face_confidence']
                            person['face_recognition_frame'] = frame_count
                        elif person.get('face_name') == face_name:
                            # Update confidence if same person
                            person['face_confidence'] = max(
                                person.get('face_confidence', 0), 
                                detection['face_confidence']
                            )
                    
                    matched = True
                    break
            
            if not matched:
                # New person detected
                new_person = {
                    'id': self.next_person_id,
                    'first_seen_frame': frame_count,
                    'last_seen_frame': frame_count,
                    'last_centroid': centroid,
                    'face_name': face_name,
                    'face_confidence': detection['face_confidence'],
                    'detection_count': 1,
                    'face_recognition_frame': frame_count if face_name != "Unknown" else None
                }
                
                self.tracked_people.append(new_person)
                self.unique_people_count += 1
                self.next_person_id += 1
                
                # Add to recognized people if face is known
                if face_name != "Unknown":
                    if face_name not in self.recognized_people:
                        self.recognized_people[face_name] = {
                            'first_seen': datetime.now(),
                            'track_id': new_person['id'],
                            'confidence': detection['face_confidence']
                        }
        
        # Clean up old tracks
        self.tracked_people = [
            p for p in self.tracked_people 
            if frame_count - p['last_seen_frame'] < 150  # 5 seconds at 30fps
        ]
    
    def draw_enhanced_annotations(self, frame, detections):
        """Draw enhanced annotations with face recognition info"""
        
        # Draw current detections
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            face_name = detection['face_name']
            face_conf = detection['face_confidence']
            
            # Choose color based on recognition status
            if face_name != "Unknown":
                color = (0, 255, 0)  # Green for recognized
                label = f"{face_name} ({face_conf:.2%})"
            else:
                color = (255, 255, 0)  # Yellow for detected but not recognized
                label = f"Person ({detection['confidence']:.2%})"
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw face region if available
            if detection['face_coords']:
                fx1, fy1, fx2, fy2 = detection['face_coords']
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 255), 1)
            
            # Add label
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw centroid
            centroid = detection['centroid']
            cv2.circle(frame, centroid, 5, color, -1)
        
        # Draw tracking information for persistent tracks
        for person in self.tracked_people:
            # Only show ID for people seen multiple times
            if person['detection_count'] > 5:
                centroid = person['last_centroid']
                
                # Choose color based on face recognition
                if person.get('face_name') and person['face_name'] != "Unknown":
                    text_color = (0, 255, 0)
                    track_label = f"ID:{person['id']} - {person['face_name']}"
                else:
                    text_color = (255, 0, 255)
                    track_label = f"ID:{person['id']}"
                
                cv2.putText(frame, track_label,
                           (centroid[0] + 10, centroid[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        
        return frame
    
    def get_statistics(self):
        """Get current tracking and recognition statistics"""
        current_people = len([p for p in self.tracked_people 
                            if self.frame_count - p['last_seen_frame'] < 30])
        
        recognized_count = len([p for p in self.tracked_people 
                               if p.get('face_name') and p['face_name'] != "Unknown"])
        
        return {
            'unique_people': self.unique_people_count,
            'current_people': current_people,
            'recognized_people': recognized_count,
            'unknown_people': self.unique_people_count - recognized_count,
            'recognized_names': list(self.recognized_people.keys())
        }
    
    def reset_tracking(self):
        """Reset all tracking data"""
        self.tracked_people = []
        self.next_person_id = 1
        self.unique_people_count = 0
        self.recognized_people = {}
        self.frame_count = 0

def run_integrated_people_tracking():
    """Main interface for integrated people tracking with face recognition"""
    
    st.header("🎯 Integrated People Tracking & Face Recognition")
    st.markdown("**Advanced people counting with face recognition - Know exactly who enters your space!**")
    
    # Initialize system
    if 'integrated_tracker' not in st.session_state:
        st.session_state.integrated_tracker = IntegratedPeopleTracker()
    
    tracker = st.session_state.integrated_tracker
    
    # Camera source selection (reuse from camera.py)
    from camera import get_camera_sources, initialize_camera
    
    st.subheader("📷 Camera Setup")
    camera_sources = get_camera_sources()
    
    if not camera_sources:
        st.error("❌ No camera sources configured!")
        st.info("💡 Configure your cameras in the `.env` file")
        return
    
    selected_camera_name = st.selectbox(
        "Select Camera Source:",
        options=list(camera_sources.keys())
    )
    selected_camera = camera_sources[selected_camera_name]
    
    # Settings
    st.subheader("⚙️ Detection & Recognition Settings")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        confidence = st.slider("YOLO Confidence:", 0.1, 1.0, 0.5, 0.1)
    with col2:
        tracking_distance = st.slider("Tracking Distance:", 30, 150, 80, 10)
    with col3:
        face_tolerance = st.slider("Face Recognition Tolerance:", 0.3, 0.8, 0.6, 0.05)
    
    # Control buttons
    st.subheader("🎮 Controls")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        start_button = st.button("🟢 Start Tracking", type="primary")
    with col2:
        stop_button = st.button("🔴 Stop Tracking")
    with col3:
        reset_button = st.button("🔄 Reset All")
    with col4:
        screenshot_button = st.button("📸 Screenshot")
    
    # Session state management
    if 'integrated_running' not in st.session_state:
        st.session_state.integrated_running = False
    
    if start_button:
        st.session_state.integrated_running = True
    if stop_button:
        st.session_state.integrated_running = False
    if reset_button:
        tracker.reset_tracking()
        st.success("🔄 All tracking data reset!")
    
    # Live metrics
    st.subheader("📊 Live Analytics")
    metrics_container = st.container()
    
    # Video display
    video_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Recognized people sidebar
    recognition_placeholder = st.empty()
    
    # Main tracking loop
    if st.session_state.integrated_running:
        cap, error_msg = initialize_camera(selected_camera)
        if cap is None:
            st.error(f"❌ {error_msg}")
            return
        
        status_placeholder.success(f"🎬 Integrated tracking active on {selected_camera_name}")
        
        while st.session_state.integrated_running:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to read from camera")
                break
            
            # Process frame with integrated system
            detections = tracker.process_frame_with_face_recognition(
                frame, confidence, tracking_distance
            )
            
            # Draw annotations
            annotated_frame = tracker.draw_enhanced_annotations(frame.copy(), detections)
            
            # Get statistics
            stats = tracker.get_statistics()
            
            # Update metrics
            with metrics_container:
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("🚶‍♂️ Total Unique", stats['unique_people'])
                with col2:
                    st.metric("👥 Currently Visible", stats['current_people'])
                with col3:
                    st.metric("✅ Recognized", stats['recognized_people'])
                with col4:
                    st.metric("❓ Unknown", stats['unknown_people'])
                with col5:
                    fps = getattr(tracker, 'fps', 0)
                    st.metric("📊 FPS", f"{fps:.1f}")
            
            # Show recognized people
            if stats['recognized_names']:
                with recognition_placeholder:
                    st.success(f"👤 Recognized People: {', '.join(stats['recognized_names'])}")
            
            # Add overlay to frame
            cv2.rectangle(annotated_frame, (10, 10), (500, 120), (0, 0, 0), -1)
            cv2.putText(annotated_frame, f"Unique People: {stats['unique_people']}", 
                       (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Recognized: {stats['recognized_people']}", 
                       (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(annotated_frame, f"Current: {stats['current_people']}", 
                       (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display frame
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Handle screenshot
            if screenshot_button:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"results/integrated_tracking_{timestamp}.jpg"
                os.makedirs('results', exist_ok=True)
                cv2.imwrite(filename, annotated_frame)
                st.success(f"📸 Screenshot saved: {stats['unique_people']} unique, {stats['recognized_people']} recognized")
            
            # Update FPS
            if hasattr(tracker, 'last_time'):
                current_time = time.time()
                tracker.fps = 1.0 / (current_time - tracker.last_time)
            tracker.last_time = time.time()
            
            time.sleep(0.03)  # ~30 FPS
        
        cap.release()
        status_placeholder.info("📹 Integrated tracking stopped")
    
    # Show detailed tracking results when stopped
    if not st.session_state.integrated_running and tracker.unique_people_count > 0:
        st.subheader("📈 Session Summary")
        
        stats = tracker.get_statistics()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🚶‍♂️ Total Unique People", stats['unique_people'])
            st.metric("✅ People Recognized", stats['recognized_people'])
        
        with col2:
            st.metric("❓ Unknown People", stats['unknown_people'])
            if stats['recognized_people'] > 0:
                recognition_rate = (stats['recognized_people'] / stats['unique_people']) * 100
                st.metric("🎯 Recognition Rate", f"{recognition_rate:.1f}%")
        
        # Show recognized people details
        if tracker.recognized_people:
            st.subheader("👤 Recognized People This Session")
            for name, info in tracker.recognized_people.items():
                st.write(f"**{name}** - First seen: {info['first_seen'].strftime('%H:%M:%S')}, "
                        f"Confidence: {info['confidence']:.2%}")
