import streamlit as st
import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
import pickle
from PIL import Image, ImageEnhance, ImageFilter
import uuid
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib

# Try to import face_recognition, fallback to OpenCV if not available
FACE_RECOGNITION_AVAILABLE = True
try:
    import face_recognition
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    st.warning("⚠️ face_recognition library not available. Using OpenCV fallback mode.")
    st.info("💡 For full functionality, install: pip install face_recognition")

class FaceRecognitionSystem:
    def __init__(self):
        self.face_encodings_file = 'data/face_encodings.pkl'
        self.face_database_file = 'data/face_database.json'
        self.models_dir = 'models/face_models'
        self.faces_dir = 'data/registered_faces'
        
        # Create directories
        os.makedirs('data', exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.faces_dir, exist_ok=True)        # Initialize face detection
        try:
            import cv2
            # Try different ways to get the cascade file
            cascade_file = None
            try:
                cascade_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_file)
            except:
                # Alternative paths
                possible_paths = [
                    'haarcascade_frontalface_default.xml',
                    os.path.join(cv2.__path__[0], 'data', 'haarcascade_frontalface_default.xml'),
                    '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        self.face_cascade = cv2.CascadeClassifier(path)
                        break
                else:
                    self.face_cascade = cv2.CascadeClassifier()
        except Exception as e:
            st.warning(f"Could not load face cascade: {e}")
            self.face_cascade = None
          # Initialize face database
        self.load_face_database()
        self.load_face_model()
    
    def detect_faces_opencv(self, image):
        """Detect faces using OpenCV as fallback"""
        if self.face_cascade is None:
            return []
            
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        face_locations = []
        for (x, y, w, h) in faces:
            # Convert to face_recognition format (top, right, bottom, left)
            face_locations.append((y, x + w, y + h, x))
        
        return face_locations
    
    def extract_face_features_opencv(self, image, face_location):
        """Extract simple face features using OpenCV (fallback)"""
        try:
            top, right, bottom, left = face_location
            face_image = image[top:bottom, left:right]
            
            if face_image.size == 0:
                return None
            
            # Resize to standard size
            face_image = cv2.resize(face_image, (100, 100))
            
            # Convert to grayscale and flatten for simple comparison
            gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Simple feature: histogram of gray values
            hist = cv2.calcHist([gray_face], [0], None, [256], [0, 256])
            
            # Normalize and flatten
            hist = hist.flatten()
            hist = hist / (hist.sum() + 1e-10)  # Normalize
            
            return hist
            
        except Exception as e:
            st.error(f"Error extracting face features: {e}")
            return None
    
    def load_face_database(self):
        """Load face database from file"""
        try:
            if os.path.exists(self.face_database_file):
                with open(self.face_database_file, 'r') as f:
                    self.face_database = json.load(f)
            else:
                self.face_database = {}
        except Exception as e:
            st.error(f"Error loading face database: {e}")
            self.face_database = {}
    
    def save_face_database(self):
        """Save face database to file"""
        try:
            with open(self.face_database_file, 'w') as f:
                json.dump(self.face_database, f, indent=2)
        except Exception as e:
            st.error(f"Error saving face database: {e}")
    
    def load_face_model(self):
        """Load trained face recognition model"""
        try:
            if os.path.exists(self.face_encodings_file):
                with open(self.face_encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_encodings = data.get('encodings', [])
                    self.known_names = data.get('names', [])
                    
                # Load SVM model if available
                svm_model_path = os.path.join(self.models_dir, 'face_classifier.pkl')
                if os.path.exists(svm_model_path):
                    self.face_classifier = joblib.load(svm_model_path)
                else:
                    self.face_classifier = None
            else:
                self.known_encodings = []
                self.known_names = []
                self.face_classifier = None
        except Exception as e:
            st.error(f"Error loading face model: {e}")
            self.known_encodings = []
            self.known_names = []
            self.face_classifier = None
    
    def save_face_model(self):
        """Save face encodings and train SVM model"""
        try:
            # Save encodings
            data = {
                'encodings': self.known_encodings,
                'names': self.known_names
            }
            with open(self.face_encodings_file, 'wb') as f:
                pickle.dump(data, f)
              # Train SVM classifier if we have enough data
            if len(self.known_encodings) > 0:
                le = LabelEncoder()
                labels = le.fit_transform(self.known_names)
                
                # Convert to numpy array for sklearn
                encodings_array = np.array(self.known_encodings)
                
                # Train SVM
                self.face_classifier = SVC(kernel='linear', probability=True)
                self.face_classifier.fit(encodings_array, labels)
                
                # Save SVM model and label encoder
                joblib.dump(self.face_classifier, os.path.join(self.models_dir, 'face_classifier.pkl'))
                joblib.dump(le, os.path.join(self.models_dir, 'label_encoder.pkl'))
                
        except Exception as e:
            st.error(f"Error saving face model: {e}")
    
    def augment_image(self, image):
        """Apply data augmentation to face images"""
        augmented_images = []
        
        # Original image
        augmented_images.append(image)
        
        # Brightness variations
        enhancer = ImageEnhance.Brightness(image)
        augmented_images.append(enhancer.enhance(0.8))  # Darker
        augmented_images.append(enhancer.enhance(1.2))  # Brighter
        
        # Contrast variations
        enhancer = ImageEnhance.Contrast(image)
        augmented_images.append(enhancer.enhance(0.8))  # Lower contrast
        augmented_images.append(enhancer.enhance(1.2))  # Higher contrast
          # Slight blur
        augmented_images.append(image.filter(ImageFilter.GaussianBlur(radius=0.5)))        # Flip horizontally (skip if PIL version doesn't support it)
        try:
            flipped = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            augmented_images.append(flipped)
        except:
            # Skip horizontal flip if not supported
            pass
        
        return augmented_images
    
    def extract_face_encoding(self, image):
        """Extract face encoding from image - supports both face_recognition and OpenCV fallback"""
        try:
            # Convert PIL to numpy
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if FACE_RECOGNITION_AVAILABLE:
                # Use face_recognition library
                face_locations = face_recognition.face_locations(image)
                if len(face_locations) > 0:
                    face_encodings = face_recognition.face_encodings(image, face_locations)
                    if len(face_encodings) > 0:
                        return face_encodings[0], face_locations[0]
            else:
                # Use OpenCV fallback
                if self.face_cascade is not None:
                    face_locations = self.detect_faces_opencv(image)
                    if len(face_locations) > 0:
                        features = self.extract_face_features_opencv(image, face_locations[0])
                        if features is not None:
                            return features, face_locations[0]
            
            return None, None
        except Exception as e:
            st.error(f"Error extracting face encoding: {e}")
            return None, None
    
    def register_new_person(self, name, face_images):
        """Register a new person with multiple face images"""
        try:
            person_id = str(uuid.uuid4())
            person_dir = os.path.join(self.faces_dir, person_id)
            os.makedirs(person_dir, exist_ok=True)
            
            encodings = []
            saved_images = []
            
            for i, image in enumerate(face_images):
                # Save original image
                image_path = os.path.join(person_dir, f"{name}_{i}.jpg")
                if isinstance(image, Image.Image):
                    image.save(image_path)
                else:
                    cv2.imwrite(image_path, image)
                saved_images.append(image_path)
                
                # Extract encoding
                encoding, location = self.extract_face_encoding(image)
                if encoding is not None:
                    encodings.append(encoding.tolist())
                    self.known_encodings.append(encoding)
                    self.known_names.append(name)
                
                # Apply augmentation and extract more encodings
                if isinstance(image, np.ndarray):
                    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                else:
                    pil_image = image
                
                augmented_images = self.augment_image(pil_image)
                for j, aug_image in enumerate(augmented_images[1:], 1):  # Skip original
                    aug_encoding, _ = self.extract_face_encoding(aug_image)
                    if aug_encoding is not None:
                        encodings.append(aug_encoding.tolist())
                        self.known_encodings.append(aug_encoding)
                        self.known_names.append(name)
                    
                    # Save augmented image
                    aug_path = os.path.join(person_dir, f"{name}_{i}_aug_{j}.jpg")
                    aug_image.save(aug_path)
                    saved_images.append(aug_path)
            
            # Update database
            self.face_database[person_id] = {
                'name': name,
                'registered_date': datetime.now().isoformat(),
                'encodings_count': len(encodings),
                'image_paths': saved_images,
                'encodings': encodings
            }
            
            # Save everything
            self.save_face_database()
            self.save_face_model()
            
            return True, f"Successfully registered {name} with {len(encodings)} face encodings"
            
        except Exception as e:
            return False, f"Error registering person: {e}"
    
    def recognize_face(self, image, tolerance=0.6):
        """Recognize face in image - supports both face_recognition and OpenCV fallback"""
        try:
            encoding, location = self.extract_face_encoding(image)
            if encoding is None:
                return None, None, 0.0, None
            
            if len(self.known_encodings) == 0:
                return "Unknown", None, 0.0, location
            
            if FACE_RECOGNITION_AVAILABLE:
                # Compare with known faces using face_recognition
                distances = face_recognition.face_distance(self.known_encodings, encoding)
                best_match_index = np.argmin(distances)
                
                if distances[best_match_index] <= tolerance:
                    name = self.known_names[best_match_index]
                    confidence = 1 - distances[best_match_index]
                    
                    # Find person ID
                    person_id = None
                    for pid, person_data in self.face_database.items():
                        if person_data['name'] == name:
                            person_id = pid
                            break
                    
                    return name, person_id, confidence, location
                else:
                    return "Unknown", None, 0.0, location
            else:
                # Use simple distance comparison for OpenCV fallback
                min_distance = float('inf')
                best_match_index = -1
                
                for i, known_encoding in enumerate(self.known_encodings):
                    # Simple euclidean distance for histogram comparison
                    distance = np.linalg.norm(encoding - known_encoding)
                    if distance < min_distance:
                        min_distance = distance
                        best_match_index = i
                
                # Adjust tolerance for histogram comparison (different scale)
                cv_tolerance = tolerance * 100  
                
                if best_match_index >= 0 and min_distance <= cv_tolerance:
                    name = self.known_names[best_match_index]
                    confidence = max(0.0, float(1.0 - (min_distance / cv_tolerance)))
                    
                    # Find person ID
                    person_id = None
                    for pid, person_data in self.face_database.items():
                        if person_data['name'] == name:
                            person_id = pid
                            break
                    
                    return name, person_id, confidence, location
                else:
                    return "Unknown", None, 0.0, location
                
        except Exception as e:
            st.error(f"Error recognizing face: {e}")
            return None, None, 0.0, None
    
    def get_registered_people(self):
        """Get list of all registered people"""
        people = []
        for person_id, person_data in self.face_database.items():
            people.append({
                'id': person_id,
                'name': person_data['name'],
                'registered_date': person_data['registered_date'],
                'encodings_count': person_data['encodings_count']
            })
        return people
    
    def delete_person(self, person_id):
        """Delete a registered person"""
        try:
            if person_id in self.face_database:
                person_data = self.face_database[person_id]
                name = person_data['name']
                
                # Remove from encodings
                indices_to_remove = [i for i, n in enumerate(self.known_names) if n == name]
                for i in sorted(indices_to_remove, reverse=True):
                    del self.known_encodings[i]
                    del self.known_names[i]
                
                # Remove images
                person_dir = os.path.join(self.faces_dir, person_id)
                if os.path.exists(person_dir):
                    import shutil
                    shutil.rmtree(person_dir)
                
                # Remove from database
                del self.face_database[person_id]
                
                # Save changes
                self.save_face_database()
                self.save_face_model()
                
                return True, f"Successfully deleted {name}"
            else:
                return False, "Person not found"
                
        except Exception as e:
            return False, f"Error deleting person: {e}"

def draw_face_border(frame, location, color=(0, 255, 0), thickness=2):
    """Draw border around detected face"""
    top, right, bottom, left = location
    cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
    return frame

def run_face_recognition_system():
    """Main face recognition interface"""
    st.header("👤 Advanced Face Recognition System")
    
    if not FACE_RECOGNITION_AVAILABLE:
        st.warning("⚠️ **OpenCV Fallback Mode Active**")
        st.info("💡 For best results, install face_recognition: `pip install face_recognition`")
        st.markdown("Currently using basic OpenCV face detection with limited accuracy.")
    else:
        st.success("✅ **Full Face Recognition Mode Active**")
    
    st.markdown("Recognize people, register new faces, and manage your face database")
    
    # Initialize face recognition system
    if 'face_system' not in st.session_state:
        st.session_state.face_system = FaceRecognitionSystem()
    
    face_system = st.session_state.face_system
    
    # Mode selection
    mode = st.selectbox(
        "Select Mode:",
        ["🔍 Live Face Recognition", "➕ Register New Person", "👥 Manage Database"]
    )
    
    if mode == "🔍 Live Face Recognition":
        run_live_face_recognition(face_system)
    elif mode == "➕ Register New Person":
        run_face_registration(face_system)
    elif mode == "👥 Manage Database":
        run_database_management(face_system)

def run_live_face_recognition(face_system):
    """Live face recognition interface"""
    st.subheader("🔍 Live Face Recognition")
    
    # Get camera sources (reuse from camera.py)
    from camera import get_camera_sources, initialize_camera
    
    camera_sources = get_camera_sources()
    
    if not camera_sources:
        st.error("❌ No camera sources configured!")
        return
    
    selected_camera_name = st.selectbox(
        "Select Camera:",
        options=list(camera_sources.keys())
    )
    selected_camera = camera_sources[selected_camera_name]
    
    # Settings
    col1, col2 = st.columns(2)
    with col1:
        tolerance = st.slider("Recognition Tolerance:", 0.3, 0.8, 0.6, 0.05,
                             help="Lower = stricter matching")
    with col2:
        show_confidence = st.checkbox("Show Confidence Scores", True)
    
    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        start_recognition = st.button("🟢 Start Recognition", type="primary")
    with col2:
        stop_recognition = st.button("🔴 Stop Recognition")
    
    if 'face_recognition_running' not in st.session_state:
        st.session_state.face_recognition_running = False
    
    if start_recognition:
        st.session_state.face_recognition_running = True
    if stop_recognition:
        st.session_state.face_recognition_running = False
    
    # Recognition display
    video_placeholder = st.empty()
    info_placeholder = st.empty()
    
    if st.session_state.face_recognition_running:
        cap, error_msg = initialize_camera(selected_camera)
        if cap is None:
            st.error(f"❌ {error_msg}")
            return
        
        recognized_people = set()
        
        while st.session_state.face_recognition_running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Recognize faces
            name, person_id, confidence, location = face_system.recognize_face(frame, tolerance)
            
            if location is not None:
                # Draw face border
                if name and name != "Unknown":
                    color = (0, 255, 0)  # Green for known
                    recognized_people.add(name)
                else:
                    color = (0, 0, 255)  # Red for unknown
                
                frame = draw_face_border(frame, location, color)
                
                # Add name and confidence
                top, right, bottom, left = location
                label = name if name else "Unknown"
                if show_confidence and confidence > 0:
                    label += f" ({confidence:.2%})"
                
                cv2.putText(frame, label, (left, top - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Display frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Update info
            if recognized_people:
                info_placeholder.success(f"✅ Recognized: {', '.join(recognized_people)}")
            else:
                info_placeholder.info("🔍 Scanning for faces...")
            
            time.sleep(0.03)
        
        cap.release()

def run_face_registration(face_system):
    """Face registration interface"""
    st.subheader("➕ Register New Person")
    
    # Person information
    name = st.text_input("Person Name:", placeholder="Enter full name")
    
    if not name:
        st.warning("⚠️ Please enter a name to continue")
        return
    
    # Check if name already exists
    existing_people = [p['name'] for p in face_system.get_registered_people()]
    if name in existing_people:
        st.error(f"❌ Person '{name}' is already registered!")
        return
    
    st.markdown("### 📸 Capture Face Images")
    st.info("💡 For best results, capture 3-5 images: front view, slight left turn, slight right turn")
    
    # Camera setup for registration
    from camera import get_camera_sources, initialize_camera
    
    camera_sources = get_camera_sources()
    if camera_sources:
        selected_camera_name = st.selectbox(
            "Select Camera for Registration:",
            options=list(camera_sources.keys())
        )
        selected_camera = camera_sources[selected_camera_name]
        
        # Face capture interface
        col1, col2 = st.columns(2)
        with col1:
            start_capture = st.button("📷 Start Face Capture", type="primary")
        with col2:
            capture_photo = st.button("📸 Capture Photo")
        
        if 'face_capture_running' not in st.session_state:
            st.session_state.face_capture_running = False
        if 'captured_faces' not in st.session_state:
            st.session_state.captured_faces = []
        
        if start_capture:
            st.session_state.face_capture_running = True
        
        # Live capture interface
        video_placeholder = st.empty()
        instruction_placeholder = st.empty()
        
        if st.session_state.face_capture_running:
            cap, error_msg = initialize_camera(selected_camera)
            if cap is None:
                st.error(f"❌ {error_msg}")
                return
            
            # Instructions based on number of captured faces
            num_captured = len(st.session_state.captured_faces)
            instructions = [
                "📸 Look straight at the camera",
                "📸 Turn your head slightly to the left",
                "📸 Turn your head slightly to the right",
                "📸 Look slightly up",
                "📸 Look slightly down"
            ]
            
            if num_captured < len(instructions):
                instruction_placeholder.info(f"Step {num_captured + 1}/5: {instructions[num_captured]}")
            else:
                instruction_placeholder.success("✅ Enough photos captured! You can register now.")
            
            while st.session_state.face_capture_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect face and draw border
                _, location = face_system.extract_face_encoding(frame)
                if location is not None:
                    frame = draw_face_border(frame, location, (0, 255, 0))
                    
                    # Show capture indicator
                    cv2.putText(frame, "Face Detected - Ready to Capture", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No Face Detected", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Capture photo if button pressed
                if capture_photo and location is not None:
                    st.session_state.captured_faces.append(frame.copy())
                    st.success(f"📸 Photo {len(st.session_state.captured_faces)} captured!")
                    
                    if len(st.session_state.captured_faces) >= 5:
                        st.session_state.face_capture_running = False
                        break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                time.sleep(0.03)
            
            cap.release()
    
    # Show captured faces
    if st.session_state.captured_faces:
        st.markdown("### 📷 Captured Face Images")
        cols = st.columns(min(5, len(st.session_state.captured_faces)))
        
        for i, face_img in enumerate(st.session_state.captured_faces):
            with cols[i % 5]:
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                st.image(face_rgb, caption=f"Photo {i+1}", use_container_width=True)
        
        # Registration button
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ Register Person", type="primary"):
                with st.spinner("🔄 Processing and registering face..."):
                    success, message = face_system.register_new_person(name, st.session_state.captured_faces)
                    
                    if success:
                        st.success(f"🎉 {message}")
                        st.session_state.captured_faces = []
                        st.balloons()
                    else:
                        st.error(f"❌ {message}")
        
        with col2:
            if st.button("🔄 Retake Photos"):
                st.session_state.captured_faces = []
                st.rerun()
        
        with col3:
            if st.button("❌ Cancel"):
                st.session_state.captured_faces = []
                st.session_state.face_capture_running = False
                st.rerun()

def run_database_management(face_system):
    """Database management interface"""
    st.subheader("👥 Face Database Management")
    
    people = face_system.get_registered_people()
    
    if not people:
        st.info("📭 No people registered yet. Use 'Register New Person' to add faces.")
        return
    
    st.markdown(f"### 📊 Registered People ({len(people)})")
    
    for person in people:
        with st.expander(f"👤 {person['name']} - {person['encodings_count']} encodings"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**ID:** {person['id'][:8]}...")
                st.write(f"**Registered:** {person['registered_date'][:10]}")
            
            with col2:
                st.write(f"**Encodings:** {person['encodings_count']}")
                
            with col3:
                if st.button(f"🗑️ Delete {person['name']}", key=f"delete_{person['id']}"):
                    if st.button(f"⚠️ Confirm Delete", key=f"confirm_{person['id']}"):
                        success, message = face_system.delete_person(person['id'])
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
    
    # Database statistics
    st.markdown("### 📈 Database Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("👥 Total People", len(people))
    with col2:
        total_encodings = sum(p['encodings_count'] for p in people)
        st.metric("🧠 Total Encodings", total_encodings)
    with col3:
        avg_encodings = total_encodings / len(people) if people else 0
        st.metric("📊 Avg per Person", f"{avg_encodings:.1f}")
