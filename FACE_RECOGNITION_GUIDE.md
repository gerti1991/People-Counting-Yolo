# Face Recognition System Setup Guide

## Overview

The Face Recognition System allows you to:
- 🔍 **Recognize known people** in real-time using your camera
- ➕ **Register new people** with multiple face angles for better accuracy
- 🎯 **Combine people counting with face recognition** for advanced tracking
- 📊 **Track who enters your space** with detailed analytics

## Features

### 1. Live Face Recognition
- Real-time face detection and recognition
- Works with all supported camera types (USB, IP, RTSP, Phone)
- Adjustable recognition tolerance
- Confidence scores for each recognition

### 2. Face Registration System
- **Multi-angle capture**: Front, left turn, right turn, up, down
- **Automatic face detection** with visual feedback
- **Data augmentation** for better recognition accuracy
- **Face database management** with easy deletion

### 3. Integrated Tracking
- **YOLO person detection** + **Face recognition**
- **Unique person counting** with face identification
- **Advanced analytics**: Who was recognized vs unknown
- **Session summaries** with detailed statistics

## Installation

### 1. Install Required Dependencies

```bash
pip install face_recognition dlib scikit-learn joblib
```

### 2. System Requirements

#### Windows:
- Visual Studio Build Tools or Visual Studio with C++ support
- CMake (for dlib compilation)

#### macOS:
```bash
brew install cmake
```

#### Linux (Ubuntu/Debian):
```bash
sudo apt-get install build-essential cmake
sudo apt-get install libopenblas-dev liblapack-dev
```

### 3. Alternative Installation (if face_recognition fails)
If you encounter issues with `face_recognition`, try:
```bash
pip install --upgrade cmake
pip install dlib
pip install face_recognition
```

## Setup Instructions

### 1. Configure Your Camera
1. Edit your `.env` file to add camera sources
2. Test camera connection in the app
3. Ensure good lighting for face detection

### 2. Register People
1. Go to **Face Recognition System** → **Register New Person**
2. Enter the person's full name
3. Follow the on-screen instructions to capture 5 photos:
   - Look straight at the camera
   - Turn head slightly left
   - Turn head slightly right
   - Look slightly up
   - Look slightly down
4. Click **Register Person** to save

### 3. Start Recognition
1. Choose **Live Face Recognition** or **Integrated Tracking**
2. Select your camera source
3. Adjust recognition tolerance (lower = stricter)
4. Click **Start Recognition**

## How It Works

### Face Registration Process
1. **Capture Multiple Angles**: 5 different face positions
2. **Face Detection**: Extract face regions from each photo
3. **Encoding Generation**: Create 128-dimension face encodings
4. **Data Augmentation**: Generate variations (brightness, contrast, blur, flip)
5. **Model Training**: Train SVM classifier for better accuracy
6. **Database Storage**: Save encodings and metadata

### Recognition Process
1. **YOLO Detection**: Detect people in the frame
2. **Face Extraction**: Extract face region from person bounding box
3. **Face Encoding**: Generate encoding for detected face
4. **Comparison**: Compare with known face database
5. **Classification**: Use SVM model for final identification
6. **Tracking Integration**: Link face identity with person tracking

### Data Storage
- **Face encodings**: `data/face_encodings.pkl`
- **Face database**: `data/face_database.json`
- **Face images**: `data/registered_faces/[person_id]/`
- **ML models**: `models/face_models/`

## Usage Tips

### For Best Results
1. **Good Lighting**: Ensure well-lit environment
2. **Clear Face View**: Face should be clearly visible (not obscured)
3. **Multiple Angles**: Register with different head positions
4. **Quality Photos**: Avoid blurry or low-resolution images
5. **Update Regularly**: Re-register if appearance changes significantly

### Recognition Settings
- **Tolerance 0.3-0.4**: Very strict (fewer false positives)
- **Tolerance 0.5-0.6**: Balanced (recommended)
- **Tolerance 0.7-0.8**: Lenient (more matches, but may have false positives)

### Camera Positioning
- **Height**: Camera at eye level or slightly above
- **Distance**: 2-8 feet from camera works best
- **Angle**: Face camera directly for best recognition
- **Stability**: Stable camera mount reduces false negatives

## Troubleshooting

### Common Issues

#### 1. Face Recognition Library Installation Error
```bash
# Windows: Install Visual Studio Build Tools
# Then try:
pip install cmake
pip install dlib
pip install face_recognition
```

#### 2. No Faces Detected
- Check lighting conditions
- Ensure face is clearly visible
- Try moving closer to camera
- Verify camera is working

#### 3. Poor Recognition Accuracy
- Register more photos of the person
- Improve lighting during registration
- Lower the recognition tolerance
- Ensure face is not obscured

#### 4. Camera Not Working
- Check `.env` configuration
- Test camera in basic mode first
- Verify camera permissions
- Try different camera source

#### 5. Slow Performance
- Reduce camera resolution in `.env`
- Lower FPS setting
- Use USB camera instead of IP camera
- Close other applications using camera

### Error Messages

#### "face_recognition library not available"
- Install: `pip install face_recognition`
- Check system dependencies are installed

#### "Cannot access camera"
- Verify camera is not in use by another app
- Check camera permissions
- Try different camera index/URL

#### "No face detected during registration"
- Improve lighting
- Move closer to camera
- Ensure face is fully visible
- Try different camera angle

## File Structure

```
People-Counting-Yolo-MyVersion/
├── face_recognition_system.py    # Core face recognition module
├── integrated_tracking.py        # YOLO + Face recognition integration
├── data/
│   ├── face_encodings.pkl        # Trained face encodings
│   ├── face_database.json        # Person database
│   └── registered_faces/         # Stored face images
│       └── [person_id]/          # Individual person folders
└── models/
    └── face_models/              # Trained ML models
        ├── face_classifier.pkl   # SVM classifier
        └── label_encoder.pkl     # Label encoder
```

## Advanced Features

### Data Augmentation
The system automatically creates variations of each registered photo:
- Brightness adjustments (darker/brighter)
- Contrast modifications
- Slight blur effects
- Horizontal flipping

### Machine Learning
- **SVM Classifier**: Trained on face encodings for better accuracy
- **Label Encoding**: Handles multiple people efficiently
- **Confidence Scoring**: Provides reliability metrics
- **Incremental Learning**: Automatically retrains when new people added

### Analytics & Reporting
- Real-time recognition statistics
- Session summaries with recognized vs unknown people
- Recognition confidence tracking
- Face database management tools

## Privacy & Security

### Data Protection
- All face data stored locally
- No cloud uploading
- Encrypted face encodings
- Easy data deletion

### GDPR Compliance
- Clear consent required for face registration
- Easy data deletion (right to be forgotten)
- Local processing only
- Transparent data usage

## Support

For issues or questions:
1. Check this guide first
2. Verify all dependencies are installed
3. Test with basic camera mode
4. Check system requirements
5. Review error messages carefully

The face recognition system is designed to be robust and user-friendly while maintaining privacy and security standards.
