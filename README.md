# 🎥 People Counting System using YOLO

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLO](https://img.shields.io/badge/YOLO-v9-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45+-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.11+-orange.svg)

## 📋 Project Overview

Advanced real-time people counting system powered by **YOLOv9** and **Streamlit**. This system provides accurate people detection and counting with support for multiple camera types and video file processing.

### 🌟 Key Features

- **🎥 Real-time Live Camera Counting** - Instant people detection using webcam/USB cameras
- **📹 Video File Processing** - Upload and analyze pre-recorded videos
- **🌐 Universal Camera Support** - USB, IP cameras, RTSP streams
- **📊 Live Analytics** - Real-time FPS, people count, and performance metrics
- **📸 Screenshot Capture** - Save detection results with timestamps
- **🎯 High Accuracy** - YOLOv9 model with adjustable confidence thresholds
- **🖥️ Web Interface** - User-friendly Streamlit dashboard

### 🔧 Technology Stack

- **AI Model:** YOLOv9c (49.4MB) - State-of-the-art object detection
- **Computer Vision:** OpenCV 4.11+ - Image processing and camera handling
- **Web Framework:** Streamlit 1.45+ - Interactive web interface
- **Deep Learning:** PyTorch 2.7+ - Model inference engine
- **Language:** Python 3.8+ - Core development language

---

## 🚀 Quick Start Guide

### Step 1: Prerequisites Check

**System Requirements:**
- Python 3.8 or higher
- Webcam or USB camera
- 4GB+ RAM recommended
- Internet connection (for initial model download)

**Check Python version:**
```bash
python --version
```

### Step 2: Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd People-Counting-Yolo-MyVersion
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: First run will download YOLOv9c model (~49MB)*

### Step 3: Verify Installation

**Run system test:**
```bash
python quick_test.py
```

**Expected output:**
```
🚀 People Counting System - Quick Test
==================================================
🔵 Testing OpenCV...                    ✅ PASS
🎥 Testing Camera Access...             ✅ PASS  
🤖 Testing YOLO Model...                ✅ PASS
🔄 Testing YOLO + Camera Integration... ✅ PASS
==================================================
🎉 ALL TESTS PASSED! Your system is ready.
```

### Step 4: Launch Application

**Option A: Full Web Interface (Recommended)**
```bash
streamlit run app.py
```
- Opens: http://localhost:8501
- Features: Video processing + Live camera modes

**Option B: Live Camera Only**
```bash
streamlit run live_test.py
```
- Opens: http://localhost:8503  
- Features: Direct live camera interface

### Step 5: Start Counting!

1. **Open your browser** to the provided URL
2. **Select mode:**
   - **📹 Video File Processing:** Upload MP4/AVI/MOV files
   - **🎥 Live Camera Counting:** Real-time camera detection
3. **Configure settings:**
   - Confidence threshold (0.1-1.0)
   - Camera source selection
   - Detection parameters
4. **Click "Start Camera"** and watch live people detection!

---

## 📖 Detailed Usage Instructions

### 🎥 Live Camera Mode

1. **Launch the application:**
   ```bash
   streamlit run app.py
   ```

2. **Access web interface:**
   - Open browser to: http://localhost:8501

3. **Navigate to Live Camera:**
   - Click sidebar menu (► arrow if collapsed)
   - Select **"🎥 Live Camera Counting"**

4. **Configure camera:**
   - **Camera Source:** Select from detected cameras
   - **Confidence:** Adjust detection sensitivity (0.5 recommended)
   - **Messages:** Enable/disable notifications

5. **Start detection:**
   - Click **"🟢 Start Camera"**
   - Allow camera access if prompted
   - Watch real-time people detection!

### 📹 Video Processing Mode

1. **Select Video Processing mode** in sidebar

2. **Upload video file:**
   - Drag & drop or browse files
   - Supported: MP4, AVI, MOV, MKV (max 200MB)

3. **Configure processing:**
   - **Batch Size:** 1-8 (higher = faster, more memory)
   - **Confidence:** Detection threshold

4. **Process video:**
   - Click **"Process Video"**
   - Wait for completion
   - Download processed result

### 🛠️ Advanced Configuration

#### Camera Settings
- **USB Cameras:** Auto-detected (Camera 0, 1, 2...)
- **IP Cameras:** Enter URL (e.g., `http://192.168.1.100:8080/video`)
- **RTSP Streams:** Enter RTSP URL (e.g., `rtsp://admin:pass@192.168.1.100:554/stream`)

#### Detection Parameters
- **Confidence Threshold:** 0.1 (detect everything) to 1.0 (only certain detections)
- **Target Class:** Person (COCO class 0)
- **Model:** YOLOv9c (automatically downloaded)

---

## 🔧 Troubleshooting

### Common Issues & Solutions

#### ❌ "No camera detected"
```bash
# Test camera connectivity
python test_camera.py
```
**Solutions:**
- Check camera connections
- Close other apps using camera
- Try different camera indices (0, 1, 2...)

#### ❌ "YOLO model download fails"
**Solutions:**
- Check internet connection
- Manual download: Model auto-downloads on first use
- Verify disk space (need ~50MB free)

#### ❌ "Streamlit won't start"
```bash
# Use custom port
streamlit run app.py --server.port 8502
```
**Solutions:**
- Check if port 8501 is in use
- Try different port numbers
- Run with `--server.headless true` for server mode

#### ❌ "App shows blank page"
**Solutions:**
- Clear browser cache
- Try incognito/private mode
- Check console for JavaScript errors
- Restart Streamlit app

#### ❌ "Low detection accuracy"
**Solutions:**
- Increase confidence threshold
- Ensure good lighting
- Check camera focus
- Verify camera resolution

### Performance Optimization

#### For Better Speed:
- Lower batch size for video processing
- Reduce camera resolution
- Close other applications
- Use GPU if available

#### For Better Accuracy:
- Good lighting conditions
- Camera at eye level
- Stable camera position
- Adjust confidence threshold

---

## 📁 Project Structure

```
People-Counting-Yolo-MyVersion/
├── 📄 README.md                    # This file - Complete documentation
├── 🚀 app.py                       # Main Streamlit application
├── 🎥 live_camera_app.py           # Live camera interface
├── 🎯 live_test.py                 # Simple live camera test
├── 🧪 quick_test.py                # System validation script
├── 📷 test_camera.py               # Camera connectivity test
├── 📋 requirements.txt             # Python dependencies
├── 📊 TEST_REPORT.md               # Detailed test results
├── 🔧 .streamlit/config.toml       # Streamlit configuration
├── 🤖 models/                      # Model files
│   ├── model_streamlit.py          # YOLO integration
│   ├── modelv1.py                  # Version 1 model
│   └── modelv2.py                  # Version 2 model
├── 📁 data/                        # Input data directory
├── 📁 results/                     # Output results directory
└── 🔍 yolov9c.pt                   # YOLO model file (auto-downloaded)
```

---

## 🎯 Testing & Validation

### Quick System Test
```bash
python quick_test.py
```

### Camera Connectivity Test  
```bash
python test_camera.py
```

### Manual Testing Checklist

- [ ] **Camera Detection:** Camera 0 detected and working
- [ ] **YOLO Model:** Model loads successfully
- [ ] **Live Detection:** People detected in real-time
- [ ] **Web Interface:** Streamlit app accessible
- [ ] **Video Processing:** Upload and process test video
- [ ] **Screenshot:** Capture functionality works

---

## 🔧 Technical Specifications

### System Requirements
- **Minimum:** Python 3.8, 4GB RAM, 1GB storage
- **Recommended:** Python 3.9+, 8GB RAM, 2GB storage
- **Camera:** USB/Built-in webcam or IP camera
- **Network:** Required for initial model download

### Model Information  
- **Architecture:** YOLOv9c
- **File Size:** 49.4MB
- **Input Resolution:** 640x640 (auto-resized)
- **Detection Classes:** Person (COCO class 0)
- **Inference Speed:** Real-time capable

### Performance Metrics
- **Detection Accuracy:** High (YOLOv9c)
- **Processing Speed:** Real-time (depends on hardware)
- **Memory Usage:** ~2GB during operation
- **Supported Formats:** MP4, AVI, MOV, MKV

---

## 🆘 Support & Resources

### Getting Help
1. **Check the troubleshooting section** above
2. **Run diagnostic tests:** `python quick_test.py`
3. **Review test report:** `TEST_REPORT.md`
4. **Check system logs** in terminal output

### Useful Commands
```bash
# Full system test
python quick_test.py

# Camera test only
python test_camera.py

# Start main app
streamlit run app.py

# Start simple live test
streamlit run live_test.py

# Check dependencies
pip list | grep -E "(streamlit|ultralytics|opencv|torch)"
```

### External Resources
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [OpenCV Documentation](https://docs.opencv.org/)

---

## 🎉 Success Indicators

Your system is working correctly when you see:

✅ **All tests pass** in `quick_test.py`  
✅ **Green bounding boxes** around detected people  
✅ **Live count updates** in real-time  
✅ **Smooth video stream** without lag  
✅ **Accurate people detection** in various lighting  

---

---

## ⚡ Quick Execution Guide

### 🎯 **FOR FIRST-TIME USERS (Easy Setup):**

1. **Download/Clone this project**
2. **Run the automated setup:**
   ```bash
   python setup.py
   ```
   This will:
   - ✅ Check Python compatibility
   - ✅ Install all dependencies
   - ✅ Download YOLO model
   - ✅ Test camera connectivity
   - ✅ Validate system functionality

3. **Launch the application:**
   ```bash
   streamlit run app.py
   ```

4. **Open browser:** http://localhost:8501

5. **Start counting:**
   - Select **"🎥 Live Camera Counting"** from sidebar
   - Click **"🟢 Start Camera"**
   - Watch live people detection!

---

### 🚀 **ALTERNATIVE LAUNCH OPTIONS:**

#### **📊 Full Web Interface (All Features)**
```bash
streamlit run app.py
```
- **URL:** http://localhost:8501
- **Features:** Video processing + Live camera modes
- **Best for:** Complete functionality

#### **🎥 Live Camera Only (Simple)**
```bash
streamlit run live_test.py
```
- **URL:** http://localhost:8503  
- **Features:** Direct live camera interface
- **Best for:** Quick camera testing

#### **🔧 Command Line Interface**
```bash
python live_camera_counter.py
```
- **Terminal-based:** No web browser needed
- **Features:** Basic camera counting
- **Best for:** Headless servers

---

### 🔧 **MANUAL SETUP (Advanced Users):**

If you prefer manual setup:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test system
python quick_test.py

# 3. Test camera
python test_camera.py

# 4. Launch application
streamlit run app.py
```

---

**🏆 Project Status: PRODUCTION READY**

*Last Updated: June 16, 2025*

### Live Camera Counting (Standalone)

For live camera counting only:

```
python live_camera_counter.py
```

Options:
```
python live_camera_counter.py --camera 0                    # USB camera 0
python live_camera_counter.py --camera 1                    # USB camera 1
python live_camera_counter.py --camera "http://192.168.1.100:8080/video"  # IP camera
python live_camera_counter.py --confidence 0.7              # Higher confidence
```

### Camera Testing

Test your camera connectivity first:

```
python test_camera.py
```

## Project Structure

The `models` folder contains three main scripts:

1. `modelv1.py`: Counts people only within the ROI area
2. `modelv2.py`: Counts people using ROI with cumulative tracking
3. `model_streamlit.py`: Deploys the model using Streamlit with batch processing for faster video processing

## Resources

- [YOLO Models Comparison](https://docs.ultralytics.com/de/models/yolov9/#supported-tasks-and-modes)
- [How to Use YOLOv9](https://medium.com/@Mert.A/how-to-use-yolov9-for-object-detection-93598ad88d7d)
- [Previous Work on People Counting](https://github.com/ChinmayBH/Exploring-Vision)
- [People Counter using YOLOv8](https://github.com/noorkhokhar99/People-Counter-using-YOLOv8-and-Object-Tracking-People-Counting-Entering-Leaving-)
- [YOLOv9: Latest Advancement in YOLO Series](https://medium.com/@xis.ai/yolov9-the-latest-version-in-the-yolo-series-3dd609571613)
- [YOLO: Algorithm for Object Detection](https://www.v7labs.com/blog/yolo-object-detection#)

## Future Work

- Starting with improving the performance of the tracking algorithm

## Camera Support

### Supported Camera Types

- **USB Cameras**: Built-in webcams, external USB cameras (camera ID: 0, 1, 2...)
- **IP Cameras**: Network cameras with HTTP streams (e.g., `http://192.168.1.100:8080/video`)
- **RTSP Streams**: Professional security cameras (e.g., `rtsp://admin:password@192.168.1.100:554/stream`)
- **Video Files**: For testing and development

### Camera Setup Examples

**USB Camera:**
```python
camera_source = 0  # First USB camera
```

**IP Camera:**
```python
camera_source = "http://192.168.1.100:8080/video"
```

**RTSP Stream:**
```python
camera_source = "rtsp://username:password@ip:port/stream"
```

### Features

- ✅ **Real-time detection**: Live people counting
- ✅ **Universal compatibility**: Works with any OpenCV-supported camera
- ✅ **Message notifications**: Configurable alert system
- ✅ **Screenshot capture**: Save moments with people count
- ✅ **FPS monitoring**: Performance tracking
- ✅ **Web interface**: Easy-to-use Streamlit GUI
- ✅ **Command-line interface**: Direct camera access
