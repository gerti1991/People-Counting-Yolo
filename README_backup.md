# 👥 People Counting System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLO](https://img.shields.io/badge/YOLO-v9-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Version](https://img.shields.io/badge/Version-2.0.0-brightgreen.svg)
![Maintenance](https://img.shields.io/badge/Maintained-Yes-brightgreen.svg)

> **🚀 Production-ready AI-powered people counting system using YOLOv9 and Streamlit**  
> *Reusable • Well-documented • Future-proof • Privacy-first*

[![GitHub stars](https://img.shields.io/github/stars/your-username/People-Counting-Yolo-MyVersion)](https://github.com/your-username/People-Counting-Yolo-MyVersion/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/your-username/People-Counting-Yolo-MyVersion)](https://github.com/your-username/People-Counting-Yolo-MyVersion/network)
[![GitHub issues](https://img.shields.io/github/issues/your-username/People-Counting-Yolo-MyVersion)](https://github.com/your-username/People-Counting-Yolo-MyVersion/issues)

---

## 📖 Table of Contents

- [✨ Features](#-features)
- [🚀 Quick Start](#-quick-start)
- [� What You Get](#-what-you-get)
- [�️ System Requirements](#️-system-requirements)
- [📁 Project Structure](#-project-structure)
- [👤 Face Recognition System](#-face-recognition-system)
- [🧪 Testing](#-testing)
- [� Configuration](#-configuration)
- [🎯 Use Cases](#-use-cases)
- [🆘 Troubleshooting](#-troubleshooting)
- [📈 Performance Tips](#-performance-tips)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)

---

## ✨ Features

### 🎯 Core Functionality
- **🎥 Real-time Camera Counting** - Live people detection with **unique person tracking**
- **📹 Video File Processing** - Upload and analyze videos with unique people tracking
- **🎯 Smart Tracking** - Advanced centroid-based tracking with 5-second accuracy delay
- **👤 Unique Person Recognition** - Same person won't be counted multiple times (both modes)
- **📊 Comprehensive Analytics** - Detailed statistics and visual overlays
- **🌐 Universal Camera Support** - USB, IP, RTSP, Phone cameras (via .env config)

### 🆕 Advanced Face Recognition System *(Optional)*
- **👤 Face Recognition** - Identify specific people in real-time
- **➕ Person Registration** - Register new faces with multi-angle capture
- **🎯 Integrated Tracking** - Combine people counting with face identification  
- **📊 Advanced Analytics** - Know exactly who enters your space
- **🔒 Privacy-First** - All data stored locally, no cloud processing
- **🧠 Smart Fallback** - OpenCV fallback if face_recognition unavailable

### 🛠️ Technical Features
- **🌐 Web Interface** - User-friendly Streamlit dashboard
- **📱 Multi-platform** - Works on Windows, macOS, and Linux
- **⚡ Performance Optimized** - Lazy loading and efficient resource management
- **🔧 Easy Configuration** - Environment-based settings via .env files
- **🧪 Comprehensive Testing** - Multiple test suites for validation
- **📚 Extensive Documentation** - Complete guides and API documentation
- **🚀 Easy Deployment** - One-click launchers and batch scripts

> **Note**: Face recognition is an optional enhancement. The core people counting works perfectly without it!

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd People-Counting-Yolo-MyVersion

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Application

**Option A: Quick Launch (Windows)**
- Double-click `run.bat`
- Or run `run.ps1` in PowerShell

**Option B: Manual Launch**
```bash
streamlit run app.py
```

### 3. Use the System

1. **Open your browser** to `http://localhost:8501`
2. **Choose mode:**
   - **📹 Video File Processing:** Upload videos for analysis
   - **🎥 Live Camera:** Real-time counting with webcam
   - **👤 Face Recognition System:** Register and recognize specific people
   - **🎯 Integrated Tracking:** Advanced people counting + face recognition
3. **Start counting!**

## 📊 What You Get

### Video Processing Results
- **🚶‍♂️ Unique People Count** - Total different individuals detected
- **👥 Max in Frame** - Peak occupancy at any moment
- **📈 Average per Frame** - Overall density throughout video
- **🎬 Processed Video** - Output with detection overlays and statistics

### Live Camera Features
- **Real-time unique person detection** - Same person won't be counted multiple times
- **Adjustable tracking distance** - Fine-tune person tracking sensitivity
- **Live metrics** including FPS and people count
- **Reset count functionality** - Start fresh anytime
- **📸 Snapshots** with timestamp and unique count
- **Multi-camera support** (USB/webcam)
- **Person ID tracking** - See individual tracking IDs on screen

## 🛠️ System Requirements

- **Python 3.8+**
- **4GB+ RAM** (recommended)
- **Webcam/Camera** (for live counting)
- **Internet connection** (for initial model download)

## 📁 Project Structure

```
People-Counting-Yolo-MyVersion/
├── 🚀 app.py                      # Main Streamlit application
├── 🎥 camera.py                   # Live camera functionality
├── 🧪 test.py                     # System testing
├── 📋 requirements.txt            # Dependencies
├── � README.md                   # This file
├── 🤖 models/
│   └── model.py                   # YOLO processing logic
├── 🔧 run.bat                     # Windows launcher
├── 🔧 run.ps1                     # PowerShell launcher
├── 📁 data/                       # Input videos
├── 📁 results/                    # Output files
└── 🤖 yolov9c.pt                  # YOLO model (auto-downloaded)
```

## 👤 Face Recognition System

### Setup Face Recognition
1. **Install additional dependencies:**
   ```bash
   pip install face_recognition dlib scikit-learn
   ```

2. **Register people:**
   - Go to **Face Recognition System** → **Register New Person**
   - Enter person's name
   - Capture 5 photos (front, left, right, up, down angles)
   - System automatically creates augmented training data

3. **Start recognition:**
   - Choose **Live Face Recognition** or **Integrated Tracking**
   - Select camera and adjust settings
   - Real-time face detection and identification

### Face Recognition Features
- **🎯 Multi-angle Registration** - 5 capture angles for better accuracy
- **🧠 Data Augmentation** - Automatic image variations for robust training
- **⚡ Real-time Recognition** - Live face identification with confidence scores
- **🔗 Integrated Tracking** - Combine YOLO people detection with face recognition
- **📊 Advanced Analytics** - Track who was recognized vs unknown
- **🗄️ Face Database** - Easy management of registered people
- **🔒 Privacy-First** - All processing done locally, no cloud uploads

For detailed setup instructions, see [FACE_RECOGNITION_GUIDE.md](FACE_RECOGNITION_GUIDE.md)

## 🧪 Testing

Run the system test:
```bash
python test.py
```

This will verify:
- ✅ Python dependencies
- ✅ YOLO model loading  
- ✅ Camera availability
- ✅ System functionality

## 🔧 Configuration

### Detection Settings
- **Confidence Threshold:** 0.1 (detect everything) to 1.0 (only certain detections)
- **Accuracy Delay:** 5 seconds (built-in for better counting precision)

### Camera Settings
- **USB Cameras:** Auto-detected (Camera 0, 1, 2...)
- **Resolution:** Optimized 640x480 for performance
- **FPS:** Up to 30 FPS depending on hardware

## 🎯 Use Cases

### General People Counting
- **🏪 Retail Analytics** - Customer counting and flow analysis
- **🏢 Office Monitoring** - Occupancy tracking and space utilization
- **🎪 Event Management** - Crowd counting and capacity monitoring
- **🔒 Security Applications** - People detection and monitoring
- **📊 Research Projects** - Data collection and analysis

### Face Recognition Applications
- **🏢 Employee Attendance** - Automatic check-in/check-out tracking
- **🔐 Access Control** - Identify authorized personnel
- **👥 VIP Recognition** - Identify important customers or guests
- **📊 Visitor Analytics** - Track repeat visitors vs new visitors
- **🏠 Smart Home Security** - Recognize family members vs strangers
- **🎓 Classroom Attendance** - Automatic student attendance tracking

## 🆘 Troubleshooting

### Common Issues

**❌ "No cameras detected"**
```bash
python test.py  # Check camera availability
```

**❌ "Model download fails"**
- Check internet connection
- Model auto-downloads on first use (~50MB)

**❌ "Streamlit won't start"**
```bash
streamlit run app.py --server.port 8502  # Try different port
```

**❌ "Import errors"**
```bash
pip install -r requirements.txt  # Reinstall dependencies
```

## 📈 Performance Tips

### Better Speed
- Use lower video resolution
- Close other camera applications
- Reduce confidence threshold slightly

### Better Accuracy
- Ensure good lighting conditions
- Stable camera positioning
- Use higher confidence threshold
- Allow 5-second accuracy delay to work

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python test.py`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **YOLOv9** - Ultralytics team for the excellent YOLO implementation
- **Streamlit** - For the amazing web framework
- **OpenCV** - For computer vision capabilities

## 📞 Support

- 🐛 **Bug Reports:** Open an issue on GitHub
- 💡 **Feature Requests:** Open an issue with the "enhancement" label
- 📧 **Questions:** Check existing issues or create a new one

---

<div align="center">

**Made with ❤️ for the computer vision community**

[⭐ Star this repo](https://github.com/your-username/People-Counting-Yolo-MyVersion) • [🐛 Report Bug](https://github.com/your-username/People-Counting-Yolo-MyVersion/issues) • [💡 Request Feature](https://github.com/your-username/People-Counting-Yolo-MyVersion/issues)

</div>
