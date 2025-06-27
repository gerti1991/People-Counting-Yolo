# 👥 People Counting System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLO](https://img.shields.io/badge/YOLO-v9-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Version](https://img.shields.io/badge/Version-2.0.0-brightgreen.svg)

> **🚀 Production-ready AI-powered people counting system using YOLOv9 and Streamlit**  
> *Clean • Documented • Reusable • Future-proof*

---

## ✨ Features

### 🎯 Core Functionality
- **🎥 Real-time Camera Counting** - Live people detection with unique person tracking
- **📹 Video File Processing** - Upload and analyze videos with smart tracking
- **🎯 Unique Person Tracking** - Advanced algorithm prevents double counting
- **📊 Comprehensive Analytics** - Detailed statistics and visual overlays
- **🌐 Universal Camera Support** - USB, IP, RTSP cameras via .env config

### 👤 Face Recognition *(Optional)*
- **👤 Face Recognition** - Identify specific people in real-time
- **➕ Person Registration** - Multi-angle face capture system
- **🎯 Integrated Tracking** - Combine counting with face identification
- **🔒 Privacy-First** - All processing done locally
- **🧠 Smart Fallback** - Works without face recognition dependencies

### 🛠️ Technical Features
- **🌐 Web Interface** - User-friendly Streamlit dashboard
- **📱 Cross-platform** - Windows, macOS, Linux support
- **⚡ Optimized Performance** - Efficient resource management
- **🔧 Easy Configuration** - Environment-based settings
- **🧪 Testing Suite** - Comprehensive validation tools

---

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

**Windows (Recommended):**
```cmd
# Double-click or run:
start_app.bat
```

**All Platforms:**
```bash
streamlit run app.py
```

### 3. Use the System
1. **Open browser** to `http://localhost:8501`
2. **Choose mode:**
   - 📹 **Video Processing** - Upload videos for analysis
   - 🎥 **Live Camera** - Real-time counting
   - 👤 **Face Recognition** - Register and recognize people *(optional)*
3. **Start counting!**

---

## 📊 What You Get

### Video Processing
- **Unique People Count** - Total different individuals detected
- **Peak Occupancy** - Maximum people in frame at once
- **Average Density** - People per frame throughout video
- **Processed Video** - Output with detection overlays

### Live Camera
- **Real-time counting** with unique person tracking
- **Adjustable sensitivity** for different environments
- **Live metrics** including FPS and performance stats
- **Snapshot capture** with timestamp and count data
- **Multi-camera support** (USB, IP, RTSP)

---

## 🛠️ System Requirements

- **Python 3.8+**
- **4GB+ RAM** (8GB recommended)
- **Camera** (for live counting)
- **Internet** (for initial model download)

**Platforms:** Windows 10/11, macOS 10.15+, Linux Ubuntu 20.04+

---

## 📁 Project Structure

```
People-Counting-Yolo-MyVersion/
├── 🚀 app.py                      # Main Streamlit application
├── 🎥 camera.py                   # Live camera functionality
├── 👤 face_recognition_system.py  # Face recognition (optional)
├── 🎯 integrated_tracking.py      # Combined YOLO + face recognition
├── ⚡ quick_test.py               # Quick system health check
├── 🧪 system_test.py              # Comprehensive testing
├── 📋 requirements.txt            # Dependencies
├── 🌍 .env.example               # Configuration template
├── 📖 Documentation/              # Guides and help files
├── 🤖 models/                     # AI model logic
├── 🔧 Scripts/                    # Launcher scripts
├── 📁 data/                       # Input and face database
├── 📁 results/                    # Output files
└── 🤖 yolov9c.pt                  # YOLO model (auto-downloaded)
```

---

## 👤 Face Recognition *(Optional)*

Face recognition is completely optional - the core counting works perfectly without it!

### Setup
```bash
# Optional: Install face recognition
pip install face_recognition dlib scikit-learn
```

### Features
- **Multi-angle registration** - 5-point capture for accuracy
- **Real-time identification** - Live face recognition
- **Local processing** - No cloud dependencies
- **Graceful fallback** - Works even if dependencies missing

For detailed setup: [FACE_RECOGNITION_GUIDE.md](FACE_RECOGNITION_GUIDE.md)

---

## 🧪 Testing

### Quick Health Check
```bash
python quick_test.py
```

### Full System Test
```bash
python system_test.py
```

Tests verify: dependencies, camera access, model loading, and functionality.

---

## 🔧 Configuration

### Camera Setup
Create `.env` file from `.env.example`:

```env
# Camera settings
CAMERA_SOURCE=0                    # 0=USB, URL=IP camera
CAMERA_WIDTH=640
CAMERA_HEIGHT=480

# Examples for IP cameras
# CAMERA_SOURCE=http://192.168.1.100:8080/video
# CAMERA_SOURCE=rtsp://user:pass@192.168.1.100:554/stream
```

### Detection Settings
- **Confidence:** 0.1-1.0 (detection sensitivity)
- **Tracking Distance:** Adjustable in UI
- **Accuracy Delay:** 5 seconds (built-in for precision)

---

## 🎯 Use Cases

### Business Applications
- **Retail Analytics** - Customer counting and flow analysis
- **Office Monitoring** - Occupancy tracking
- **Event Management** - Crowd monitoring
- **Security** - People detection and alerts

### Face Recognition Applications *(Optional)*
- **Employee Attendance** - Automatic check-in/out
- **Access Control** - Authorized personnel identification
- **VIP Recognition** - Important visitor identification
- **Visitor Analytics** - Repeat vs new visitor tracking

---

## 🆘 Troubleshooting

### Common Issues

**❌ No cameras detected**
```bash
python quick_test.py  # Check camera status
```

**❌ Model download fails**
- Check internet connection
- Ensure sufficient disk space (~50MB)

**❌ Import errors**
```bash
pip install -r requirements.txt  # Reinstall dependencies
```

**❌ Face recognition issues**
- Face recognition is optional - system works without it
- See [INSTALLATION_HELP.md](INSTALLATION_HELP.md) for detailed setup

### Performance Tips
- **Better Speed:** Lower resolution, close other camera apps
- **Better Accuracy:** Good lighting, stable camera, higher confidence
- **Memory:** Close unnecessary applications

---

## 🤝 Contributing

This project is designed to be:
- **🔄 Reusable** - Modular, configurable architecture
- **📚 Well-documented** - Comprehensive guides and comments
- **🔍 Trackable** - Version control, changelogs, issue tracking
- **🚀 Future-proof** - Modern design, scalable structure

### How to Contribute
1. Read [CONTRIBUTING.md](CONTRIBUTING.md)
2. Fork the repository
3. Create feature branch
4. Make changes with tests
5. Submit pull request

See [CHANGELOG.md](CHANGELOG.md) for version history.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

### Third-Party Components
- **YOLOv9**: GPL-3.0 (Ultralytics)
- **OpenCV**: Apache 2.0
- **Streamlit**: Apache 2.0
- **PyTorch**: BSD-3-Clause

---

## 🙏 Acknowledgments

- **YOLOv9** - Ultralytics team
- **Streamlit** - Amazing web framework
- **OpenCV** - Computer vision foundation
- **Community** - Contributors and users

---

## 📞 Support

- 🐛 **Bug Reports:** [GitHub Issues](https://github.com/your-username/People-Counting-Yolo-MyVersion/issues)
- 💡 **Feature Requests:** [GitHub Issues](https://github.com/your-username/People-Counting-Yolo-MyVersion/issues)
- 📚 **Documentation:** [Project Guides](https://github.com/your-username/People-Counting-Yolo-MyVersion/tree/main)

---

<div align="center">

**Made with ❤️ for the computer vision community**

[⭐ Star](https://github.com/your-username/People-Counting-Yolo-MyVersion) • [🐛 Report Bug](https://github.com/your-username/People-Counting-Yolo-MyVersion/issues) • [💡 Request Feature](https://github.com/your-username/People-Counting-Yolo-MyVersion/issues)

</div>
