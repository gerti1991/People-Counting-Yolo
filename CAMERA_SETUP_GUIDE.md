# 📷 Universal Camera Configuration Guide

## 🌟 **NEW: Enhanced Camera Support**

Your People Counting System now supports **all types of cameras** through environment variable configuration!

## 📋 **Supported Camera Types**

### 📷 **USB/Built-in Cameras**
- Webcams, laptop cameras
- USB cameras (0, 1, 2, etc.)

### 🌐 **IP Cameras**
- HTTP/MJPEG streams
- Network cameras with web interface

### 📡 **RTSP Cameras**
- Security cameras
- Professional surveillance systems
- ONVIF compatible cameras

### 📱 **Phone Cameras**
- IP Webcam app (Android)
- DroidCam
- Any phone camera app with streaming

### ⚙️ **Custom Sources**
- Any OpenCV-compatible source
- Custom video streams

## 🛠️ **Setup Instructions**

### **1. Edit the `.env` file**

Open the `.env` file in your project root and configure your cameras:

```bash
# USB Cameras (usually 0, 1, 2)
USB_CAMERA_0=0
USB_CAMERA_1=1

# IP Cameras (HTTP streams)
IP_CAMERA_1=http://192.168.1.100:8080/video
IP_CAMERA_2=http://admin:password@192.168.1.101:8080/video

# RTSP Cameras (Security cameras)
RTSP_CAMERA_1=rtsp://admin:password@192.168.1.200:554/stream1
RTSP_CAMERA_2=rtsp://192.168.1.201:554/live

# Phone Cameras (IP Webcam app)
PHONE_CAMERA_1=http://192.168.1.50:8080/video

# Custom cameras
CUSTOM_CAMERA_1=your_custom_source_here
```

### **2. Common Camera URLs**

#### **IP Webcam (Android)**
```
http://[PHONE_IP]:8080/video
```

#### **Generic IP Cameras**
```
http://[CAMERA_IP]/mjpg/video.mjpg
http://[CAMERA_IP]:8080/video
http://admin:password@[CAMERA_IP]/video
```

#### **RTSP Security Cameras**
```
rtsp://admin:password@[CAMERA_IP]:554/stream1
rtsp://[CAMERA_IP]:554/live
rtsp://username:password@[CAMERA_IP]:554/cam/realmonitor?channel=1&subtype=0
```

### **3. Camera Settings**

Configure default settings in `.env`:

```bash
# Resolution and Performance
CAMERA_WIDTH=640
CAMERA_HEIGHT=480
CAMERA_FPS=30

# Detection Settings
DEFAULT_CONFIDENCE=0.5
DEFAULT_TRACKING_DISTANCE=80

# Connection Settings
CAMERA_TIMEOUT=10
FRAME_BUFFER_SIZE=1
```

## 🔧 **Testing Your Cameras**

### **In the Application:**
1. Start the app: `streamlit run app.py`
2. Go to "🎥 Live Camera Counting"
3. Select your camera from the dropdown
4. Click "🔍 Test Camera" to verify connection
5. Use "⚙️ Manual Camera Input" for testing new sources

### **Find Your Camera IP:**
```bash
# Windows
ipconfig

# Phone IP Webcam app will show the URL
# Usually: http://192.168.1.XX:8080/video
```

## 📱 **Phone Camera Setup**

### **Android - IP Webcam App:**
1. Install "IP Webcam" from Google Play
2. Start the app and note the IP address
3. Add to `.env`: `PHONE_CAMERA_1=http://192.168.1.50:8080/video`

### **iPhone - Similar Apps:**
1. Search for "IP Camera" apps
2. Follow app instructions for URL
3. Add URL to `.env` file

## 🔒 **Security Camera Setup**

### **Common Brands:**
```bash
# Hikvision
rtsp://admin:password@192.168.1.200:554/Streaming/Channels/101

# Dahua
rtsp://admin:password@192.168.1.201:554/cam/realmonitor?channel=1&subtype=0

# Generic ONVIF
rtsp://admin:password@192.168.1.202:554/onvif1
```

## 🆘 **Troubleshooting**

### **Camera Not Detected:**
- Check IP address and port
- Verify camera is powered on
- Test URL in browser (for IP cameras)
- Check firewall settings

### **Connection Timeout:**
- Increase timeout in settings
- Check network connectivity
- Verify camera credentials

### **Poor Performance:**
- Reduce resolution in `.env`
- Lower FPS setting
- Check network bandwidth (for IP cameras)

## 🎯 **Features Available:**

- ✅ **Universal camera support** (USB, IP, RTSP, Phone)
- ✅ **Auto-detection** from `.env` configuration
- ✅ **Connection testing** before use
- ✅ **Manual camera input** for testing
- ✅ **Enhanced error handling** with reconnection
- ✅ **Unique person tracking** on all camera types
- ✅ **Screenshot naming** with camera source info

Your system now supports virtually any camera source! 🎉
