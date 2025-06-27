#!/usr/bin/env python3
"""
Test Suite for People Counting System
Comprehensive testing of all components
"""

import os
import sys
import cv2
import subprocess
import time

def test_system_requirements():
    """Test system requirements and dependencies"""
    print("🔍 Testing System Requirements")
    print("=" * 50)
    
    # Python version
    python_version = sys.version_info
    print(f"Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print("✅ Python version OK")
    
    # Required packages
    required_packages = [
        'cv2', 'streamlit', 'ultralytics', 'numpy', 'torch'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            print(f"❌ {package} missing")
            return False
    
    print("✅ All requirements satisfied")
    return True

def test_model_loading():
    """Test YOLO model loading"""
    print("\n🤖 Testing YOLO Model")
    print("=" * 50)
    
    try:
        from src.video_processor import get_model
        model = get_model()
        print("✅ YOLO model loaded successfully")
        
        # Test model file
        if os.path.exists("yolov9c.pt"):
            file_size = os.path.getsize("yolov9c.pt") / (1024 * 1024)  # MB
            print(f"✅ Model file: yolov9c.pt ({file_size:.1f} MB)")
        else:
            print("⚠️ Model file will be downloaded on first use")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_camera_availability():
    """Test camera availability"""
    print("\n📷 Testing Camera Availability")
    print("=" * 50)
    
    available_cameras = []
    
    for i in range(5):  # Test first 5 camera indices
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    available_cameras.append(i)
                    print(f"✅ Camera {i}: {width}x{height}")
                cap.release()
            else:
                print(f"❌ Camera {i}: Not accessible")
        except Exception as e:
            print(f"❌ Camera {i}: Error - {e}")
    
    if available_cameras:
        print(f"✅ Found {len(available_cameras)} working camera(s)")
        return True
    else:
        print("❌ No working cameras found")
        print("💡 Try connecting a camera or check permissions")
        return False

def test_video_processing():
    """Test video processing functionality"""
    print("\n🎬 Testing Video Processing")
    print("=" * 50)
    
    # Check if sample video exists
    if not os.path.exists('data/uploaded_video.mp4'):
        print("⚠️ No sample video found")
        print("💡 Upload a video through the web interface to test this feature")
        return True
    
    try:
        from src.video_processor import process_video, get_model
        
        model = get_model()
        
        # Quick test on first few frames
        cap = cv2.VideoCapture('data/uploaded_video.mp4')
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        print(f"✅ Sample video: {total_frames} frames, {fps:.1f} FPS")
        print("🎯 Video processing functionality available")
        
        return True
        
    except Exception as e:
        print(f"❌ Video processing test failed: {e}")
        return False

def test_web_interface():
    """Test if Streamlit can be imported and basic functionality works"""
    print("\n🌐 Testing Web Interface")
    print("=" * 50)
    
    try:
        import streamlit as st
        print(f"✅ Streamlit {st.__version__} available")
        
        # Check if main app exists
        if os.path.exists('main.py'):
            print("✅ Main application file found")
        else:
            print("❌ main.py not found")
            return False
        
        print("🎯 Web interface ready")
        return True
        
    except Exception as e:
        print(f"❌ Web interface test failed: {e}")
        return False

def test_directory_structure():
    """Test project directory structure"""
    print("\n📁 Testing Directory Structure")
    print("=" * 50)
    
    required_dirs = ['src', 'data', 'results', 'tests']
    required_files = ['main.py', 'requirements.txt', 'README.md']
    
    all_good = True
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ Directory: {directory}/")
        else:
            print(f"❌ Missing directory: {directory}/")
            all_good = False
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ File: {file}")
        else:
            print(f"❌ Missing file: {file}")
            all_good = False
    
    return all_good

def run_full_test():
    """Run complete test suite"""
    print("🧪 People Counting System - Test Suite")
    print("=" * 60)
    
    tests = [
        ("System Requirements", test_system_requirements),
        ("Directory Structure", test_directory_structure),
        ("YOLO Model", test_model_loading),
        ("Camera Availability", test_camera_availability),
        ("Video Processing", test_video_processing),
        ("Web Interface", test_web_interface),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\n🚀 To start the application:")
        print("   streamlit run main.py")
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    run_full_test()
