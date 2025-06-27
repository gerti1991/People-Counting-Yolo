#!/usr/bin/env python3
"""
People Counting System - Quick Health Check
A lightweight test to verify system readiness before launching the application.
"""

import sys
import subprocess
from datetime import datetime

def print_status(message, status="info"):
    """Print formatted status message"""
    symbols = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}
    print(f"{symbols.get(status, 'ℹ️')} {message}")

def check_python_version():
    """Check Python version compatibility"""
    print_status("Checking Python version...")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print_status(f"Python {version.major}.{version.minor}.{version.micro} - Compatible", "success")
        return True
    else:
        print_status(f"Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+", "error")
        return False

def check_core_dependencies():
    """Check essential dependencies"""
    print_status("Checking core dependencies...")
    
    core_packages = [
        ("streamlit", "Web interface"),
        ("cv2", "Computer vision"),
        ("numpy", "Numerical computing"),
        ("torch", "Deep learning"),
        ("ultralytics", "YOLO models")
    ]
    
    all_good = True
    for package, description in core_packages:
        try:
            __import__(package)
            print_status(f"{package} - {description}", "success")
        except ImportError:
            print_status(f"{package} - {description} (MISSING)", "error")
            all_good = False
    
    return all_good

def check_optional_dependencies():
    """Check optional face recognition dependencies"""
    print_status("Checking optional dependencies...")
    
    optional_packages = [
        ("face_recognition", "Face recognition"),
        ("dlib", "Face detection backend"),
        ("sklearn", "Machine learning")
    ]
    
    for package, description in optional_packages:
        try:
            __import__(package)
            print_status(f"{package} - {description}", "success")
        except ImportError:
            print_status(f"{package} - {description} (Optional - not installed)", "warning")

def check_camera_access():
    """Quick camera availability check"""
    print_status("Checking camera access...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print_status("Camera 0 - Available", "success")
            cap.release()
            return True
        else:
            print_status("Camera 0 - Not accessible", "warning")
            return False
    except Exception as e:
        print_status(f"Camera check failed: {e}", "warning")
        return False

def check_model_file():
    """Check YOLO model availability"""
    print_status("Checking YOLO model...")
    
    model_path = "yolov9c.pt"
    if os.path.exists(model_path):
        print_status("YOLOv9 model file found", "success")
        return True
    else:
        print_status("YOLOv9 model will be downloaded on first use", "info")
        return True

def main():
    """Run quick system health check"""
    print("=" * 50)
    print("🚀 PEOPLE COUNTING SYSTEM - QUICK TEST")
    print("=" * 50)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("Core Dependencies", check_core_dependencies),
        ("Optional Dependencies", check_optional_dependencies),
        ("Camera Access", check_camera_access),
        ("Model File", check_model_file)
    ]
    
    results = []
    for check_name, check_func in checks:
        print(f"\n🔍 {check_name}:")
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print_status(f"Check failed: {e}", "error")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    
    core_passed = all(result for name, result in results[:2])  # Python + Core deps
    optional_issues = not all(result for name, result in results[2:])
    
    if core_passed:
        print_status("System ready to launch! 🚀", "success")
        if optional_issues:
            print_status("Some optional features may not be available", "warning")
    else:
        print_status("System has critical issues - check dependencies", "error")
        print("\n💡 Try: pip install -r requirements.txt")
    
    print("\n🎯 To start the application:")
    print("   • Windows: Double-click start_app.bat")
    print("   • Command line: streamlit run app.py")

if __name__ == "__main__":
    import os
    main()
