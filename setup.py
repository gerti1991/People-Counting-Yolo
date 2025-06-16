#!/usr/bin/env python3
"""
🚀 People Counting System - Automated Setup Script
==================================================

This script automates the complete setup process:
- Verifies Python version compatibility
- Installs all required dependencies
- Downloads YOLO model automatically
- Tests camera connectivity
- Validates system functionality
- Provides clear launch instructions

Usage: python setup.py
"""

import subprocess
import sys
import os
import importlib.util
from pathlib import Path

def print_banner():
    """Print setup banner"""
    print("🚀 People Counting System - Setup Script")
    print("=" * 50)
    print()

def run_command(command, description, capture_output=True):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        if capture_output:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        else:
            result = subprocess.run(command, shell=True, check=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        if capture_output and e.stdout:
            print(f"   Output: {e.stdout.strip()}")
        if capture_output and e.stderr:
            print(f"   Error: {e.stderr.strip()}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        print("   Please install Python 3.8 or higher")
        return False

def verify_imports():
    """Verify that key packages can be imported"""
    print("\n🔗 Verifying package imports...")
    packages = [
        ("streamlit", "Web framework"),
        ("ultralytics", "YOLO model"),
        ("cv2", "OpenCV"),
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow")
    ]
    
    all_good = True
    for package, description in packages:
        try:
            importlib.import_module(package)
            print(f"   ✅ {package:12} - {description}")
        except ImportError:
            print(f"   ❌ {package:12} - Import failed")
            all_good = False
    
    return all_good

def main():
    """Main setup process"""
    print_banner()
    
    # Check Python version first
    if not check_python_version():
        input("\nPress Enter to exit...")
        return False
    
    # Install dependencies
    if not run_command("pip install --upgrade pip", "Upgrading pip"):
        print("   ⚠️ Pip upgrade failed, continuing anyway...")
    
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("   ❌ Failed to install dependencies!")
        input("\nPress Enter to exit...")
        return False
    
    # Verify imports
    if not verify_imports():
        print("   ⚠️ Some packages failed to import, but continuing...")
    
    # Test installation
    print("\n🧪 Running comprehensive system tests...")
    if run_command("python quick_test.py", "Testing system functionality"):
        print("\n🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("\n🌟 LAUNCH OPTIONS:")
        print("\n📊 Full Application (Recommended):")
        print("   streamlit run app.py")
        print("   📍 Opens: http://localhost:8501")
        print("   Features: Video processing + Live camera")
        
        print("\n🎥 Live Camera Only:")
        print("   streamlit run live_test.py")
        print("   📍 Opens: http://localhost:8503")
        print("   Features: Direct camera interface")
        
        print("\n🔧 Command Line:")
        print("   python live_camera_counter.py")
        print("   Features: Terminal-based counting")
        
        print("\n📋 Useful Commands:")
        print("   python test_camera.py        # Test camera")
        print("   python quick_test.py         # System test")
        
        print("\n📖 For detailed instructions, see README.md")
        print("\n🎯 Ready to count people! 🚀")
        
        input("\nPress Enter to exit...")
        return True
    else:
        print("\n⚠️ Setup completed but system tests failed.")
        print("Check the error messages above and see README.md for troubleshooting.")
        input("\nPress Enter to exit...")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
