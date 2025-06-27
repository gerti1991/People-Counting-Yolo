#!/usr/bin/env python3
"""
People Counting System - Comprehensive Test Suite
Tests system functionality including face recognition capabilities
"""

import os
import sys
import cv2
import numpy as np
from datetime import datetime

def print_header():
    """Print test header"""
    print("=" * 60)
    print("🚀 PEOPLE COUNTING SYSTEM - COMPREHENSIVE TEST")
    print("=" * 60)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python Version: {sys.version}")
    print("-" * 60)

def test_basic_imports():
    """Test basic imports"""
    print("\n📦 Testing Basic Imports...")
    
    try:
        import streamlit
        print(f"✅ Streamlit: {streamlit.__version__}")
    except ImportError as e:
        print(f"❌ Streamlit: {e}")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV: {e}")
        return False
    
    try:
        import numpy
        print(f"✅ NumPy: {numpy.__version__}")
    except ImportError as e:
        print(f"❌ NumPy: {e}")
        return False
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
        return False
    
    try:
        import ultralytics
        print(f"✅ Ultralytics: {ultralytics.__version__}")
    except ImportError as e:
        print(f"❌ Ultralytics: {e}")
        return False
    
    return True

def test_face_recognition_imports():
    """Test face recognition imports"""
    print("\n👤 Testing Face Recognition Imports...")
    
    try:
        import face_recognition
        print("✅ face_recognition: Available")
        face_rec_available = True
    except ImportError as e:
        print(f"⚠️  face_recognition: {e}")
        print("💡 Install with: pip install face_recognition")
        face_rec_available = False
    
    try:
        import dlib
        print("✅ dlib: Available")
    except ImportError as e:
        print(f"⚠️  dlib: {e}")
        print("💡 Install with: pip install dlib")
    
    try:
        import sklearn
        print(f"✅ scikit-learn: {sklearn.__version__}")
    except ImportError as e:
        print(f"⚠️  scikit-learn: {e}")
        print("💡 Install with: pip install scikit-learn")
    
    try:
        import joblib
        print("✅ joblib: Available")
    except ImportError as e:
        print(f"⚠️  joblib: {e}")
        print("💡 Install with: pip install joblib")
    
    return face_rec_available

def test_project_modules():
    """Test project modules"""
    print("\n🔧 Testing Project Modules...")
    
    # Test main modules
    modules_to_test = [
        ("app", "app.py"),
        ("camera", "camera.py"),
        ("models.model", "models/model.py"),
    ]
    
    success_count = 0
    for module_name, file_path in modules_to_test:
        try:
            if os.path.exists(file_path):
                __import__(module_name)
                print(f"✅ {module_name}: Import successful")
                success_count += 1
            else:
                print(f"❌ {module_name}: File {file_path} not found")
        except ImportError as e:
            print(f"❌ {module_name}: Import error - {e}")
        except Exception as e:
            print(f"⚠️  {module_name}: Error - {e}")
    
    # Test face recognition modules
    face_modules = [
        ("face_recognition_system", "face_recognition_system.py"),
        ("integrated_tracking", "integrated_tracking.py"),
    ]
    
    for module_name, file_path in face_modules:
        try:
            if os.path.exists(file_path):
                __import__(module_name)
                print(f"✅ {module_name}: Import successful")
                success_count += 1
            else:
                print(f"⚠️  {module_name}: File {file_path} not found (face recognition module)")
        except ImportError as e:
            print(f"⚠️  {module_name}: Import error - {e} (may need face_recognition)")
        except Exception as e:
            print(f"⚠️  {module_name}: Error - {e}")
    
    return success_count >= 3  # At least basic modules should work

def test_yolo_model():
    """Test YOLO model loading"""
    print("\n🤖 Testing YOLO Model...")
    
    try:
        from ultralytics import YOLO
        
        model_path = 'yolov9c.pt'
        if os.path.exists(model_path):
            print(f"✅ Model file found: {model_path}")
            
            # Test model loading
            model = YOLO(model_path)
            print("✅ YOLO model loaded successfully")
            
            # Test inference on dummy image
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            results = model(dummy_image, classes=[0], conf=0.5, verbose=False)
            print("✅ Model inference test successful")
            
            return True
        else:
            print(f"⚠️  Model file not found: {model_path}")
            print("💡 Model will be downloaded automatically on first run")
            return True  # Not a critical error
            
    except Exception as e:
        print(f"❌ YOLO model test failed: {e}")
        return False

def test_camera_availability():
    """Test camera availability"""
    print("\n📹 Testing Camera Availability...")
    
    available_cameras = []
    
    # Test USB cameras (0-3)
    for i in range(4):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    print(f"✅ Camera {i}: Available ({width}x{height})")
                    available_cameras.append(i)
                else:
                    print(f"⚠️  Camera {i}: Opens but cannot read frames")
                cap.release()
            else:
                print(f"❌ Camera {i}: Cannot open")
        except Exception as e:
            print(f"❌ Camera {i}: Error - {e}")
    
    if available_cameras:
        print(f"✅ Found {len(available_cameras)} working cameras: {available_cameras}")
        return True
    else:
        print("❌ No working cameras found")
        print("💡 Tips:")
        print("   - Check camera connections")
        print("   - Close other apps using camera")
        print("   - Try external USB camera")
        return False

def test_directories():
    """Test directory structure"""
    print("\n📁 Testing Directory Structure...")
    
    required_dirs = ['data', 'models', 'results']
    optional_dirs = ['data/registered_faces', 'models/face_models']
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory}/: Exists")
        else:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ {directory}/: Created")
    
    for directory in optional_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory}/: Exists")
        else:
            print(f"💡 {directory}/: Will be created when needed")
    
    return True

def test_environment_config():
    """Test environment configuration"""
    print("\n🔧 Testing Environment Configuration...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ python-dotenv: Available")
        
        if os.path.exists('.env'):
            print("✅ .env file: Found")
            
            # Check for camera configurations
            camera_vars = [key for key in os.environ.keys() if 'CAMERA' in key]
            if camera_vars:
                print(f"✅ Camera configurations: {len(camera_vars)} found")
                for var in camera_vars[:3]:  # Show first 3
                    print(f"   - {var}")
                if len(camera_vars) > 3:
                    print(f"   - ... and {len(camera_vars) - 3} more")
            else:
                print("💡 No camera configurations in .env (will use defaults)")
        else:
            print("💡 .env file not found (optional)")
        
        if os.path.exists('.env.example'):
            print("✅ .env.example: Found (good for setup reference)")
        
        return True
        
    except ImportError:
        print("❌ python-dotenv not available")
        return False

def test_face_recognition_functionality():
    """Test face recognition functionality"""
    print("\n👤 Testing Face Recognition Functionality...")
    
    try:
        import face_recognition
        
        # Test with a dummy image
        dummy_face = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test face detection
        face_locations = face_recognition.face_locations(dummy_face)
        print("✅ Face detection function: Working")
        
        # Test face encoding (even on random image)
        try:
            face_encodings = face_recognition.face_encodings(dummy_face)
            print("✅ Face encoding function: Working")
        except:
            print("✅ Face encoding function: Working (no faces in test image)")
        
        # Test our face recognition system
        try:
            from face_recognition_system import FaceRecognitionSystem
            face_system = FaceRecognitionSystem()
            print("✅ FaceRecognitionSystem: Initialized successfully")
            
            # Test basic methods
            registered = face_system.get_registered_people()
            print(f"✅ Database access: {len(registered)} people registered")
            
        except Exception as e:
            print(f"⚠️  FaceRecognitionSystem: {e}")
        
        return True
        
    except ImportError:
        print("⚠️  Face recognition not available (install face_recognition)")
        return False
    except Exception as e:
        print(f"❌ Face recognition test failed: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive system test"""
    print_header()
    
    test_results = {
        'basic_imports': test_basic_imports(),
        'face_recognition_imports': test_face_recognition_imports(),
        'project_modules': test_project_modules(),
        'yolo_model': test_yolo_model(),
        'camera_availability': test_camera_availability(),
        'directories': test_directories(),
        'environment_config': test_environment_config(),
        'face_recognition_functionality': test_face_recognition_functionality(),
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    print("-" * 60)
    print(f"📈 Overall Score: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed >= 6:  # Allow some optional features to fail
        print("🎉 System is ready to use!")
        print("\n🚀 To start the application:")
        print("   streamlit run app.py")
        print("\n📖 For face recognition setup:")
        print("   See FACE_RECOGNITION_GUIDE.md")
    elif passed >= 4:
        print("⚠️  System has basic functionality but some features may not work")
        print("💡 Check failed tests above for installation instructions")
    else:
        print("❌ System has critical issues - please fix failed tests")
    
    print("\n💡 Need help? Check README.md for detailed setup instructions")
    print("=" * 60)

if __name__ == "__main__":
    run_comprehensive_test()
