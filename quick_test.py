#!/usr/bin/env python3
"""
Quick test script to validate the People Counting System components
"""

import cv2
import sys
import numpy as np
from ultralytics import YOLO

def test_opencv():
    """Test OpenCV functionality"""
    print("🔵 Testing OpenCV...")
    try:
        # Test basic OpenCV functionality
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "OpenCV Test", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print("✅ OpenCV is working correctly")
        return True
    except Exception as e:
        print(f"❌ OpenCV error: {e}")
        return False

def test_camera():
    """Test camera access"""
    print("\n🎥 Testing Camera Access...")
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ Camera 0 working - Frame size: {frame.shape}")
                cap.release()
                return True
            else:
                print("❌ Camera detected but cannot read frames")
                cap.release()
                return False
        else:
            print("❌ No camera detected")
            return False
    except Exception as e:
        print(f"❌ Camera error: {e}")
        return False

def test_yolo():
    """Test YOLO model loading"""
    print("\n🤖 Testing YOLO Model...")
    try:
        model = YOLO("yolov9c.pt")
        print("✅ YOLO model loaded successfully")
        
        # Test with a simple image
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        results = model(test_image, classes=[0], conf=0.5, verbose=False)
        print("✅ YOLO inference test completed")
        return True
    except Exception as e:
        print(f"❌ YOLO error: {e}")
        return False

def test_yolo_with_camera():
    """Test YOLO with camera feed"""
    print("\n🔄 Testing YOLO + Camera Integration...")
    try:
        model = YOLO("yolov9c.pt")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Camera not available for YOLO test")
            return False
        
        # Capture a single frame and test detection
        ret, frame = cap.read()
        if ret:
            results = model(frame, classes=[0], conf=0.5, verbose=False)
            people_count = len(results[0].boxes) if len(results) > 0 and results[0].boxes is not None else 0
            print(f"✅ YOLO + Camera integration working - Detected {people_count} people")
            cap.release()
            return True
        else:
            print("❌ Could not capture frame from camera")
            cap.release()
            return False
    except Exception as e:
        print(f"❌ YOLO + Camera integration error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 People Counting System - Quick Test")
    print("=" * 50)
    
    # Run tests
    tests = [
        ("OpenCV", test_opencv),
        ("Camera", test_camera),
        ("YOLO Model", test_yolo),
        ("YOLO + Camera", test_yolo_with_camera)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if success:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 ALL TESTS PASSED! Your People Counting System is ready to use.")
        print("\nNext steps:")
        print("1. Run 'streamlit run app.py' to start the web interface")
        print("2. Select 'Live Camera Counting' mode")
        print("3. Choose Camera 0 and start counting!")
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Please fix the issues before using the system.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
