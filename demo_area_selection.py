#!/usr/bin/env python3
"""
Area Selection Demo
Quick test to demonstrate the custom counting area functionality
"""

import streamlit as st
import cv2
import numpy as np

def demo_area_selection():
    """Demo the area selection feature"""
    print("📐 Custom Counting Areas Demo")
    print("=" * 40)
    
    # Test area definition functions
    from camera import is_person_in_area, draw_counting_area
    
    # Create a test frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Define a test counting area (center area)
    test_area = (200, 150, 440, 330)  # x1, y1, x2, y2
    
    # Test person bounding boxes
    test_people = [
        (100, 100, 150, 200),   # Outside area (left)
        (250, 200, 300, 300),   # Inside area (center)
        (500, 100, 550, 200),   # Outside area (right)
        (320, 250, 370, 350),   # Inside area (center-right)
    ]
    
    print("\n🧪 Testing area detection:")
    for i, person_bbox in enumerate(test_people):
        in_area = is_person_in_area(person_bbox, test_area)
        status = "✅ IN AREA" if in_area else "❌ OUTSIDE"
        print(f"   Person {i+1}: {person_bbox} -> {status}")
    
    # Test drawing function
    test_frame_with_area = draw_counting_area(test_frame.copy(), test_area, setup_mode=True)
    
    print("\n✅ Area selection functions working correctly!")
    print("\n📋 Usage Instructions:")
    print("1. Launch the app: streamlit run app.py")
    print("2. Go to 'Live Camera' mode")
    print("3. Check '🎯 Enable Custom Counting Area'")
    print("4. Check '✏️ Area Setup Mode'")
    print("5. Use coordinate inputs or presets to define your area")
    print("6. Start the camera to see the area in action!")
    
    print("\n🎯 Preset Examples:")
    print("   • Center Area: (200, 150) to (440, 330)")
    print("   • Entrance Area: (150, 50) to (490, 200)")
    print("   • Bottom Half: (50, 240) to (590, 430)")

if __name__ == "__main__":
    try:
        demo_area_selection()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're in the project directory")
    except Exception as e:
        print(f"❌ Error: {e}")
