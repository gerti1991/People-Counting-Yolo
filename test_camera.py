"""
Quick Camera Test - Simple version to test camera connectivity
"""
import cv2
import sys

def test_camera_sources():
    """Test available camera sources"""
    print("🔍 Testing camera sources...")
    available_cameras = []
    
    # Test USB/built-in cameras (0-5)
    for i in range(6):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(i)
                print(f"✅ Camera {i}: Working ({frame.shape[1]}x{frame.shape[0]})")
            else:
                print(f"❌ Camera {i}: Detected but can't read frames")
            cap.release()
        else:
            print(f"❌ Camera {i}: Not available")
    
    return available_cameras

def run_simple_camera(camera_id=0):
    """Run simple camera display"""
    print(f"🎥 Starting camera {camera_id}...")
    print("Press 'q' to quit")
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ Failed to open camera {camera_id}")
        return False
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        # Add simple text overlay
        cv2.putText(frame, f"Camera {camera_id} - Press 'q' to quit", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(f'Camera Test - Camera {camera_id}', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Camera test completed")
    return True

if __name__ == "__main__":
    print("📷 Universal Camera Tester")
    print("=" * 40)
    
    # Test available cameras
    cameras = test_camera_sources()
    
    if not cameras:
        print("\n❌ No working cameras found!")
        print("Make sure you have a camera connected and permissions are granted.")
        sys.exit(1)
    
    print(f"\n✅ Found {len(cameras)} working camera(s): {cameras}")
    
    # Ask user which camera to test
    if len(cameras) == 1:
        camera_to_test = cameras[0]
        print(f"Using camera {camera_to_test}")
    else:
        print(f"\nAvailable cameras: {cameras}")
        try:
            camera_to_test = int(input("Enter camera ID to test: "))
            if camera_to_test not in cameras:
                print(f"Invalid camera ID. Using camera {cameras[0]}")
                camera_to_test = cameras[0]
        except ValueError:
            print(f"Invalid input. Using camera {cameras[0]}")
            camera_to_test = cameras[0]
    
    # Run camera test
    run_simple_camera(camera_to_test)
