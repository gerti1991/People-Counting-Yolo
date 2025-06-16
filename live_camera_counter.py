import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
from datetime import datetime
import argparse
import sys

class UniversalCameraCounter:
    def __init__(self, camera_source=0, model_path="yolov9c.pt", confidence=0.5):
        """
        Universal Camera People Counter
        
        Args:
            camera_source: Can be:
                - 0, 1, 2... for USB/built-in cameras
                - IP camera URL (e.g., 'http://192.168.1.100:8080/video')
                - RTSP stream (e.g., 'rtsp://username:password@ip:port/stream')
                - Video file path for testing
            model_path: Path to YOLO model file
            confidence: Detection confidence threshold
        """
        self.camera_source = camera_source
        self.model = YOLO(model_path)
        self.confidence = confidence
        
        # Tracking variables
        self.current_count = 0
        self.total_count = 0
        self.people_history = []
        self.last_message_time = 0
        self.message_interval = 5  # Send message every 5 seconds
        
        # Camera setup
        self.cap = None
        self.running = False
        
    def setup_camera(self):
        """Setup camera connection - works with any camera type"""
        try:
            print(f"Connecting to camera: {self.camera_source}")
            self.cap = cv2.VideoCapture(self.camera_source)
            
            # Set camera properties for better performance
            if isinstance(self.camera_source, int):  # USB/built-in camera
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Test camera connection
            ret, frame = self.cap.read()
            if not ret:
                raise Exception("Failed to read from camera")
                
            print(f"Camera connected successfully!")
            print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")
            return True
            
        except Exception as e:
            print(f"Error setting up camera: {e}")
            return False
    
    def detect_people(self, frame):
        """Detect people in frame and return count and annotated frame"""
        # Run YOLO detection
        results = self.model(frame, classes=[0], conf=self.confidence, verbose=False)
        
        people_detected = []
        annotated_frame = frame.copy()
        
        for result in results:
            for box in result.boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Calculate centroid
                centroid_x = int((x1 + x2) / 2)
                centroid_y = int((y1 + y2) / 2)
                
                # Draw centroid
                cv2.circle(annotated_frame, (centroid_x, centroid_y), 5, (255, 0, 0), -1)
                
                # Add confidence text
                label = f"Person: {confidence:.2f}"
                cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                people_detected.append({
                    'bbox': (x1, y1, x2, y2),
                    'centroid': (centroid_x, centroid_y),
                    'confidence': confidence
                })
        
        return len(people_detected), annotated_frame, people_detected
    
    def send_count_message(self, count):
        """Send message about people count - can be extended for notifications"""
        current_time = time.time()
        
        # Only send message if enough time has passed
        if current_time - self.last_message_time >= self.message_interval:
            timestamp = datetime.now().strftime("%H:%M:%S")
            message = f"[{timestamp}] 👥 People detected: {count}"
            print(message)
            
            # You can extend this to send notifications:
            # - Email notifications
            # - SMS alerts
            # - Push notifications
            # - Log to file
            # - Send to API endpoint
            
            self.last_message_time = current_time
    
    def add_info_overlay(self, frame, count, fps):
        """Add information overlay to frame"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Live People Counter", (20, 35), font, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Current Count: {count}", (20, 60), font, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 85), font, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Camera: {self.camera_source}", (20, 105), font, 0.5, (200, 200, 200), 1)
        
        # Add instructions
        cv2.putText(frame, "Press 'q' to quit, 's' to screenshot", (20, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def save_screenshot(self, frame, count):
        """Save screenshot with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/screenshot_{timestamp}_count{count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")
    
    def run(self):
        """Main loop for live camera people counting"""
        if not self.setup_camera():
            return False
        
        self.running = True
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        print("\n🎥 Starting live people counting...")
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to take screenshot")
        print("  - Press 'r' to reset total count")
        print("-" * 50)
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                # Detect people
                count, annotated_frame, people_data = self.detect_people(frame)
                self.current_count = count
                
                # Update total count (you can modify this logic)
                if count > 0:
                    self.total_count = max(self.total_count, count)
                
                # Send message if people detected
                if count > 0:
                    self.send_count_message(count)
                
                # Calculate FPS
                fps_counter += 1
                if fps_counter >= 30:  # Update FPS every 30 frames
                    fps_end_time = time.time()
                    current_fps = fps_counter / (fps_end_time - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Add overlay information
                display_frame = self.add_info_overlay(annotated_frame, count, current_fps)
                
                # Display frame
                cv2.imshow('Universal Camera People Counter', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_screenshot(display_frame, count)
                elif key == ord('r'):
                    self.total_count = 0
                    print("Total count reset!")
        
        except KeyboardInterrupt:
            print("\nStopping camera...")
        
        finally:
            self.cleanup()
        
        return True
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Camera stopped and resources cleaned up.")

def main():
    parser = argparse.ArgumentParser(description='Universal Camera People Counter')
    parser.add_argument('--camera', default=0, 
                       help='Camera source: 0,1,2... for USB cameras, URL for IP cameras, file path for videos')
    parser.add_argument('--model', default='yolov9c.pt', 
                       help='Path to YOLO model file')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Detection confidence threshold (0.0-1.0)')
    
    args = parser.parse_args()
    
    # Convert camera source to int if it's a number
    camera_source = args.camera
    try:
        camera_source = int(camera_source)
    except ValueError:
        pass  # Keep as string for IP cameras or file paths
    
    print("🚀 Universal Camera People Counter")
    print("=" * 50)
    print(f"Camera Source: {camera_source}")
    print(f"Model: {args.model}")
    print(f"Confidence: {args.confidence}")
    print("=" * 50)
    
    # Create and run counter
    counter = UniversalCameraCounter(
        camera_source=camera_source,
        model_path=args.model,
        confidence=args.confidence
    )
    
    success = counter.run()
    
    if success:
        print("✅ Program completed successfully!")
    else:
        print("❌ Program failed to start!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
