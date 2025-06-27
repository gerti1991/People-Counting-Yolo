import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov9c.pt")

# Define prediction functions
def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)
    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes, conf=conf)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    return img, results

def process_video(video_path, output_path, model, confidence=0.5, progress_callback=None):
    """
    Process video with advanced people counting similar to live camera functionality
    
    Args:
        video_path: Path to input video
        output_path: Path to output video
        model: YOLO model
        confidence: Detection confidence threshold
        progress_callback: Function to call with progress updates
    
    Returns:
        dict: Processing results with total count, max count, etc.
    """
    import numpy as np
    import time
    
    video = cv2.VideoCapture(video_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Video writer
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Tracking variables
    unique_people_count = 0
    max_people_in_frame = 0
    total_detections = 0
    frame_counts = []
    centroids_history = []
    detection_delay_frames = fps * 5  # 5 seconds delay
    frame_number = 0
    
    # For tracking unique people with simple centroid tracking
    tracked_people = []
    next_person_id = 1
    
    def calculate_distance(point1, point2):
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def is_new_person(centroid, tracked_people, threshold=100):
        """Check if centroid represents a new person"""
        for person in tracked_people:
            if calculate_distance(centroid, person['last_centroid']) < threshold:
                person['last_centroid'] = centroid
                person['last_seen_frame'] = frame_number
                return False, person['id']
        return True, None
    
    def cleanup_old_tracks(tracked_people, current_frame, max_gap=fps*2):
        """Remove people who haven't been seen for a while"""
        return [p for p in tracked_people if current_frame - p['last_seen_frame'] < max_gap]
    
    print("Starting video processing with people counting...")
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        frame_number += 1
        
        # Run detection
        results = model(frame, classes=[0], conf=confidence, verbose=False)
        current_frame_count = 0
        current_centroids = []
        
        # Process detections
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    current_frame_count += 1
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    # Calculate centroid
                    centroid_x = int((x1 + x2) / 2)
                    centroid_y = int((y1 + y2) / 2)
                    centroid = (centroid_x, centroid_y)
                    current_centroids.append(centroid)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # Draw centroid
                    cv2.circle(frame, centroid, 5, (255, 0, 0), -1)
                    
                    # Add confidence label
                    label = f"Person: {conf:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Track unique people (only after delay period for better accuracy)
        if frame_number > detection_delay_frames:
            for centroid in current_centroids:
                is_new, person_id = is_new_person(centroid, tracked_people)
                if is_new:
                    tracked_people.append({
                        'id': next_person_id,
                        'first_seen_frame': frame_number,
                        'last_seen_frame': frame_number,
                        'last_centroid': centroid
                    })
                    unique_people_count += 1
                    next_person_id += 1
            
            # Cleanup old tracks
            tracked_people = cleanup_old_tracks(tracked_people, frame_number)
        
        # Update statistics
        max_people_in_frame = max(max_people_in_frame, current_frame_count)
        total_detections += current_frame_count
        frame_counts.append(current_frame_count)
        
        # Add overlay information
        overlay_height = 120
        cv2.rectangle(frame, (10, 10), (400, overlay_height), (0, 0, 0), -1)
        
        # Current frame info
        cv2.putText(frame, f"People in frame: {current_frame_count}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Max people seen: {max_people_in_frame}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if frame_number > detection_delay_frames:
            cv2.putText(frame, f"Unique people: {unique_people_count}", (20, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            remaining_delay = (detection_delay_frames - frame_number) / fps
            cv2.putText(frame, f"Counting starts in: {remaining_delay:.1f}s", (20, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        cv2.putText(frame, f"Frame: {frame_number}/{total_frames}", (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Write frame to output video
        out.write(frame)
        
        # Progress callback
        if progress_callback and frame_number % 30 == 0:  # Update every 30 frames
            progress = frame_number / total_frames
            progress_callback(progress, {
                'current_frame': frame_number,
                'total_frames': total_frames,
                'current_count': current_frame_count,
                'max_count': max_people_in_frame,
                'unique_count': unique_people_count if frame_number > detection_delay_frames else 0
            })
    
    # Release resources
    video.release()
    out.release()
    
    # Calculate final statistics
    avg_people_per_frame = np.mean(frame_counts) if frame_counts else 0
    
    results = {
        'output_path': output_path,
        'total_frames_processed': frame_number,
        'unique_people_count': unique_people_count,
        'max_people_in_frame': max_people_in_frame,
        'average_people_per_frame': avg_people_per_frame,
        'total_detections': total_detections,
        'video_duration_seconds': frame_number / fps,
        'detection_delay_seconds': 5,
        'confidence_threshold': confidence
    }
    
    print(f"Video processing completed!")
    print(f"Unique people counted: {unique_people_count}")
    print(f"Max people in single frame: {max_people_in_frame}")
    print(f"Average people per frame: {avg_people_per_frame:.1f}")
    
    return results

    # video.release()
    # out.release()
    # cv2.destroyAllWindows()
