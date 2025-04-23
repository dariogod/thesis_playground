import cv2
import numpy as np
import os
from detection import PersonDetector
from tracking import TrackManager
from color_clustering import RefereeIdentifier
from pitch_detection import PitchDetector
from minimap import Minimap
from export_json import JSONExporter

def process_video(video_path, output_json=None, visualize=True):
    # Initialize components
    detector = PersonDetector()
    tracker = TrackManager()
    referee_identifier = RefereeIdentifier()
    pitch_detector = PitchDetector()
    minimap = Minimap()
    exporter = JSONExporter()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get first frame for pitch detection
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return
    
    # Detect pitch lines and set up homography
    lines = pitch_detector.detect_lines(first_frame)
    # This is a simplified example - in a real scenario, you would need to map 
    # image points to actual pitch coordinates
    image_points = np.float32([[0, 0], [first_frame.shape[1], 0], 
                              [first_frame.shape[1], first_frame.shape[0]], 
                              [0, first_frame.shape[0]]])
    pitch_points = np.float32([[0, 0], [105, 0], [105, 68], [0, 68]])
    homography = pitch_detector.find_homography(image_points, pitch_points)
    
    # Add pitch info to exporter
    if output_json:
        exporter.add_pitch_info(homography, lines.tolist() if lines is not None else [])
    
    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    frame_idx = 0
    color_features = []
    detections_with_colors = []
    
    # Process first few frames to collect color data
    while frame_idx < 10:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect people
        detections = detector.detect(frame)
        
        # Extract colors
        for d in detections:
            color = referee_identifier.extract_color(frame, d['bbox'])
            if not np.isnan(color).any():
                color_features.append(color)
                detections_with_colors.append(d)
                
        frame_idx += 1
    
    # Cluster jerseys if we have enough data
    if len(color_features) > 2:
        # Reshape color_features to ensure it's a 2D array with shape (n_samples, n_features)
        color_features_array = np.array(color_features).reshape(len(color_features), -1)
        labels, centers = referee_identifier.cluster_jerseys(color_features_array)
        
        # Identify referee
        detections_with_roles = referee_identifier.identify_referee(
            detections_with_colors, color_features, labels)
    
    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    
    # Process all frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect people
        detections = detector.detect(frame)
        
        # Update tracks
        tracked_objects = tracker.update(detections)
        
        # Process tracked objects
        processed_objects = []
        for idx, obj in enumerate(tracked_objects):
            # Extract object data
            obj_data = {
                'id': int(obj.id),
                'bbox': [int(coord) for coord in [obj.estimate[0][0] - 50, obj.estimate[0][1] - 100, 
                                                obj.estimate[0][0] + 50, obj.estimate[0][1]]]
            }
            
            # Determine if referee based on color (simplified)
            # In a real implementation, you would need to match this with the previously identified referee
            if idx < len(detections) and 'role' in detections[idx]:
                obj_data['role'] = detections[idx]['role']
            else:
                obj_data['role'] = 'player'
                
            # Project to field coordinates using homography
            center_point = np.array([[(obj.estimate[0][0] + obj.estimate[0][0]) / 2, 
                                      obj.estimate[0][1]]], dtype=np.float32)
            center_point = center_point.reshape(-1, 1, 2)
            field_coord = cv2.perspectiveTransform(center_point, homography)
            field_coord = field_coord.flatten().tolist()
            obj_data['field_coord'] = field_coord
            
            processed_objects.append(obj_data)
        
        # Add frame to exporter
        if output_json:
            exporter.add_frame(frame_idx, processed_objects)
        
        # Visualize if requested
        if visualize:
            # Draw bounding boxes and roles on frame
            for obj in processed_objects:
                x1, y1, x2, y2 = obj['bbox']
                color = (0, 0, 255) if obj['role'] == 'referee' else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID: {obj['id']}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw minimap
            minimap_canvas = minimap.draw_pitch()
            positions = [obj['field_coord'] for obj in processed_objects if obj['field_coord']]
            minimap_canvas = minimap.draw_players(minimap_canvas, positions)
            
            # Resize minimap to fit in corner of frame
            minimap_h, minimap_w = 150, 250
            minimap_resized = cv2.resize(minimap_canvas, (minimap_w, minimap_h))
            
            # Overlay minimap on frame
            frame[30:30+minimap_h, 30:30+minimap_w] = minimap_resized
            
            # Display frame
            cv2.imshow('Referee Detection', frame)
            if cv2.waitKey(30) & 0xFF == 27:  # ESC to exit
                break
                
        frame_idx += 1
    
    # Save JSON if requested
    if output_json:
        exporter.save(output_json)
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    
    # Check if a video path was provided as an argument
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "input_soccer_clip.mp4"
        
    if os.path.exists(video_path):
        process_video(video_path, output_json="output.json", visualize=True)
    else:
        print(f"Error: Video file {video_path} not found")
        print("Please provide a valid video file path as an argument") 