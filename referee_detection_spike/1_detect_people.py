import cv2
from ultralytics import YOLO
import numpy as np
import random
import json
import os

# Load a pre-trained YOLO model (here we use YOLOv8n pre-trained on COCO)
model = YOLO('yolov8n.pt')  # contains 'person' class

# Open the input soccer video
cap = cv2.VideoCapture('input_soccer_clip.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define video writer to save output with bounding box
out = cv2.VideoWriter('output_people_detected.mp4', cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))

# Create frames directory if it doesn't exist
frames_dir = 'frames'
os.makedirs(frames_dir, exist_ok=True)

# List to store all detections
all_detections = []
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # end of video
    
    # Use YOLO to detect and track persons in the frame
    # Enable tracking with the 'track' argument
    results = model.track(frame, persist=True, verbose=False)  # get predictions with tracking
    
    # Store detections for this frame
    frame_detections = []
    
    # Process detections
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            if model.names[cls_id] == 'person':  # if detection is a person
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get tracking ID (if available)
                if hasattr(box, 'id') and box.id is not None:
                    track_id = int(box.id[0])
                    color = (0, 255, 0)  # bright green
                else:
                    track_id = None
                    color = (0, 255, 0)  # bright green
                
                # Draw bounding box for person
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Show tracking ID without confidence
                conf = float(box.conf[0])
                label = f"{track_id}" if track_id is not None else "/"
                
                # Improve label visibility with background and better sizing
                font_scale = 0.9  # Increased from 0.5
                font_thickness = 1
                font = cv2.FONT_HERSHEY_PLAIN
                
                # Get text size for background rectangle
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                
                # Draw background rectangle for text
                cv2.rectangle(frame, 
                             (x1, y1 - text_height - 10), 
                             (x1 + text_width + 10, y1), 
                             (0, 0, 0), 
                             -1)  # Filled rectangle
                
                # Draw text with improved visibility
                cv2.putText(frame, 
                           label, 
                           (x1 + 5, y1 - 5), 
                           font, 
                           font_scale, 
                           (255, 255, 255),  # White text
                           font_thickness, 
                           cv2.LINE_AA)
                
                # Store detection data
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'track_id': track_id,
                    'class': model.names[cls_id]
                }
                frame_detections.append(detection)
    
    # Add frame detections to master list
    all_detections.append({
        'frame_id': frame_count,
        'detections': frame_detections
    })
    frame_count += 1
    
    # Write the frame with detections
    out.write(frame)
    
    # Save frame as image file
    frame_filename = os.path.join(frames_dir, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_filename, frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Save all detections to JSON file
with open('people_detections.json', 'w') as f:
    json.dump(all_detections, f, indent=4)

print(f"Processed {frame_count} frames. Detections saved to 'people_detections.json'")
