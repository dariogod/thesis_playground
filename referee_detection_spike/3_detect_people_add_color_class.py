import cv2
from ultralytics import YOLO
import numpy as np
import json
import os

# Load a pre-trained YOLO model (here we use YOLOv8n pre-trained on COCO)
model = YOLO('yolov8n.pt')  # contains 'person' class

# Open the input soccer video
cap = cv2.VideoCapture('input_soccer_clip.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define video writer to save output with bounding boxes
out = cv2.VideoWriter('output_people_detected.mp4', cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))

# Create frames directory if it doesn't exist
frames_dir = 'frames'
os.makedirs(frames_dir, exist_ok=True)

# List to store all detections
all_detections = []
frame_count = 0

def get_team_color(frame, bbox):
    """Extract dominant color in top 60% of the bounding box to determine team color."""
    x1, y1, x2, y2 = bbox
    x_width = x2 - x1
    x_start = int(x1 + 0.1 * x_width)  # Start at 10% from left
    x_end = int(x2 - 0.1 * x_width)    # End at 10% from right
    top_cut = int(y1 + 0.6 * (y2 - y1))
    cropped = frame[y1:top_cut, x_start:x_end]

    if cropped.size == 0:
        return "unknown"

    hsv_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_crop)
    
    # Compute color masks with expanded ranges
    red_mask = ((h < 15) | (h > 155)) & (s > 70)  # Widened red hue range, lowered saturation threshold
    yellow_mask = ((h > 15) & (h < 40)) & (s > 70)  # Widened yellow hue range, lowered saturation threshold
    blue_mask = ((h > 85) & (h < 135)) & (s > 40)  # Widened blue hue range, lowered saturation threshold
    
    # Count the number of pixels in each mask
    red_count = np.count_nonzero(red_mask)
    yellow_count = np.count_nonzero(yellow_mask)
    blue_count = np.count_nonzero(blue_mask)

    # Return label of the dominant color
    counts = {"red": red_count, "yellow": yellow_count, "referee": blue_count}
    return max(counts, key=counts.get) if max(counts.values()) > 0 else "unknown"

while True:
    ret, frame = cap.read()
    if not ret:
        break  # end of video
    
    # Use YOLO to detect and track persons in the frame
    results = model.track(frame, persist=True, verbose=False)

    frame_detections = []
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            if model.names[cls_id] == 'person':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                track_id = int(box.id[0]) if hasattr(box, 'id') and box.id is not None else None
                
                # Determine team color from top 60% of bbox
                team = get_team_color(frame, (x1, y1, x2, y2))
                
                # Draw bounding box
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"{track_id}"
                font_scale = 0.9
                font_thickness = 1
                font = cv2.FONT_HERSHEY_PLAIN

                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                cv2.rectangle(frame, 
                              (x1, y1 - text_height - 10), 
                              (x1 + text_width + 10, y1), 
                              (0, 0, 0), -1)
                cv2.putText(frame, 
                            label, 
                            (x1 + 5, y1 - 5), 
                            font, font_scale, 
                            (255, 255, 255), font_thickness, cv2.LINE_AA)
                
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'track_id': track_id,
                    'class': model.names[cls_id],
                    'team': team
                }
                frame_detections.append(detection)

    all_detections.append({
        'frame_id': frame_count,
        'detections': frame_detections
    })
    frame_count += 1
    
    out.write(frame)
    frame_filename = os.path.join(frames_dir, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_filename, frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Save all detections to JSON
with open('people_detections_with_color_info_wider_range.json', 'w') as f:
    json.dump(all_detections, f, indent=4)

print(f"Processed {frame_count} frames. Detections saved to 'people_detections.json'")
