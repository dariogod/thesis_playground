import json
import numpy as np
import cv2
from PIL import Image
import os
from sklearn.cluster import KMeans
from ultralytics import YOLO

def get_dominant_color(frame, bbox, top_percent=0.5, middle_x_percent=0.8, n_clusters=3, pixel_threshold=50):
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    w = x2 - x1

    y_start = y1
    y_end = y1 + int(h * top_percent)
    x_start = x1 + int(w * (1 - middle_x_percent) / 2)
    x_end = x2 - int(w * (1 - middle_x_percent) / 2)

    cropped = frame[y_start:y_end, x_start:x_end]
    
    # Convert BGR to RGB and then to HSV
    rgb_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb_cropped, cv2.COLOR_RGB2HSV)

    # Initial green mask
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    non_green_pixels = rgb_cropped[mask == 0]

    # If not enough non-green pixels, relax the green mask
    if len(non_green_pixels) < pixel_threshold:
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        non_green_pixels = rgb_cropped[mask == 0]

    # Fallback if still too few
    if len(non_green_pixels) < n_clusters:
        non_green_pixels = rgb_cropped.reshape(-1, 3)

    try:
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        kmeans.fit(non_green_pixels)
        cluster_centers = kmeans.cluster_centers_.astype(int)
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        # Convert numpy int values to Python int in the tuple
        dominant_color = tuple(int(v) for v in cluster_centers[np.argmax(counts)])
    except Exception:
        dominant_color = (0, 0, 0)

    return dominant_color

# Load a pre-trained YOLO model
model = YOLO('yolov8n.pt')  # contains 'person' class

# Open the input soccer video
cap = cv2.VideoCapture('input_soccer_clip.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define video writer to save output with bounding boxes
out = cv2.VideoWriter('output_people_detected_rgb.mp4', cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))

# Create frames directory if it doesn't exist
frames_dir = 'frames_rgb'
os.makedirs(frames_dir, exist_ok=True)

# List to store all detections
all_detections = []
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # end of video
    
    # Create a copy of the original frame for color detection
    original_frame = frame.copy()
    
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
                
                # Get dominant RGB color from the original unmodified frame
                dominant_color = get_dominant_color(original_frame, (x1, y1, x2, y2))
                
                # Draw bounding box
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw a colored rectangle at the top of the bounding box to show the dominant color
                dc_bgr = (int(dominant_color[2]), int(dominant_color[1]), int(dominant_color[0]))  # Convert RGB to BGR
                cv2.rectangle(frame, (x1, y1-20), (x1+20, y1), dc_bgr, -1)
                
                label = f"{track_id}"
                font_scale = 0.9
                font_thickness = 1
                font = cv2.FONT_HERSHEY_PLAIN

                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                cv2.rectangle(frame, 
                              (x1+25, y1 - text_height - 10), 
                              (x1+25 + text_width + 10, y1), 
                              (0, 0, 0), -1)
                cv2.putText(frame, 
                            label, 
                            (x1+30, y1 - 5), 
                            font, font_scale, 
                            (255, 255, 255), font_thickness, cv2.LINE_AA)
                
                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'track_id': int(track_id) if track_id is not None else None,
                    'class': model.names[cls_id],
                    'dominant_color_rgb': [int(c) for c in dominant_color]
                }
                frame_detections.append(detection)

    all_detections.append({
        'frame_id': int(frame_count),
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
with open('people_detections_with_rgb_color.json', 'w') as f:
    json.dump(all_detections, f, indent=4)

print(f"Processed {frame_count} frames. Detections saved to 'people_detections_with_rgb_color.json'")
