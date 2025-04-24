import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
from collections import defaultdict

# Load a pre-trained YOLO model (here we use YOLOv8n pre-trained on COCO)
model = YOLO('yolov8n.pt')  # contains 'person' class
# Enable tracking in the model
model.tracker = "bytetrack.yaml"  # Using ByteTrack for object tracking

# Open the input soccer video
cap = cv2.VideoCapture('input_soccer_clip.mp4')
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

# Define video writer to save output with bounding box
out = cv2.VideoWriter('output_referee_detected.mp4', cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))

# Dictionary to store color history for each track_id
track_color_history = defaultdict(list)
# Dictionary to store referee prediction for each track_id
referee_predictions = defaultdict(int)
# Threshold for referee prediction confidence
REFEREE_THRESHOLD = 5

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # end of video
    
    frame_count += 1

    # Use YOLO to detect and track persons in the frame
    results = model.track(frame, persist=True, verbose=False)  # get predictions with tracking
    detections = []  # to store (bbox, color_feature, track_id)
    
    if results[0].boxes is not None and hasattr(results[0].boxes, 'id'):
        for box, track_id in zip(results[0].boxes, results[0].boxes.id):
            cls_id = int(box.cls[0])
            if model.names[cls_id] == 'person':  # if detection is a person
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Extract the person's bounding box region
                person_roi = frame[y1:y2, x1:x2]
                if person_roi.size == 0:
                    continue  # skip if any invalid ROI

                # Use the entire person ROI instead of just the top 60%
                # Convert to HSV color space
                hsv = cv2.cvtColor(person_roi, cv2.COLOR_BGR2HSV)
                
                # Mask out green field by creating a mask for green color range
                # Define range for green color in HSV
                lower_green = np.array([35, 50, 50])
                upper_green = np.array([85, 255, 255])
                
                # Create a mask to identify green pixels
                green_mask = cv2.inRange(hsv, lower_green, upper_green)
                # Invert the mask to get non-green pixels
                non_green_mask = cv2.bitwise_not(green_mask)
                
                # Apply the mask to get only non-green pixels
                non_green_pixels = cv2.bitwise_and(person_roi, person_roi, mask=non_green_mask)
                
                # Count non-zero pixels in the mask
                non_green_pixel_count = cv2.countNonZero(non_green_mask)
                
                # Compute mean color in BGR (only for non-green pixels)
                if non_green_pixel_count > 0:
                    # Reshape to get all non-green pixels
                    valid_pixels = non_green_pixels[non_green_mask > 0]
                    mean_color = valid_pixels.mean(axis=0)
                else:
                    # If all pixels are green, use the original mean (fallback)
                    mean_color = person_roi.reshape(-1, 3).mean(axis=0)
                
                # Add to detections with track_id
                track_id = int(track_id)
                detections.append(((x1, y1, x2, y2), mean_color, track_id))
                
                # Store color feature in track history
                track_color_history[track_id].append(mean_color)
                # Keep last 30 frames of history per track
                if len(track_color_history[track_id]) > 30:
                    track_color_history[track_id].pop(0)

    # If no people detected, just write frame and continue
    if not detections:
        out.write(frame)
        continue

    # Process each track that has enough history
    tracks_with_sufficient_history = {
        track_id: np.mean(colors, axis=0) 
        for track_id, colors in track_color_history.items() 
        if len(colors) >= 5  # Require at least 5 frames of history
    }
    
    if len(tracks_with_sufficient_history) >= 2:
        # Cluster the tracks by their average color
        track_ids = list(tracks_with_sufficient_history.keys())
        avg_colors = list(tracks_with_sufficient_history.values())
        
        n_clusters = min(3, len(avg_colors))  # up to 3 clusters (two teams & referee)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        track_labels = kmeans.fit_predict(avg_colors)
        
        # Determine which cluster is likely the referee cluster
        unique, counts = np.unique(track_labels, return_counts=True)
        # Find cluster with minimum count (smallest group)
        ref_cluster = unique[np.argmin(counts)]
        
        # Update referee predictions based on cluster assignment
        for track_id, label in zip(track_ids, track_labels):
            if label == ref_cluster:
                referee_predictions[track_id] += 1
            else:
                referee_predictions[track_id] = max(0, referee_predictions[track_id] - 1)

    # Draw bounding boxes based on tracking and referee predictions
    for (bbox, color, track_id) in detections:
        x1, y1, x2, y2 = bbox
        
        # Determine if this track is likely a referee
        is_referee = referee_predictions.get(track_id, 0) >= REFEREE_THRESHOLD
        
        if is_referee:
            # Draw a red rectangle around the referee (thicker line)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, f"Referee ID:{track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, (0,0,255), 2, cv2.LINE_AA)
        else:
            # Draw thinner boxes for players
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            # Optionally show track ID
            cv2.putText(frame, f"ID:{track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0,255,0), 1, cv2.LINE_AA)
    
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
