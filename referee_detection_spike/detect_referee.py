import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans

# Load a pre-trained YOLO model (here we use YOLOv8n pre-trained on COCO)
model = YOLO('yolov8n.pt')  # contains 'person' class

# Open the input soccer video
cap = cv2.VideoCapture('input_soccer_clip.mp4')
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

# Define video writer to save output with bounding box
out = cv2.VideoWriter('output_referee_detected.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break  # end of video

    # Use YOLO to detect persons in the frame
    results = model(frame, verbose=False)  # get predictions
    detections = []  # to store (bbox, color_feature)
    for r in results:
        for box in r.boxes:
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
                
                detections.append(((x1, y1, x2, y2), mean_color))

    # If no people detected, just write frame and continue
    if not detections:
        out.write(frame)
        continue

    # Cluster the detected people by color (if at least 2 detections)
    colors = [det[1] for det in detections]
    n_clusters = min(3, len(colors))  # up to 3 clusters (two teams & referee)
    labels = [0] * len(colors)
    if len(colors) >= 2:
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        labels = kmeans.fit_predict(colors)

    # Determine which cluster is likely the referee cluster
    # We assume the referee cluster will have the fewest members (distinct color)
    unique, counts = np.unique(labels, return_counts=True)
    # Find cluster with minimum count (smallest group)
    ref_cluster = unique[np.argmin(counts)]

    # Draw bounding box for any detection in the referee cluster
    for (bbox, color), label in zip(detections, labels):
        x1, y1, x2, y2 = bbox
        if label == ref_cluster:
            # Draw a red rectangle around the referee (thicker line)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, "Referee", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, (0,0,255), 2, cv2.LINE_AA)
        else:
            # (Optional) draw thinner boxes for players for illustration
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    out.write(frame)

# Release resources
cap.release()
out.release()
