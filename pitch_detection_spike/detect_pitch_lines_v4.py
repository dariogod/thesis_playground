import cv2
import numpy as np
import json
import os

# 0. Green Masking (to exclude audience and only keep field)
def mask_green_field(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply morphological operations to smooth the mask and remove player edges
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    result = cv2.bitwise_and(image, image, mask=mask)
    return result, mask

# 1. Line Segment Detection
def detect_lines(image, min_length=30):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(gray)[0]  # Position 0 is the lines
    
    # Filter out short lines that are likely player edges
    if lines is not None:
        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length >= min_length:
                filtered_lines.append(line)
        
        if filtered_lines:
            return np.array(filtered_lines)
    
    return lines

# 2. Vanishing Point Estimation (RANSAC based)
def estimate_vanishing_points(lines, image_shape):
    def ransac_vanishing_point(line_segments, iterations=1000, threshold=np.deg2rad(5)):
        best_vp = None
        max_inliers = 0
        for _ in range(iterations):
            idx = np.random.choice(len(line_segments), 2, replace=False)
            l1, l2 = line_segments[idx]

            p1 = np.cross(np.append(l1[0], 1), np.append(l1[1], 1))
            p2 = np.cross(np.append(l2[0], 1), np.append(l2[1], 1))

            vp_candidate = np.cross(p1, p2)
            if np.abs(vp_candidate[2]) < 1e-6:
                continue
            vp_candidate /= vp_candidate[2]

            angles = []
            for l in line_segments:
                dir_vec = l[1] - l[0]
                dir_vec /= np.linalg.norm(dir_vec)
                vp_dir = (vp_candidate[:2] - l[0])
                vp_dir /= np.linalg.norm(vp_dir)
                angle = np.arccos(np.clip(np.dot(dir_vec, vp_dir), -1.0, 1.0))
                angles.append(angle)

            angles = np.array(angles)
            inliers = np.sum(angles < threshold)
            if inliers > max_inliers:
                max_inliers = inliers
                best_vp = vp_candidate
        return best_vp

    # Convert lines to endpoints
    endpoints = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        endpoints.append((np.array([x1, y1]), np.array([x2, y2])))

    vp_h = ransac_vanishing_point(np.array(endpoints))
    vp_v = ransac_vanishing_point(np.array(endpoints))

    return vp_h, vp_v

# 3. Line Assignment
def assign_lines_to_vps(lines, vp_h, vp_v):
    assignments = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        midpoint = np.array([(x1 + x2) / 2, (y1 + y2) / 2, 1.0])

        dir_line = np.array([x2 - x1, y2 - y1])
        dir_line /= np.linalg.norm(dir_line)

        dir_vp_h = vp_h[:2] - midpoint[:2]
        dir_vp_h /= np.linalg.norm(dir_vp_h)

        dir_vp_v = vp_v[:2] - midpoint[:2]
        dir_vp_v /= np.linalg.norm(dir_vp_v)

        score_h = np.abs(np.dot(dir_line, dir_vp_h))
        score_v = np.abs(np.dot(dir_line, dir_vp_v))

        if score_h > score_v:
            assignments.append('horizontal')
        else:
            assignments.append('vertical')
    return assignments

# 4. Simple Field Ray Estimation
def estimate_field_rays(lines, assignments, vp_h, vp_v, image_shape):
    height, width = image_shape[:2]

    h_lines = [l for l, a in zip(lines, assignments) if a == 'horizontal']
    v_lines = [l for l, a in zip(lines, assignments) if a == 'vertical']

    # If too few lines are found, return None
    if len(h_lines) < 2 or len(v_lines) < 2:
        return None

    # Sort by distance from center to get extreme lines
    center = np.array([width/2, height/2])
    def distance_from_center(line):
        p = np.array([(line[0][0] + line[0][2])/2, (line[0][1] + line[0][3])/2])
        return np.linalg.norm(p - center)

    h_lines_sorted = sorted(h_lines, key=distance_from_center)
    v_lines_sorted = sorted(v_lines, key=distance_from_center)

    # Take the outermost two lines for each direction
    y1 = h_lines_sorted[0][0]
    y2 = h_lines_sorted[-1][0]
    y3 = v_lines_sorted[0][0]
    y4 = v_lines_sorted[-1][0]

    rays = (y1, y2, y3, y4)
    return rays

# 5. Main Runner
def detect_pitch_lines(image_path):
    image = cv2.imread(image_path)
    green_only, green_mask = mask_green_field(image)
    lines = detect_lines(green_only)
    if lines is None or len(lines) == 0:
        print("No lines detected!")
        return

    vp_h, vp_v = estimate_vanishing_points(lines, image.shape)
    assignments = assign_lines_to_vps(lines, vp_h, vp_v)
    rays = estimate_field_rays(lines, assignments, vp_h, vp_v, image.shape)

    # Visualization
    output = image.copy()
    for i, l in enumerate(lines):
        x1, y1, x2, y2 = map(int, l[0])
        if assignments[i] == 'horizontal':
            color = (255, 0, 0)  # Blue for horizontal
        else:
            color = (0, 0, 255)  # Red for vertical
        cv2.line(output, (x1, y1), (x2, y2), color, 2)

    # Draw rays (field boundary)
    if rays:
        for pt in rays:
            x1, y1, x2, y2 = map(int, pt)
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 255), 2)
    
    # Create mask overlay image
    mask_overlay = image.copy()
    blue_color = np.zeros_like(image)
    blue_color[:,:,0] = 255  # Blue in BGR format
    # Apply the mask as an overlay with 50% opacity
    mask_3channel = cv2.merge([green_mask, green_mask, green_mask])
    mask_overlay = np.where(mask_3channel > 0, 
                          cv2.addWeighted(mask_overlay, 0.5, blue_color, 0.5, 0), 
                          mask_overlay)

    return output, mask_overlay, lines, assignments, vp_h, vp_v, rays

# 6. Create JSON output with detections
def create_detection_json(lines, assignments, vp_h, vp_v, rays):
    detections = {
        "lines": [],
        "vanishing_points": {
            "horizontal": vp_h.tolist() if vp_h is not None else None,
            "vertical": vp_v.tolist() if vp_v is not None else None
        },
        "field_rays": []
    }
    
    # Add lines with their type (horizontal/vertical)
    if lines is not None:
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = map(float, line[0])
            detections["lines"].append({
                "points": [x1, y1, x2, y2],
                "type": assignments[i]
            })
    
    # Add field rays if detected
    if rays is not None:
        for ray in rays:
            x1, y1, x2, y2 = map(float, ray)
            detections["field_rays"].append([x1, y1, x2, y2])
    
    return detections

# 7. Process Video File
def process_video(video_path, output_folder="output"):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer for output videos
    lines_video = cv2.VideoWriter(
        os.path.join(output_folder, 'lines_output.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )
    
    mask_video = cv2.VideoWriter(
        os.path.join(output_folder, 'mask_output.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )
    
    # Store detections for all frames
    all_detections = []
    
    frame_idx = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
            
        print(f"Processing frame {frame_idx}/{total_frames}")
        
        # Process the frame
        green_only, green_mask = mask_green_field(frame)
        lines = detect_lines(green_only)
        
        if lines is not None and len(lines) > 0:
            vp_h, vp_v = estimate_vanishing_points(lines, frame.shape)
            assignments = assign_lines_to_vps(lines, vp_h, vp_v)
            rays = estimate_field_rays(lines, assignments, vp_h, vp_v, frame.shape)
            
            # Visualization
            output = frame.copy()
            for i, l in enumerate(lines):
                x1, y1, x2, y2 = map(int, l[0])
                if assignments[i] == 'horizontal':
                    color = (255, 0, 0)  # Blue for horizontal
                else:
                    color = (0, 0, 255)  # Red for vertical
                cv2.line(output, (x1, y1), (x2, y2), color, 2)
            
            # Draw rays (field boundary)
            if rays:
                for pt in rays:
                    x1, y1, x2, y2 = map(int, pt)
                    cv2.line(output, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            # Create mask overlay image
            mask_overlay = frame.copy()
            blue_color = np.zeros_like(frame)
            blue_color[:,:,0] = 255  # Blue in BGR format
            # Apply the mask as an overlay with 50% opacity
            mask_3channel = cv2.merge([green_mask, green_mask, green_mask])
            mask_overlay = np.where(mask_3channel > 0, 
                                cv2.addWeighted(mask_overlay, 0.5, blue_color, 0.5, 0), 
                                mask_overlay)
            
            # Write frames to output videos
            lines_video.write(output)
            mask_video.write(mask_overlay)
            
            # Create and store JSON detections for this frame
            detections = create_detection_json(lines, assignments, vp_h, vp_v, rays)
            detections["frame_index"] = frame_idx
            all_detections.append(detections)
        else:
            # If no lines detected, just write the original frame
            lines_video.write(frame)
            
            # Create basic mask overlay with just the green mask
            mask_overlay = frame.copy()
            blue_color = np.zeros_like(frame)
            blue_color[:,:,0] = 255
            mask_3channel = cv2.merge([green_mask, green_mask, green_mask])
            mask_overlay = np.where(mask_3channel > 0, 
                                cv2.addWeighted(mask_overlay, 0.5, blue_color, 0.5, 0), 
                                mask_overlay)
            mask_video.write(mask_overlay)
            
            # Add empty detection for this frame
            all_detections.append({
                "frame_index": frame_idx,
                "lines": [],
                "vanishing_points": {"horizontal": None, "vertical": None},
                "field_rays": []
            })
        
        frame_idx += 1
    
    # Release resources
    video.release()
    lines_video.release()
    mask_video.release()
    
    # Save all detections to a JSON file
    with open(os.path.join(output_folder, "all_detections.json"), "w") as f:
        json.dump(all_detections, f, indent=2)
    
    print(f"Processing complete. Results saved to {output_folder} folder.")

# Example Usage
if __name__ == "__main__":
    # Process video
    process_video("input_soccer_clip.mp4")