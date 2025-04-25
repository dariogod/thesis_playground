import cv2
import numpy as np

# 0. Green Masking (to exclude audience and only keep field)
def mask_green_field(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result, mask

# 1. Line Segment Detection with Length and Orientation Filtering
def detect_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(gray)[0]  # Position 0 is the lines
    
    if lines is None:
        return []
    
    # Filter lines based on length and orientation
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate line length
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Calculate angle in degrees
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        
        # Keep only horizontal (near 0° or 180°) or vertical (near 90°) lines
        is_horizontal = (angle < 20) or (angle > 160)
        is_vertical = (angle > 70) and (angle < 110)
        
        # Keep only lines that are long enough (field markings) and have proper orientation
        if length > 50 and (is_horizontal or is_vertical):
            filtered_lines.append(line)
    
    return filtered_lines

# 2. Vanishing Point Estimation (RANSAC based)
def estimate_vanishing_points(lines, image_shape):
    def ransac_vanishing_point(line_segments, iterations=1000, threshold=np.deg2rad(5)):
        best_vp = None
        max_inliers = 0
        
        # Prepare line segments in the format needed
        endpoints = []
        for l in line_segments:
            x1, y1, x2, y2 = l[0]
            endpoints.append((np.array([x1, y1]), np.array([x2, y2])))
        
        if len(endpoints) < 2:
            return None
            
        endpoints = np.array(endpoints)
        
        for _ in range(iterations):
            idx = np.random.choice(len(endpoints), 2, replace=False)
            l1, l2 = endpoints[idx]

            p1 = np.cross(np.append(l1[0], 1), np.append(l1[1], 1))
            p2 = np.cross(np.append(l2[0], 1), np.append(l2[1], 1))

            vp_candidate = np.cross(p1, p2)
            if np.abs(vp_candidate[2]) < 1e-6:
                continue
            vp_candidate /= vp_candidate[2]

            angles = []
            for l in endpoints:
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

    # Separate lines by orientation
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        
        if (angle < 20) or (angle > 160):
            horizontal_lines.append(line)
        elif (angle > 70) and (angle < 110):
            vertical_lines.append(line)
    
    vp_h = ransac_vanishing_point(horizontal_lines)
    vp_v = ransac_vanishing_point(vertical_lines)

    return vp_h, vp_v

# 3. Line Assignment
def assign_lines_to_vps(lines, vp_h, vp_v):
    if vp_h is None or vp_v is None:
        return []
    
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
    if not lines or not assignments or vp_h is None or vp_v is None:
        return None
        
    height, width = image_shape[:2]

    h_lines = []
    v_lines = []
    
    # Separate horizontal and vertical lines using assignments
    for i, (line, assignment) in enumerate(zip(lines, assignments)):
        if assignment == 'horizontal':
            h_lines.append(line)
        else:
            v_lines.append(line)
    
    # Sort by distance from center to get extreme lines
    center = np.array([width/2, height/2])
    
    def distance_from_center(line):
        x1, y1, x2, y2 = line[0]
        p = np.array([(x1 + x2)/2, (y1 + y2)/2])
        return np.linalg.norm(p - center)

    h_lines_sorted = sorted(h_lines, key=distance_from_center)
    v_lines_sorted = sorted(v_lines, key=distance_from_center)

    if len(h_lines_sorted) < 2 or len(v_lines_sorted) < 2:
        return None

    # Take the outermost two lines for each direction
    top_line = h_lines_sorted[0]
    bottom_line = h_lines_sorted[-1]
    left_line = v_lines_sorted[0]
    right_line = v_lines_sorted[-1]

    rays = (top_line, bottom_line, left_line, right_line)
    return rays

# 5. Main Runner
def detect_pitch_lines(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None
        
    # Step 0: Mask the green field
    green_only, green_mask = mask_green_field(image)
    
    # Step 1: Detect lines with length and orientation filtering
    lines = detect_lines(green_only)
    if lines is None or len(lines) == 0:
        print("No lines detected!")
        return image, None
    
    # Step 2: Estimate vanishing points
    vp_h, vp_v = estimate_vanishing_points(lines, image.shape)
    if vp_h is None or vp_v is None:
        print("Could not estimate vanishing points!")
        return image, None
        
    # Step 3: Assign lines to vanishing points
    assignments = assign_lines_to_vps(lines, vp_h, vp_v)
    
    # Step 4: Estimate field rays
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
            x1, y1, x2, y2 = map(int, pt[0])
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow for field boundary

    # Create mask overlay image
    mask_overlay = image.copy()
    blue_color = np.zeros_like(image)
    blue_color[:,:,0] = 255  # Blue in BGR format
    # Apply the mask as an overlay with 50% opacity
    mask_3channel = cv2.merge([green_mask, green_mask, green_mask])
    mask_overlay = np.where(mask_3channel > 0, 
                          cv2.addWeighted(mask_overlay, 0.5, blue_color, 0.5, 0), 
                          mask_overlay)

    return output, mask_overlay

# Example Usage
if __name__ == "__main__":
    output_images = detect_pitch_lines("frames/frame_0105.jpg")
    if output_images is not None:
        output_image, mask_overlay = output_images
        cv2.imwrite("detected_pitch_lines_v3.jpg", output_image)
        cv2.imwrite("mask_overlay_v3.jpg", mask_overlay)
