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

# 1. Line Segment Detection
def detect_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(gray)[0]  # Position 0 is the lines
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

    # Sort by distance from center to get extreme lines
    center = np.array([width/2, height/2])
    def distance_from_center(line):
        p = np.array([(line[0][0] + line[0][2])/2, (line[0][1] + line[0][3])/2])
        return np.linalg.norm(p - center)

    h_lines_sorted = sorted(h_lines, key=distance_from_center)
    v_lines_sorted = sorted(v_lines, key=distance_from_center)

    if len(h_lines_sorted) < 2 or len(v_lines_sorted) < 2:
        return None

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
    if lines is None:
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

    return output, mask_overlay

# Example Usage
if __name__ == "__main__":
    output_images = detect_pitch_lines("frames/frame_0105.jpg")
    if output_images is not None:
        output_image, mask_overlay = output_images
        cv2.imwrite("detected_pitch_lines_v2.jpg", output_image)
        cv2.imwrite("mask_overlay_v2.jpg", mask_overlay)