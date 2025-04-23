import cv2
import numpy as np
import argparse


def create_mask(frame):
    """Create a mask for the green pitch area."""
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range for green color - wider range to capture more of the pitch
    lower_green = np.array([30, 30, 40])
    upper_green = np.array([100, 255, 255])
    
    # Create mask
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask


def enhance_lines(frame, mask):
    """Enhance the visibility of white lines on the pitch."""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply mask to focus on pitch area
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(masked_gray)
    
    # Apply threshold to highlight white lines
    _, binary = cv2.threshold(enhanced, 160, 255, cv2.THRESH_BINARY)
    
    # Remove small noise
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return binary


def detect_lines(frame, mask):
    """Detect lines on the pitch using Hough transform."""
    # Enhance line visibility
    line_img = enhance_lines(frame, mask)
    
    # Apply edge detection on the enhanced image
    edges = cv2.Canny(line_img, 30, 100, apertureSize=3)
    
    # Apply Hough Line Transform with more sensitive parameters
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=50,  # Lower threshold to detect more lines
        minLineLength=50,  # Shorter min line length
        maxLineGap=100  # Larger gap allowed
    )
    
    return lines, edges


def filter_lines(lines, frame_height, frame_width):
    """Filter lines based on orientation and location."""
    if lines is None:
        return [], []
    
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate line angle
        if x2 - x1 == 0:  # Avoid division by zero
            angle = 90
        else:
            angle = abs(np.degrees(np.arctan((y2 - y1) / (x2 - x1))))
        
        # Classify as horizontal or vertical with more lenient thresholds
        if angle < 30:  # More lenient threshold for horizontal
            horizontal_lines.append(line[0])
        elif angle > 60:  # More lenient threshold for vertical
            vertical_lines.append(line[0])
    
    # Cluster and merge similar lines to avoid duplicates
    horizontal_lines = merge_similar_lines(horizontal_lines)
    vertical_lines = merge_similar_lines(vertical_lines)
    
    return horizontal_lines, vertical_lines


def merge_similar_lines(lines, distance_threshold=30, angle_threshold=10):
    """Merge lines that are similar in position and angle."""
    if not lines:
        return []
    
    # Convert to numpy array for easier manipulation
    lines_array = np.array(lines)
    
    # Cluster similar lines
    merged_lines = []
    processed = np.zeros(len(lines), dtype=bool)
    
    for i in range(len(lines)):
        if processed[i]:
            continue
            
        processed[i] = True
        current_line = lines[i]
        x1, y1, x2, y2 = current_line
        
        # Calculate current line properties
        if x2 - x1 == 0:  # Avoid division by zero
            current_angle = 90
            current_dist = x1  # Distance from origin for vertical lines
        else:
            current_angle = np.degrees(np.arctan((y2 - y1) / (x2 - x1)))
            # Use perpendicular distance from origin for horizontal lines
            current_dist = (y1 + y2) / 2
        
        # Find similar lines
        similar_indices = []
        
        for j in range(i+1, len(lines)):
            if processed[j]:
                continue
                
            x1j, y1j, x2j, y2j = lines[j]
            
            # Calculate angle and distance for comparison
            if x2j - x1j == 0:
                angle_j = 90
                dist_j = x1j
            else:
                angle_j = np.degrees(np.arctan((y2j - y1j) / (x2j - x1j)))
                dist_j = (y1j + y2j) / 2
            
            # Check if lines are similar
            if (abs(current_angle - angle_j) < angle_threshold and 
                abs(current_dist - dist_j) < distance_threshold):
                similar_indices.append(j)
                processed[j] = True
        
        # If there are similar lines, merge them
        if similar_indices:
            # Include the current line in the group
            group = [i] + similar_indices
            
            # Extract points from all lines in the group
            all_points = np.vstack([
                [lines[idx][0:2], lines[idx][2:4]] for idx in group
            ])
            
            # For vertical lines, sort by y
            if abs(current_angle) > 45:
                sorted_points = all_points[np.argsort(all_points[:, 1])]
                merged_line = [sorted_points[0, 0], sorted_points[0, 1], 
                              sorted_points[-1, 0], sorted_points[-1, 1]]
            # For horizontal lines, sort by x
            else:
                sorted_points = all_points[np.argsort(all_points[:, 0])]
                merged_line = [sorted_points[0, 0], sorted_points[0, 1], 
                              sorted_points[-1, 0], sorted_points[-1, 1]]
                
            merged_lines.append(merged_line)
        else:
            merged_lines.append(current_line)
    
    return merged_lines


def draw_lines(frame, horizontal_lines, vertical_lines):
    """Draw the detected lines on the frame."""
    line_image = frame.copy()
    
    # Draw horizontal lines in blue
    for line in horizontal_lines:
        x1, y1, x2, y2 = line
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # Draw vertical lines in red
    for line in vertical_lines:
        x1, y1, x2, y2 = line
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    return line_image


def process_video(video_path, output_path=None, display=True):
    """Process video to detect pitch lines."""
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer if output path is provided
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create mask for pitch
        mask = create_mask(frame)
        
        # Detect lines
        lines, edges = detect_lines(frame, mask)
        
        # Filter and classify lines
        horizontal_lines, vertical_lines = filter_lines(lines, frame_height, frame_width)
        
        # Print vertical lines for debugging
        print(vertical_lines)
        
        # Draw lines on the frame
        result_frame = draw_lines(frame, horizontal_lines, vertical_lines)
        
        # Display additional debug info
        if display:
            # Resize for display
            mask_display = cv2.resize(mask, (frame_width//2, frame_height//2))
            edges_display = cv2.resize(edges, (frame_width//2, frame_height//2))
            
            # Create debug view
            cv2.imshow('Mask', mask_display)
            cv2.imshow('Edges', edges_display)
            cv2.imshow('Pitch Line Detection', result_frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    
    # Release resources
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect soccer pitch lines from a video.')
    parser.add_argument('input', help='Path to the input video file.')
    parser.add_argument('--output', help='Path to save the output video file. If not provided, no output will be saved.')
    parser.add_argument('--no-display', action='store_true', help='Do not display video while processing.')
    
    args = parser.parse_args()
    
    process_video(args.input, args.output, not args.no_display)
