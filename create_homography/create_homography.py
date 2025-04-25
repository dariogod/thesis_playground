#!/usr/bin/env python
import argparse
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import os
import sys
import math

# Add parent directory to path so we can import from invoke_gamestate
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from invoke_gamestate.visualization.pitch_drawer import (
    create_pitch_image, calculate_pitch_dimensions, draw_pitch, 
    get_pitch_coordinates, Position
)

# Standard soccer pitch dimensions in meters
PITCH_WIDTH = 105.0
PITCH_HEIGHT = 68.0
HALF_PITCH_WIDTH = PITCH_WIDTH / 2
HALF_PITCH_HEIGHT = PITCH_HEIGHT / 2

# Standard pitch elements with coordinates in meters
PITCH_ELEMENTS = {
    "center_line": [
        (0, -HALF_PITCH_HEIGHT), 
        (0, HALF_PITCH_HEIGHT)
    ],
    "left_goal_line": [
        (-HALF_PITCH_WIDTH, -HALF_PITCH_HEIGHT), 
        (-HALF_PITCH_WIDTH, HALF_PITCH_HEIGHT)
    ],
    "right_goal_line": [
        (HALF_PITCH_WIDTH, -HALF_PITCH_HEIGHT), 
        (HALF_PITCH_WIDTH, HALF_PITCH_HEIGHT)
    ],
    "top_touch_line": [
        (-HALF_PITCH_WIDTH, -HALF_PITCH_HEIGHT), 
        (HALF_PITCH_WIDTH, -HALF_PITCH_HEIGHT)
    ],
    "bottom_touch_line": [
        (-HALF_PITCH_WIDTH, HALF_PITCH_HEIGHT), 
        (HALF_PITCH_WIDTH, HALF_PITCH_HEIGHT)
    ],
    "left_penalty_area_top": [
        (-HALF_PITCH_WIDTH, -20.16), 
        (-HALF_PITCH_WIDTH + 16.5, -20.16)
    ],
    "left_penalty_area_bottom": [
        (-HALF_PITCH_WIDTH, 20.16), 
        (-HALF_PITCH_WIDTH + 16.5, 20.16)
    ],
    "right_penalty_area_top": [
        (HALF_PITCH_WIDTH, -20.16), 
        (HALF_PITCH_WIDTH - 16.5, -20.16)
    ],
    "right_penalty_area_bottom": [
        (HALF_PITCH_WIDTH, 20.16), 
        (HALF_PITCH_WIDTH - 16.5, 20.16)
    ],
    "left_penalty_area_side": [
        (-HALF_PITCH_WIDTH + 16.5, -20.16), 
        (-HALF_PITCH_WIDTH + 16.5, 20.16)
    ],
    "right_penalty_area_side": [
        (HALF_PITCH_WIDTH - 16.5, -20.16), 
        (HALF_PITCH_WIDTH - 16.5, 20.16)
    ]
}

def load_detections(json_file_path: str) -> List[dict]:
    """
    Load detection data from a JSON file.
    
    Args:
        json_file_path: Path to the JSON file containing detections
        
    Returns:
        List[dict]: The loaded JSON data
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return data

def extract_frame_data(detection_data: List[dict]) -> Dict[str, Dict]:
    """
    Extract data organized by frame from the detections.
    
    Args:
        detection_data: List of detection dictionaries
        
    Returns:
        Dict[str, Dict]: Dictionary mapping frame IDs to frame data
    """
    frame_data = {}
    
    # Check if the data structure matches our expected format
    # Process if it's a list of frames with line detections
    if isinstance(detection_data, list) and all(isinstance(item, dict) for item in detection_data):
        for i, frame in enumerate(detection_data):
            frame_id = f"frame_{i}"  # Create a frame ID if none exists
            frame_data[frame_id] = frame
    
    logging.info(f"Extracted data for {len(frame_data)} frames")
    return frame_data

def classify_lines(lines: List[dict]) -> Dict[str, List[dict]]:
    """
    Classify lines based on their orientation and position.
    
    Args:
        lines: List of detected lines
        
    Returns:
        Dict[str, List[dict]]: Dictionary mapping line types to lists of lines
    """
    classified_lines = {
        "horizontal": [],
        "vertical": [],
        "other": []
    }
    
    for line in lines:
        if "type" in line:
            line_type = line["type"]
            if line_type == "horizontal":
                classified_lines["horizontal"].append(line)
            elif line_type == "vertical":
                classified_lines["vertical"].append(line)
            else:
                classified_lines["other"].append(line)
        elif "points" in line and len(line["points"]) >= 4:
            # Classify based on points if type is not available
            x1, y1, x2, y2 = line["points"][:4]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            
            if dx > 3*dy:  # Mostly horizontal
                line["type"] = "horizontal"
                classified_lines["horizontal"].append(line)
            elif dy > 3*dx:  # Mostly vertical
                line["type"] = "vertical"
                classified_lines["vertical"].append(line)
            else:
                line["type"] = "other"
                classified_lines["other"].append(line)
    
    return classified_lines

def line_length(line: dict) -> float:
    """
    Calculate the length of a line.
    
    Args:
        line: Dictionary containing line points
        
    Returns:
        float: Length of the line
    """
    if "points" in line and len(line["points"]) >= 4:
        x1, y1, x2, y2 = line["points"][:4]
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return 0.0

def midpoint(line: dict) -> Tuple[float, float]:
    """
    Calculate the midpoint of a line.
    
    Args:
        line: Dictionary containing line points
        
    Returns:
        Tuple[float, float]: Midpoint coordinates
    """
    if "points" in line and len(line["points"]) >= 4:
        x1, y1, x2, y2 = line["points"][:4]
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    return (0.0, 0.0)

def assign_pitch_coordinates(classified_lines: Dict[str, List[dict]], 
                          image_width: int = 1280, image_height: int = 720) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Assign standard pitch coordinates to detected lines.
    
    Args:
        classified_lines: Dictionary mapping line types to lists of lines
        image_width: Width of the source image
        image_height: Height of the source image
        
    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: List of pairs of image points and pitch coordinates
    """
    # Sort lines by length (longest first)
    horizontal_lines = sorted(classified_lines["horizontal"], key=line_length, reverse=True)
    vertical_lines = sorted(classified_lines["vertical"], key=line_length, reverse=True)
    
    # Initialize point correspondences
    point_pairs = []
    
    # Try to identify the pitch elements
    identified_elements = {}
    
    # Find potential touchlines (longest horizontal lines)
    touchlines = []
    if len(horizontal_lines) >= 2:
        # Get y-coordinates of horizontal lines
        y_coords = [midpoint(line)[1] for line in horizontal_lines[:min(len(horizontal_lines), 5)]]
        # Sort by y-coordinate
        sorted_indices = np.argsort(y_coords)
        
        if len(sorted_indices) >= 2:
            # Top touchline (smallest y-coordinate)
            top_idx = sorted_indices[0]
            touchlines.append(("top_touch_line", horizontal_lines[top_idx]))
            
            # Bottom touchline (largest y-coordinate)
            bottom_idx = sorted_indices[-1]
            touchlines.append(("bottom_touch_line", horizontal_lines[bottom_idx]))
    
    # Find potential goal lines (longest vertical lines)
    goal_lines = []
    if len(vertical_lines) >= 2:
        # Get x-coordinates of vertical lines
        x_coords = [midpoint(line)[0] for line in vertical_lines[:min(len(vertical_lines), 5)]]
        # Sort by x-coordinate
        sorted_indices = np.argsort(x_coords)
        
        if len(sorted_indices) >= 2:
            # Left goal line (smallest x-coordinate)
            left_idx = sorted_indices[0]
            goal_lines.append(("left_goal_line", vertical_lines[left_idx]))
            
            # Right goal line (largest x-coordinate)
            right_idx = sorted_indices[-1]
            goal_lines.append(("right_goal_line", vertical_lines[right_idx]))
    
    # Find potential center line (vertical line near image center)
    center_line = None
    if len(vertical_lines) >= 1:
        # Find the vertical line closest to the image center
        center_x = image_width / 2
        center_dists = [(abs(midpoint(line)[0] - center_x), i) for i, line in enumerate(vertical_lines)]
        center_dists.sort()
        
        if center_dists:
            center_idx = center_dists[0][1]
            center_line = ("center_line", vertical_lines[center_idx])
    
    # Combine identified elements
    identified_elements = dict(touchlines + goal_lines)
    if center_line:
        identified_elements[center_line[0]] = center_line[1]
    
    # Create point correspondences for identified elements
    for element_name, line in identified_elements.items():
        if element_name in PITCH_ELEMENTS and "points" in line:
            # Get image points
            x1, y1, x2, y2 = line["points"][:4]
            img_points = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
            
            # Get corresponding pitch points
            pitch_points = np.array(PITCH_ELEMENTS[element_name], dtype=np.float32)
            
            point_pairs.append((img_points, pitch_points))
    
    return point_pairs

def compute_homography_from_point_pairs(point_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[np.ndarray]:
    """
    Compute homography from point pairs.
    
    Args:
        point_pairs: List of pairs of image points and pitch coordinates
        
    Returns:
        Optional[np.ndarray]: Homography matrix or None if computation fails
    """
    if not point_pairs:
        return None
    
    # Combine all point pairs
    img_points = np.vstack([pair[0] for pair in point_pairs])
    pitch_points = np.vstack([pair[1] for pair in point_pairs])
    
    if len(img_points) < 4:
        logging.warning(f"Not enough point correspondences for homography (found {len(img_points)}, need at least 4)")
        return None
    
    try:
        # Compute homography matrix (maps from image to pitch coordinates)
        H, mask = cv2.findHomography(img_points, pitch_points, cv2.RANSAC, 5.0)
        return H
    except Exception as e:
        logging.error(f"Error computing homography: {e}")
        return None

def visualize_homography_on_pitch(frame_data: dict, homography: Optional[np.ndarray], 
                              output_path: Optional[str] = None,
                              image_width: int = 1280, image_height: int = 720) -> None:
    """
    Visualize detected lines and their pitch mapping using homography.
    
    Args:
        frame_data: Dictionary containing line detections
        homography: Homography matrix (image to pitch coordinates)
        output_path: Path to save the visualization
        image_width: Width of the source image
        image_height: Height of the source image
    """
    # Create soccer pitch image
    pitch_img_width, pitch_img_height = 1920, 1080
    pitch_img = create_pitch_image(pitch_img_width, pitch_img_height)
    
    # Calculate pitch dimensions
    dimensions = calculate_pitch_dimensions(pitch_img_width, pitch_img_height)
    
    # Draw the pitch on the image
    draw_pitch(pitch_img, dimensions, line_thickness=2)
    
    # Draw detected lines on the pitch
    if "lines" in frame_data and isinstance(frame_data["lines"], list):
        # Classify lines
        classified_lines = classify_lines(frame_data["lines"])
        
        for line_type, lines in classified_lines.items():
            for line in lines:
                if "points" in line and isinstance(line["points"], list) and len(line["points"]) >= 4:
                    # Extract image points
                    x1, y1, x2, y2 = map(float, line["points"][:4])
                    
                    # Choose color based on line type
                    if line_type == "horizontal":
                        color = (0, 0, 255)  # Red
                    elif line_type == "vertical":
                        color = (0, 255, 0)  # Green
                    else:
                        color = (255, 0, 0)  # Blue
                    
                    if homography is not None:
                        # Transform image points to pitch coordinates using homography
                        img_points = np.array([[[x1, y1]], [[x2, y2]]], dtype=np.float32)
                        try:
                            pitch_points = cv2.perspectiveTransform(img_points, homography)
                            
                            # Convert pitch coordinates to visualization image coordinates
                            vis_x1, vis_y1 = get_pitch_coordinates(dimensions, pitch_points[0][0][0], pitch_points[0][0][1])
                            vis_x2, vis_y2 = get_pitch_coordinates(dimensions, pitch_points[1][0][0], pitch_points[1][0][1])
                            
                            # Draw line
                            cv2.line(pitch_img, (int(vis_x1), int(vis_y1)), (int(vis_x2), int(vis_y2)), color, 2)
                        except Exception as e:
                            logging.error(f"Error transforming points: {e}")
                    else:
                        # Without homography, use a simple mapping (like before)
                        # Normalize to pitch coordinates (approximation)
                        rel_x1 = x1 / image_width
                        rel_y1 = y1 / image_height
                        rel_x2 = x2 / image_width
                        rel_y2 = y2 / image_height
                        
                        # Map to pitch coordinates
                        pitch_x1 = (rel_x1 - 0.5) * PITCH_WIDTH
                        pitch_y1 = (rel_y1 - 0.5) * PITCH_HEIGHT
                        pitch_x2 = (rel_x2 - 0.5) * PITCH_WIDTH
                        pitch_y2 = (rel_y2 - 0.5) * PITCH_HEIGHT
                        
                        # Convert pitch coordinates to image coordinates
                        img_x1, img_y1 = get_pitch_coordinates(dimensions, pitch_x1, pitch_y1)
                        img_x2, img_y2 = get_pitch_coordinates(dimensions, pitch_x2, pitch_y2)
                        
                        # Draw line
                        cv2.line(pitch_img, (int(img_x1), int(img_y1)), (int(img_x2), int(img_y2)), color, 2)
    
    # Add a legend for line types
    legend_y = 50
    cv2.putText(pitch_img, "Line Types:", (50, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.line(pitch_img, (250, legend_y), (350, legend_y), (0, 0, 255), 2)
    cv2.putText(pitch_img, "Horizontal", (360, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.line(pitch_img, (250, legend_y + 30), (350, legend_y + 30), (0, 255, 0), 2)
    cv2.putText(pitch_img, "Vertical", (360, legend_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.line(pitch_img, (250, legend_y + 60), (350, legend_y + 60), (255, 0, 0), 2)
    cv2.putText(pitch_img, "Other", (360, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add homography status
    status_text = "Homography: "
    if homography is not None:
        status_text += "Valid"
        status_color = (0, 255, 0)  # Green
    else:
        status_text += "Not available"
        status_color = (0, 0, 255)  # Red
    
    cv2.putText(pitch_img, status_text, (50, legend_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, pitch_img)
    else:
        plt.figure(figsize=(16, 9))
        plt.imshow(cv2.cvtColor(pitch_img, cv2.COLOR_BGR2RGB))
        plt.title("Detected Lines with Homography Mapping" if homography is not None else "Detected Lines (Approximate Mapping)")
        plt.axis('off')
        plt.show()

def create_structured_json(frame_data: Dict[str, Dict], output_file: Optional[str] = None) -> Dict[str, Dict]:
    """
    Create a structured JSON with classified lines and pitch coordinates.
    
    Args:
        frame_data: Dictionary mapping frame IDs to frame data
        output_file: Path to save the structured JSON
        
    Returns:
        Dict[str, Dict]: Structured JSON data
    """
    structured_data = {}
    
    for frame_id, frame in frame_data.items():
        if "lines" in frame and isinstance(frame["lines"], list):
            # Classify lines
            classified_lines = classify_lines(frame["lines"])
            
            # Create structured frame data
            structured_frame = {
                "frame_id": frame_id,
                "lines": {
                    "horizontal": [],
                    "vertical": [],
                    "other": []
                },
                "identified_pitch_elements": []
            }
            
            # Add classified lines
            for line_type, lines in classified_lines.items():
                for line in lines:
                    if "points" in line and isinstance(line["points"], list) and len(line["points"]) >= 4:
                        x1, y1, x2, y2 = map(float, line["points"][:4])
                        structured_line = {
                            "image_points": [x1, y1, x2, y2],
                            "length": line_length(line),
                            "midpoint": midpoint(line)
                        }
                        structured_frame["lines"][line_type].append(structured_line)
            
            # Identify pitch elements and assign pitch coordinates
            point_pairs = assign_pitch_coordinates(classified_lines)
            
            # Add identified pitch elements
            for i, (img_points, pitch_points) in enumerate(point_pairs):
                element = {
                    "id": f"element_{i}",
                    "image_points": img_points.tolist(),
                    "pitch_points": pitch_points.tolist()
                }
                structured_frame["identified_pitch_elements"].append(element)
            
            structured_data[frame_id] = structured_frame
    
    # Save structured JSON if output_file is provided
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(structured_data, f, indent=2)
    
    return structured_data

def create_homographies(json_file_path: str, output_dir: Optional[str] = None) -> Dict[str, Dict]:
    """
    Process line detection data, extract pitch coordinates, and compute homographies.
    
    Args:
        json_file_path: Path to the JSON file containing detections
        output_dir: Directory to save visualizations and data
        
    Returns:
        Dict[str, Dict]: Dictionary mapping frame IDs to processed data
    """
    # Load detection data
    detection_data = load_detections(json_file_path)
    
    # Extract frame data
    frame_data = extract_frame_data(detection_data)
    
    # Create output directory if provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create structured JSON
    structured_json_path = os.path.join(output_dir, "structured_detections.json") if output_dir else None
    structured_data = create_structured_json(frame_data, structured_json_path)
    
    # Process each frame
    results = {}
    for frame_id, frame in tqdm(frame_data.items(), desc="Processing frames"):
        if frame_id not in structured_data:
            continue
        
        # Get structured frame data
        structured_frame = structured_data[frame_id]
        
        # Extract point pairs
        point_pairs = []
        for element in structured_frame["identified_pitch_elements"]:
            img_points = np.array(element["image_points"], dtype=np.float32)
            pitch_points = np.array(element["pitch_points"], dtype=np.float32)
            point_pairs.append((img_points, pitch_points))
        
        # Compute homography
        homography = compute_homography_from_point_pairs(point_pairs)
        
        # Store results
        results[frame_id] = {
            "frame_data": frame,
            "structured_data": structured_frame,
            "homography": homography.tolist() if homography is not None else None
        }
        
        # Save homography if output directory is provided and homography is valid
        if output_dir and homography is not None:
            homography_path = os.path.join(output_dir, f"{frame_id}_homography.npy")
            np.save(homography_path, homography)
        
        # Visualize lines on soccer pitch if output directory is provided
        if output_dir:
            vis_path = os.path.join(output_dir, f"{frame_id}_homography_mapping.jpg")
            visualize_homography_on_pitch(frame, homography, vis_path)
    
    # Save summary if output directory is provided
    if output_dir:
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, 'w') as f:
            summary_data = {
                "num_frames": len(results),
                "frames": [
                    {
                        "frame_id": k,
                        "num_lines": {
                            "horizontal": len(v["structured_data"]["lines"]["horizontal"]),
                            "vertical": len(v["structured_data"]["lines"]["vertical"]),
                            "other": len(v["structured_data"]["lines"]["other"])
                        },
                        "num_identified_elements": len(v["structured_data"]["identified_pitch_elements"]),
                        "has_homography": v["homography"] is not None
                    } for k, v in results.items()
                ]
            }
            json.dump(summary_data, f, indent=2)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Process line detections and compute homographies")
    parser.add_argument("json_file", help="Path to the JSON file containing detections")
    parser.add_argument("--output-dir", help="Directory to save results", default="homography_results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")
    
    # Check if input file exists
    if not os.path.isfile(args.json_file):
        logging.error(f"Input file not found: {args.json_file}")
        return
    
    # Process detections and compute homographies
    results = create_homographies(
        args.json_file,
        args.output_dir
    )
    
    # Print summary
    total_frames = len(results)
    
    if total_frames == 0:
        print("No frames were found in the input file. Please check the JSON format.")
    else:
        frames_with_homography = sum(1 for v in results.values() if v["homography"] is not None)
        
        print(f"Processed {total_frames} frames")
        print(f"Successfully computed homographies for {frames_with_homography} frames "
              f"({frames_with_homography/total_frames*100:.2f}%)")
        print(f"Results saved to: {args.output_dir}")
        print(f"\nStructured detection data: {args.output_dir}/structured_detections.json")
        print(f"Summary: {args.output_dir}/summary.json")

if __name__ == "__main__":
    main()
