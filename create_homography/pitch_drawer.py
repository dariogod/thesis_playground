import cv2
import numpy as np
import json
from typing import Tuple, Dict, Any


def get_soccer_pitch_coordinates() -> Dict[str, Any]:
    """
    Generate standard soccer pitch coordinates in meters, with (0,0) at the center.
    
    Returns:
        Dict[str, Any]: Dictionary containing all key pitch coordinates and dimensions
    """
    # Standard pitch dimensions (in meters)
    pitch_width = 105.0
    pitch_height = 68.0
    half_width = pitch_width / 2
    half_height = pitch_height / 2
    
    # Standard measurements
    penalty_area_width = 16.5
    penalty_area_height = 40.32
    goal_area_width = 5.5
    goal_area_height = 18.32
    penalty_spot_distance = 11.0
    center_circle_radius = 9.15
    corner_arc_radius = 1.0
    goal_width = 7.32  # Distance between posts
    
    # Create the dictionary of coordinates
    coordinates = {
        "pitch_dimensions": {
            "width": pitch_width,
            "height": pitch_height
        },
        "corners": {
            "top_left": [-half_width, -half_height],
            "top_right": [half_width, -half_height],
            "bottom_left": [-half_width, half_height],
            "bottom_right": [half_width, half_height]
        },
        "center": {
            "point": [0.0, 0.0],
            "circle_radius": center_circle_radius
        },
        "penalty_spots": {
            "left": [-penalty_spot_distance, 0.0],
            "right": [penalty_spot_distance, 0.0]
        },
        "penalty_areas": {
            "left": {
                "top_left": [-half_width, -penalty_area_height/2],
                "top_right": [-half_width + penalty_area_width, -penalty_area_height/2],
                "bottom_left": [-half_width, penalty_area_height/2],
                "bottom_right": [-half_width + penalty_area_width, penalty_area_height/2]
            },
            "right": {
                "top_left": [half_width - penalty_area_width, -penalty_area_height/2],
                "top_right": [half_width, -penalty_area_height/2],
                "bottom_left": [half_width - penalty_area_width, penalty_area_height/2],
                "bottom_right": [half_width, penalty_area_height/2]
            }
        },
        "goal_areas": {
            "left": {
                "top_left": [-half_width, -goal_area_height/2],
                "top_right": [-half_width + goal_area_width, -goal_area_height/2],
                "bottom_left": [-half_width, goal_area_height/2],
                "bottom_right": [-half_width + goal_area_width, goal_area_height/2]
            },
            "right": {
                "top_left": [half_width - goal_area_width, -goal_area_height/2],
                "top_right": [half_width, -goal_area_height/2],
                "bottom_left": [half_width - goal_area_width, goal_area_height/2],
                "bottom_right": [half_width, goal_area_height/2]
            }
        },
        "goals": {
            "left": {
                "post_top": [-half_width, -goal_width/2],
                "post_bottom": [-half_width, goal_width/2]
            },
            "right": {
                "post_top": [half_width, -goal_width/2],
                "post_bottom": [half_width, goal_width/2]
            }
        },
        "center_line": {
            "top": [0.0, -half_height],
            "bottom": [0.0, half_height]
        },
        "corner_arcs": {
            "radius": corner_arc_radius
        },
        "penalty_arcs": {
            "radius": center_circle_radius  # Same as center circle
        }
    }
    
    return coordinates


def create_pitch_image(image_width: int = 1920, image_height: int = 1080, bg_color: Tuple[int, int, int] = (0, 100, 0)) -> np.ndarray:
    """
    Create a blank image with background color for the pitch.
    
    Args:
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
        bg_color: Background color in BGR format
        
    Returns:
        np.ndarray: The initialized image
    """
    # Create a blank image
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    
    # Fill the entire background with green
    image[:] = bg_color
    
    return image


def calculate_pitch_dimensions(image_width: int, image_height: int, pitch_width: float = 105, pitch_height: float = 68, margin_factor: float = 0.9) -> dict:
    """
    Calculate pitch dimensions to fit the screen.
    
    Args:
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
        pitch_width: Width of the pitch in meters
        pitch_height: Height of the pitch in meters
        margin_factor: Factor to leave margin around the pitch (0.9 = 90% of screen)
        
    Returns:
        dict: Dictionary containing pitch dimensions and positions
    """
    # Calculate scale to fit pitch on screen while maintaining aspect ratio
    width_scale = image_width / pitch_width
    height_scale = image_height / pitch_height
    pitch_scale = min(width_scale, height_scale) * margin_factor
    
    # Calculate centered position
    radar_center_x = int(image_width/2)
    radar_center_y = int(image_height/2)
    radar_width = int(pitch_width * pitch_scale)
    radar_height = int(pitch_height * pitch_scale)
    radar_top_x = int(radar_center_x - radar_width / 2)
    radar_top_y = int(radar_center_y - radar_height / 2)
    
    return {
        "pitch_scale": pitch_scale,
        "radar_center_x": radar_center_x,
        "radar_center_y": radar_center_y,
        "radar_width": radar_width,
        "radar_height": radar_height,
        "radar_top_x": radar_top_x,
        "radar_top_y": radar_top_y
    }


def draw_pitch(image: np.ndarray, dimensions: dict, line_thickness: int = 2, line_color: Tuple[int, int, int] = (255, 255, 255)) -> None:
    """
    Draw a soccer pitch on the given image.
    
    Args:
        image: Image to draw on
        dimensions: Dictionary containing pitch dimensions from calculate_pitch_dimensions
        line_thickness: Thickness of the lines
        line_color: Color of the lines in BGR format
        
    Returns:
        None (modifies the image in-place)
    """
    # Extract dimensions
    pitch_scale = dimensions["pitch_scale"]
    radar_center_x = dimensions["radar_center_x"]
    radar_center_y = dimensions["radar_center_y"]
    radar_width = dimensions["radar_width"]
    radar_height = dimensions["radar_height"]
    radar_top_x = dimensions["radar_top_x"]
    radar_top_y = dimensions["radar_top_y"]
    
    # Draw field lines
    # Center line
    cv2.line(image, 
            (radar_center_x, radar_top_y), 
            (radar_center_x, radar_top_y + radar_height), 
            line_color, line_thickness)
    
    # Draw pitch outline
    cv2.rectangle(image, 
                (radar_top_x, radar_top_y), 
                (radar_top_x + radar_width, radar_top_y + radar_height), 
                line_color, line_thickness)
    
    # Center circle
    cv2.circle(image, 
            (int(radar_center_x), int(radar_center_y)), 
            int(9.15 * pitch_scale), 
            line_color, line_thickness)
    
    # Center spot
    cv2.circle(image, 
            (int(radar_center_x), int(radar_center_y)), 
            int(0.5 * pitch_scale), 
            line_color, -1)
    
    # Goal areas - 5.5m from each goal post and 5.5m into the field (total width of 18.32m)
    goal_area_width = 5.5 * pitch_scale
    goal_area_height = 18.32 * pitch_scale  # 9.16m on each side of the center
    
    # Left goal area
    left_goal_x = radar_top_x
    cv2.rectangle(image, 
                (left_goal_x, int(radar_center_y - goal_area_height/2)), 
                (int(left_goal_x + goal_area_width), int(radar_center_y + goal_area_height/2)), 
                line_color, line_thickness)
    
    # Right goal area
    right_goal_x = radar_top_x + radar_width - int(goal_area_width)
    cv2.rectangle(image, 
                (right_goal_x, int(radar_center_y - goal_area_height/2)), 
                (radar_top_x + radar_width, int(radar_center_y + goal_area_height/2)), 
                line_color, line_thickness)
    
    # Penalty areas - 16.5m from each goal post and 16.5m into the field (total width of 40.32m)
    penalty_area_width = 16.5 * pitch_scale
    penalty_area_height = 40.32 * pitch_scale  # 20.16m on each side of the center
    
    # Left penalty area
    cv2.rectangle(image, 
                (radar_top_x, int(radar_center_y - penalty_area_height/2)), 
                (int(radar_top_x + penalty_area_width), int(radar_center_y + penalty_area_height/2)), 
                line_color, line_thickness)
    
    # Right penalty area
    cv2.rectangle(image, 
                (radar_top_x + radar_width - int(penalty_area_width), int(radar_center_y - penalty_area_height/2)),
                (radar_top_x + radar_width, int(radar_center_y + penalty_area_height/2)), 
                line_color, line_thickness)
    
    # Penalty spots - 11m from goal line
    penalty_spot_distance = 11 * pitch_scale
    
    # Left penalty spot
    left_penalty_spot_x = int(radar_top_x + penalty_spot_distance)
    cv2.circle(image, 
            (left_penalty_spot_x, int(radar_center_y)), 
            int(0.5 * pitch_scale), 
            line_color, -1)
    
    # Right penalty spot
    right_penalty_spot_x = int(radar_top_x + radar_width - penalty_spot_distance)
    cv2.circle(image, 
            (right_penalty_spot_x, int(radar_center_y)), 
            int(0.5 * pitch_scale), 
            line_color, -1)
    
    # Penalty arcs - 9.15m radius from penalty spot
    penalty_arc_radius = 9.15 * pitch_scale
    
    # Left penalty arc
    cv2.ellipse(image, 
                (left_penalty_spot_x, int(radar_center_y)), 
                (int(penalty_arc_radius), int(penalty_arc_radius)),
                0, 127 + 180, 233 + 180,  # Draw only the part outside the penalty area
                line_color, line_thickness)
    
    # Right penalty arc
    cv2.ellipse(image, 
                (right_penalty_spot_x, int(radar_center_y)), 
                (int(penalty_arc_radius), int(penalty_arc_radius)),
                0, 127, 233,  # Draw only the part outside the penalty area
                line_color, line_thickness)
    
    # Corner arcs - 1m radius
    corner_radius = 1 * pitch_scale
    
    # Top-left corner
    cv2.ellipse(image, 
                (int(radar_top_x), int(radar_top_y)), 
                (int(corner_radius), int(corner_radius)),
                0, 0, 90, 
                line_color, line_thickness)
    
    # Top-right corner
    cv2.ellipse(image, 
                (int(radar_top_x + radar_width), int(radar_top_y)), 
                (int(corner_radius), int(corner_radius)),
                0, 90, 180, 
                line_color, line_thickness)
    
    # Bottom-left corner
    cv2.ellipse(image, 
                (int(radar_top_x), int(radar_top_y + radar_height)), 
                (int(corner_radius), int(corner_radius)),
                0, 270, 360, 
                line_color, line_thickness)
    
    # Bottom-right corner
    cv2.ellipse(image, 
                (int(radar_top_x + radar_width), int(radar_top_y + radar_height)), 
                (int(corner_radius), int(corner_radius)),
                0, 180, 270, 
                line_color, line_thickness)


def draw_pitch_only(pitch_scale: float = 3, line_thickness: int = 3) -> np.ndarray:
    """
    Draw only the soccer pitch without any player or analysis information.
    
    Args:
        pitch_scale: Scale factor for the pitch
        line_thickness: Thickness of lines
        
    Returns:
        np.ndarray: The visualization image containing only the pitch
    """
    # Create a blank image with green background
    image_width, image_height = 1920, 1080
    image = create_pitch_image(image_width, image_height)
    
    # Calculate pitch dimensions
    dimensions = calculate_pitch_dimensions(image_width, image_height)
    
    # Draw the pitch on the image
    draw_pitch(image, dimensions, line_thickness)
    
    return image


def save_soccer_pitch_coordinates(filepath: str) -> None:
    """
    Generate and save soccer pitch coordinates to a JSON file.
    
    Args:
        filepath: Path where to save the JSON file
    """
    coordinates = get_soccer_pitch_coordinates()
    with open(filepath, 'w') as f:
        json.dump(coordinates, f, indent=2)


if __name__ == "__main__":
    """
    Main function to generate and save a soccer pitch image.
    """
    # Generate pitch image
    pitch_image = draw_pitch_only(line_thickness=3)
    
    # Save the image
    cv2.imwrite("soccer_pitch.png", pitch_image)
    
    # Generate and save pitch coordinates as JSON
    save_soccer_pitch_coordinates("soccer_pitch_coordinates.json")
    
    # Display the image (optional - uncomment to display)
    # cv2.imshow("Soccer Pitch", pitch_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    print("Soccer pitch image created and saved as 'soccer_pitch.png'")
    print("Soccer pitch coordinates saved as 'soccer_pitch_coordinates.json'") 